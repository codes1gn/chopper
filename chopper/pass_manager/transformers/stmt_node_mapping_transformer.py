import ast
import astunparse
from astunparse import printer
from typing import Tuple, List, Optional
import torch

from chopper.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase
from chopper.pass_manager.symbol_table import feed_forward_symbol_table, SymbolEntry

from mlir import astnodes
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
    Dimension,
    RankedTensorType,
    NoneType,
    DenseElementsAttr,
)
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp, TCF_ExpOp
from chopper.scaffold.mlir_dialects.dialect_atir import (
    ATIR_ConstOp,
    ATIR_IdentityOp,
    ATIR_NegateOp,
    ATIR_AddOp,
    ATIR_SubOp,
    ATIR_MulOp,
    ATIR_ExpOp,
    ATIR_TanhOp,
    ATIR_MatmulOp,
    ATIR_Conv2DChannelFirstOp,
    ATIR_ConstShapeOp,
    ATIR_TransposeOp,
    UnitTensorType,
)

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId
MlirType = astnodes.FloatTypeEnum

__all__ = [
    "StmtNodeMappingTransformer",
]


class StmtNodeMappingTransformer(NodeTransformerBase):
    """This is class that mapping python ast stmt node to MLIR ast node.

    Transformer python ast stmt node to MLIR ast node
    by setattr "mast_node" respectively.

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self):
        """initialize the StatementConversionPass class."""
        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Method that constructs the FunctionDef's corresponding MLIR node.

        Construct MLIR node by set FunctionDef attribute "mast_node".

        Args:
            node (ast.AST): FunctionDef node of python stmt.

        Returns:
            ast.AST: FunctionDef node with corresponding MLIR ast node.
        """

        super().generic_visit(node)
        # print(self.__str__(),
        #       "Map transformer::handling visit_FunctionDef on node.\n")
        # print(">>>>>Python FunctionDef Node:<<<<<\n", astunparse.dump(node))
        # print(node._torch_dsl_arg_annotations)

        _block = astnodes.Block(label=None, body=[])
        _region = astnodes.Region(body=[_block])
        _name = astnodes.SymbolRefId(value=node.name)
        _args = []
        # only handles arguments > 0
        if len(node.args.args) > 0:
            # print(len(node.args.args))
            func_args = node.args.args
            for arg_index in range(len(func_args)):
                arg = func_args[arg_index]
                # TODO move this hardcode into base
                # case 1, arguments is float type
                _arg_id = feed_forward_symbol_table.lookup(arg.arg).get_name()
                _type = feed_forward_symbol_table.lookup(arg.arg).get_type()
                if _arg_id == "self":
                    # HARDCODE + WORKAROUND, this is a temperal handle to avoid runtime error in type conversion by IREE Runtime
                    continue

                if hasattr(arg.annotation, "id") and arg.annotation.id == "float":
                    assert 0, "wait for tuning"
                    _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=None))
                # case 2, arguments is list type
                elif hasattr(arg.annotation, "id") and arg.annotation.id == "list":
                    assert 0, "wait for tuning"
                    # TODO: list -> <?xf32> <?x?xf32> <?x?x?f32>
                    _type = astnodes.RankedTensorType(
                        dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32)
                    )
                    _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=_type))
                elif (
                    isinstance(arg.annotation, ast.Subscript)
                    and isinstance(arg.annotation.slice, ast.Index)
                    and arg.annotation.value.id == "List"
                ):
                    assert 0, "wait for tuning"
                    # TODO: List[float] -> <?xf32> <?x?xf32> <?x?x?f32>
                    if arg.annotation.slice.value.id == "float":
                        # TODO: Other type, only support float now
                        _type = astnodes.RankedTensorType(
                            dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32)
                        )
                        _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=_type))
                # use pattern match to replace if-elif-else branching, since misplace
                # the sequence of each block may result in failed catching
                elif isinstance(arg, ast.arg):
                    _args.append(NamedArgument(name=MlirSsaId(value=arg.arg, op_no=None), type=_type))
                else:
                    # TODO: Other type
                    assert 0, "not supported scenario, please check the inputs"
                    pass
        else:
            # None Arguments still need a empty NamedArgument
            _args.append(NamedArgument())

        _result_type = feed_forward_symbol_table.lookup("ReturnTypeForFunctionDef").get_type()
        # print(feed_forward_symbol_table)
        # print(_result_type)
        # assert 0, "bad"
        _attributes = None

        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=_result_type,
            region=_region,
            attributes=_attributes,
        )
        _function_wrapper = astnodes.Operation(result_list=[], op=_function, location=None)

        # handle backward
        _autodiff_op = ReturnOperation(match=1)
        _autodiff_op.values = [_.name for _ in _args]
        _autodiff_op.types = [_.type for _ in _args]

        _autodiff_wrapper = [astnodes.Operation(result_list=None, op=_autodiff_op, location=None)]

        setattr(node, "mast_node", _function_wrapper)
        setattr(node, "mast_node_autodiff", _autodiff_wrapper)
        # print("\n ++++ show MLIR node ++++ \n", node.mast_node.dump())
        # print(node.mast_node_autodiff[0].dump())

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Module's corresponding MLIR node.

        Construct MLIR node by set Module attribute "mast_node".

        Args:
            node (ast.AST): Module node of python ast.

        Returns:
            ast.AST: Module node with corresponding MLIR ast node.
        """

        super().generic_visit(node)
        # print(self.__str__(),
        #       "Map transformer::handling visit_Module on node.\n")
        # print(">>>>>Python Module Node:<<<<<\n", astunparse.dump(node))
        _name = None
        _attributes = None

        # Construct the autodiff block for Function Op
        # FunctionOp <=> ReturnOp
        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=_name, attributes=_attributes, region=_out_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        setattr(node, "mast_node", _mlirfile)
        # setattr(node, "mast_node_autodiff", _mlirfile)
        # print("\n>>>>>MLIR Node for Module:<<<<<\n",
        #       self.pretty_mlir(node.mast_node))

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Return corresponding MLIR node.

        Construct MLIR node by set Return attribute mast_node.

        Args:
            node (ast.AST): Module node of python ast.

        Returns:
            ast.AST: Module node with corresponding MLIR ast node.
        """

        super().generic_visit(node)
        # print(self.__str__(),
        #       "Map transformer::handling visit_Return on node.\n")
        # print(">>>>>Python Return Node:<<<<<\n", astunparse.dump(node))

        # if return value is not None, set match=1
        match = 0
        if node.value:
            match = 1
        _returnop = ReturnOperation(match)
        _returnop.values = node.value
        _values = list()
        _types = list()

        # two case of ReturnOp:
        # 1. return 1.0 -> return a specific value
        # 2. return var -> return the value specified by a variable
        if isinstance(node.value, ast.Constant):
            _value = "ret" + str(match)
            _op_no = None

            _values.append(MlirSsaId(value=_value, op_no=_op_no))
            if isinstance(node.value.value, float):
                _types.append(astnodes.FloatType(MlirType.f32))
            else:
                _types = None
            _returnop.values = _values
            _returnop.types = _types

        if isinstance(node.value, ast.Name):
            _value = node.value.id
            _type = feed_forward_symbol_table.lookup(_value).get_type()
            _op_no = None

            _values.append(MlirSsaId(value=_value, op_no=_op_no))
            _types.append(_type)
            _returnop.values = _values
            _returnop.types = _types

        _returnop_wrapper = astnodes.Operation(result_list=None, op=_returnop, location=None)

        # Construct the autodiff block for Return Op
        # ReturnOp <=> FunctionOp
        _args_with_activations = feed_forward_symbol_table.lookup("ActivationSaveForAutodiff").get_type()
        _args = [NamedArgument(name=_values[idx], type=_types[idx])
            for idx in range(len(_values))]

        # merge grad at output + activations at inputs, as the total inputs for bp
        _args.append(_args_with_activations)
        _block = astnodes.Block(label=None, body=[])
        _region = astnodes.Region(body=[_block])
        _attributes = None
        _name = astnodes.SymbolRefId(value="bpfunction")
        _autodiff_ret_type = feed_forward_symbol_table.lookup("AutodiffFuncReturnType").get_type()
        _autodiff_op = astnodes.Function(
            name=_name,
            args=_args,
            result_types=_autodiff_ret_type,
            region=_region,
            attributes=_attributes,
        )
        _autodiff_wrapper = [astnodes.Operation(result_list=[], op=_autodiff_op, location=None)]

        setattr(node, "mast_node", _returnop_wrapper)
        setattr(node, "mast_node_autodiff", _autodiff_wrapper)
        # print("\n ++++ show MLIR node ++++ \n", node.mast_node.dump())
        # print(node.mast_node_autodiff[0].dump())

        return node

    def visit_AugAssign(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Assign  corresponding MLIR node.

        Construct MLIR node by set Assign astnode  attribute mast_node.

        Args:
            node (ast.AST): Assign astnode of python

        Returns:
            ast.AST: Assign astnode of python with mast_node attributions.
        """
        super().generic_visit(node)
        # print(self.__str__(),
        #       "Map transformer::handling visit_AugAssign on node.\n")
        # print(">>>>>Python AugAssign Node:<<<<<\n", astunparse.dump(node))

        _type = None
        _namespace = "tcf"
        _name = None
        if isinstance(node.op, ast.Add):
            _name = "add"
        elif isinstance(node.op, ast.LShift):
            _name = "lshift"
        elif isinstance(node.op, ast.RShift):
            _name = "rshift"
        elif isinstance(node.op, ast.Sub):
            _name = "sub"
        elif isinstance(node.op, ast.Mult):
            _name = "mul"
        elif isinstance(node.op, ast.Div):
            _name = "div"
        elif isinstance(node.op, ast.Mod):
            _name = "mod"
        elif isinstance(node.op, ast.Pow):
            _name = "pow"
        elif isinstance(node.op, ast.BitOr):
            _name = "bitor"
        elif isinstance(node.op, ast.BitXor):
            _name = "bitxor"
        elif isinstance(node.op, ast.BitAnd):
            _name = "bitand"
        elif isinstance(node.op, ast.FloorDiv):
            _name = "floordiv"
        else:
            pass

        _args = list()

        # * binary op have two condition:
        # 1. scalar binary op
        # TODO: 2. list binart op, via numpy to implement
        _SsaId_left = _SsaId_right = None
        _SsaId_left = MlirSsaId(value=node.target.id, op_no=None)
        _SsaId_right = MlirSsaId(value=node.target.id, op_no=None)
        _args.extend([_SsaId_left, _SsaId_right])

        _argument_types = [_type, _type]
        _result_types = [_type]
        _type_binop = FunctionType(argument_types=_argument_types, result_types=_result_types)

        _assignop = CustomOperation(namespace=_namespace, name=_name, args=_args, type=_type_binop)

        _result_list = list()
        _result_list.append(astnodes.OpResult(value=MlirSsaId(value=node.target.id, op_no=None), count=None))
        _assignop_wrapper = astnodes.Operation(result_list=_result_list, op=_assignop, location=None)
        setattr(node, "mast_node", _assignop_wrapper)
        # print(">>>>>MLIR Node for AugAssign BinOp:<<<<<\n",
        #       self.pretty_mlir(node.mast_node))

        return node

    def visit_Assign(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Assign  corresponding MLIR node.

        Construct MLIR node by set Assign astnode  attribute mast_node.

        Args:
            node (ast.AST): Assign astnode of python

        Returns:
            ast.AST: Assign astnode of python with mast_node attributions.
        """

        super().generic_visit(node)
        # print(self.__str__(),
        #       "Map transformer::handling visit_Assign on node.\n")
        # print(">>>>>Python Assign Node:<<<<<\n", astunparse.dump(node))
        # print(type(node.value))
        _type = None

        if isinstance(node.value, ast.Call):
            _call_method = node.value.func.attr
            # split logics for unary and binary
            # TODO change it into pattern match or calltype(node.value) -> unary | binary | cmp or other
            if _call_method == "exp" or _call_method == "tanh":

                # build arguments
                _argname = node.value.args[0].id
                _SsaId_operand = MlirSsaId(value=_argname, op_no=None)

                # build arguments types
                _arg_type_entry = feed_forward_symbol_table.lookup(_argname)
                assert _arg_type_entry is not None, "expected valid symbol entry, found None"
                _argument_types = [_arg_type_entry.get_type()]

                # build targets and their types
                _res_argnames_list = [target.id for target in node.targets]
                _result_list = [
                    astnodes.OpResult(value=MlirSsaId(value=_res_argname, op_no=None), count=None)
                    for _res_argname in _res_argnames_list
                ]
                # _result_types = [feed_forward_symbol_table.lookup(_res_argname).get_type() for _res_argname in _res_argnames_list]

                # build function types for this operation
                # _whole_op_type_def = FunctionType(argument_types=_argument_types, result_types=_result_types)

                # build mlir.op according to ast.op
                if _call_method == "exp":
                    _assign_op = ATIR_ExpOp(match=0, operand=_SsaId_operand, type=FunctionType(argument_types=_argument_types, result_types=_argument_types))
                    # TODO
                    _autodiff_op = ATIR_ExpOp(
                            match=0,
                            operand=MlirSsaId(value=_SsaId_operand.value + "_activation", op_no=None),
                            type=FunctionType(argument_types=_argument_types, result_types=_argument_types),
                    )
                    _autodiff_result = [astnodes.OpResult(value=_SsaId_operand, count=None)]
                    _autodiff_wrapper = [astnodes.Operation(result_list=_autodiff_result, op=_autodiff_op, location=None)]
                elif _call_method == "tanh":
                    _assign_op = ATIR_TanhOp(match=0, operand=_SsaId_operand, type=FunctionType(argument_types=_argument_types, result_types=_argument_types))
                    # TODO
                    _autodiff_intermediate_result_0 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_operand.value+"_0", op_no=None), count=None)]
                    _autodiff_op_0 = ATIR_TanhOp(
                            match=0,
                            operand=MlirSsaId(value=_SsaId_operand.value + "_activation", op_no=None),
                            type=FunctionType(argument_types=_argument_types, result_types=_argument_types),
                    )
                    _autodiff_wrapper_0 = astnodes.Operation(result_list=_autodiff_intermediate_result_0, op=_autodiff_op_0, location=None,)
                    _autodiff_op_1 = ATIR_MulOp(
                            match=0,
                            operand_a=_autodiff_intermediate_result_0,
                            operand_b=_autodiff_intermediate_result_0,
                            dtype=FunctionType(argument_types=[*_argument_types, *_argument_types], result_types=_argument_types),
                    )
                    _autodiff_intermediate_result_1 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_operand.value+"_1", op_no=None), count=None)]
                    _autodiff_intermediate_result_2 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_operand.value+"_2", op_no=None), count=None)]
                    _autodiff_result = [astnodes.OpResult(value=_SsaId_operand, count=None)]
                    _autodiff_wrapper_1 = astnodes.Operation(result_list=_autodiff_intermediate_result_1, op=_autodiff_op_1, location=None,)
                    # print(_autodiff_wrapper_1.dump())
                    _type_result_2 = UnitTensorType(element_type=astnodes.FloatType(MlirType.f32))
                    # anchor
                    _autodiff_wrapper_2 = astnodes.Operation(
                        result_list=_autodiff_intermediate_result_2,
                        op=ATIR_ConstOp(
                            match=0,
                            value=1.0,
                            dtype=_type_result_2,
                        ),
                        location=None,
                    )
                    """
                    _autodiff_wrapper_2 = astnodes.Operation(
                        result_list=_autodiff_intermediate_result_2,
                        op=DenseElementsAttr(
                                attribute=astnodes.FloatAttr(
                                    value=1.0,
                                    type=None,
                                ),
                                type=_type_result_2,
                            ),
                        location=None,
                    )
                            ConstantOperation(
                                match=0,
                                value=DenseElementsAttr(
                                    attribute=astnodes.FloatAttr(
                                        value=1.0,
                                        type=None,
                                    ),
                                    type=None
                                ),
                                type=_type_result_2,
                            ),
                    """
                    # print(_autodiff_wrapper_2.dump())
                    # anchor
                    # assert 0
                    _type_result_3 = FunctionType(argument_types=[_type_result_2, *_argument_types], result_types=_argument_types)
                    _op_3 = ATIR_SubOp(
                        match=0,
                        operand_a=_autodiff_intermediate_result_2,
                        operand_b=_autodiff_intermediate_result_1,
                        dtype=_type_result_3,
                    )
                    _autodiff_wrapper_3 = astnodes.Operation(
                            result_list=_autodiff_result,
                            op=_op_3,
                            location=None,
                    )
                    # print(_autodiff_wrapper_3.dump())
                    # assert 0
                    _autodiff_wrapper = [
                            _autodiff_wrapper_3,
                            _autodiff_wrapper_2,
                            _autodiff_wrapper_1,
                            _autodiff_wrapper_0,
                            ]
                    # anchor
                    # assert 0

                else:
                    assert 0, "unsupported unary op"

                _assign_op_wrapper = astnodes.Operation(result_list=_result_list, op=_assign_op, location=None)
                setattr(node, "mast_node", _assign_op_wrapper)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                # print("\n ++++ show MLIR node ++++ \n", node.mast_node_autodiff[0].dump())

            elif (
                _call_method == "add"
                or _call_method == "sub"
                or _call_method == "mul"
                or _call_method == "linear"
                or _call_method == "matmul"
                or _call_method == "conv2d"
            ):

                # build arguments
                _lhs_argname = node.value.args[0].id
                _rhs_argname = node.value.args[1].id
                _SsaId_lhs_operand = MlirSsaId(value=_lhs_argname, op_no=None)
                _SsaId_rhs_operand = MlirSsaId(value=_rhs_argname, op_no=None)

                # build arguments types
                _lhs_type_entry = feed_forward_symbol_table.lookup(_lhs_argname)
                _rhs_type_entry = feed_forward_symbol_table.lookup(_rhs_argname)
                assert _lhs_type_entry is not None, "expected valid symbol entry at lhs arg, found None"
                assert _rhs_type_entry is not None, "expected valid symbol entry at rhs arg, found None"
                _lhs_type = _lhs_type_entry.get_type()
                _rhs_type = _rhs_type_entry.get_type()
                _argument_types = [_lhs_type, _rhs_type]

                # build targets and their types
                _res_argnames_list = [target.id for target in node.targets]
                _result_list = [
                    astnodes.OpResult(value=MlirSsaId(value=_res_argname, op_no=None), count=None)
                    for _res_argname in _res_argnames_list
                ]
                _result_types = [
                    feed_forward_symbol_table.lookup(_res_argname).get_type() for _res_argname in _res_argnames_list
                ]

                # build function types for this operation
                _whole_op_type_def = FunctionType(argument_types=_argument_types, result_types=_result_types)

                # build mlir.op according to ast.op
                if _call_method == "add":
                    _assign_op = ATIR_AddOp(
                        match=0,
                        operand_a=_SsaId_lhs_operand,
                        operand_b=_SsaId_rhs_operand,
                        dtype=_whole_op_type_def,
                    )
                    _autodiff_lhs_op = ATIR_IdentityOp(
                            match=0,
                            operand=MlirSsaId(value=node.targets[0].id),
                            type=FunctionType(argument_types=_result_types, result_types=[_lhs_type]),
                    )
                    _autodiff_rhs_op = ATIR_IdentityOp(
                            match=0,
                            operand=MlirSsaId(value=node.targets[0].id),
                            type=FunctionType(argument_types=_result_types, result_types=[_rhs_type]),
                    )
                    _autodiff_result_lhs = [astnodes.OpResult(value=_SsaId_lhs_operand, count=None)]
                    _autodiff_result_rhs = [astnodes.OpResult(value=_SsaId_rhs_operand, count=None)]
                    _autodiff_wrapper_lhs = [astnodes.Operation(result_list=_autodiff_result_lhs, op=_autodiff_lhs_op, location=None)]
                    _autodiff_wrapper_rhs = [astnodes.Operation(result_list=_autodiff_result_rhs, op=_autodiff_rhs_op, location=None)]
                    setattr(node, "mast_node_autodiff_lhs", _autodiff_wrapper_lhs)
                    setattr(node, "mast_node_autodiff_rhs", _autodiff_wrapper_rhs)
                elif _call_method == "sub":
                    _assign_op = ATIR_SubOp(
                        match=0,
                        operand_a=_SsaId_lhs_operand,
                        operand_b=_SsaId_rhs_operand,
                        dtype=_whole_op_type_def,
                    )
                    _autodiff_lhs_op = ATIR_IdentityOp(
                            match=0,
                            operand=MlirSsaId(value=node.targets[0].id),
                            type=FunctionType(argument_types=_result_types, result_types=[_lhs_type]),
                    )
                    _autodiff_rhs_op = ATIR_NegateOp(
                            match=0,
                            operand=MlirSsaId(value=node.targets[0].id),
                            type=FunctionType(argument_types=_result_types, result_types=[_rhs_type]),
                    )
                    _autodiff_result_lhs = [astnodes.OpResult(value=_SsaId_lhs_operand, count=None)]
                    _autodiff_result_rhs = [astnodes.OpResult(value=_SsaId_rhs_operand, count=None)]
                    _autodiff_wrapper_lhs = [astnodes.Operation(result_list=_autodiff_result_lhs, op=_autodiff_lhs_op, location=None)]
                    _autodiff_wrapper_rhs = [astnodes.Operation(result_list=_autodiff_result_rhs, op=_autodiff_rhs_op, location=None)]
                    setattr(node, "mast_node_autodiff_lhs", _autodiff_wrapper_lhs)
                    setattr(node, "mast_node_autodiff_rhs", _autodiff_wrapper_rhs)
                elif _call_method == "mul":
                    _assign_op = ATIR_MulOp(
                        match=0,
                        operand_a=_SsaId_lhs_operand,
                        operand_b=_SsaId_rhs_operand,
                        dtype=_whole_op_type_def,
                    )
                    _SsaId_left_activation = MlirSsaId(value=_lhs_argname + "_activation", op_no=None)
                    _SsaId_right_activation = MlirSsaId(value=_rhs_argname + "_activation", op_no=None)
                    _autodiff_lhs_op = ATIR_MulOp(
                            match=0,
                            operand_a=MlirSsaId(value=node.targets[0].id),
                            operand_b=_SsaId_right_activation,
                            dtype=FunctionType(argument_types=[_result_types[0], _rhs_type], result_types=[_lhs_type]),
                    )
                    _autodiff_rhs_op = ATIR_MulOp(
                            match=0,
                            operand_a=_SsaId_left_activation,
                            operand_b=MlirSsaId(value=node.targets[0].id),
                            dtype=FunctionType(argument_types=[_lhs_type, _result_types[0]], result_types=[_rhs_type]),
                    )
                    _autodiff_result_lhs = [astnodes.OpResult(value=_SsaId_lhs_operand, count=None)]
                    _autodiff_result_rhs = [astnodes.OpResult(value=_SsaId_rhs_operand, count=None)]
                    _autodiff_wrapper_lhs = [astnodes.Operation(result_list=_autodiff_result_lhs, op=_autodiff_lhs_op, location=None)]
                    _autodiff_wrapper_rhs = [astnodes.Operation(result_list=_autodiff_result_rhs, op=_autodiff_rhs_op, location=None)]
                    setattr(node, "mast_node_autodiff_lhs", _autodiff_wrapper_lhs)
                    setattr(node, "mast_node_autodiff_rhs", _autodiff_wrapper_rhs)
                elif _call_method == "linear" or _call_method == "matmul":
                    _assign_op = ATIR_MatmulOp(
                        match=0,
                        operand_a=_SsaId_lhs_operand,
                        operand_b=_SsaId_rhs_operand,
                        dtype=_whole_op_type_def,
                    )

                    _SsaId_left_activation = MlirSsaId(value=_lhs_argname + "_activation", op_no=None)
                    _SsaId_right_activation = MlirSsaId(value=_rhs_argname + "_activation", op_no=None)

                    # build the const shape as transpose operand
                    # %a_0 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
                    _type_result_0 = RankedTensorType(dimensions=[Dimension(2)], element_type=astnodes.SignlessIntegerType(32))
                    _autodiff_lhs_intermediate_result_0 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_lhs_operand.value+"_0", op_no=None), count=None)]
                    _autodiff_wrapper_lhs_0 = astnodes.Operation(
                        result_list=_autodiff_lhs_intermediate_result_0,
                        op=ATIR_ConstShapeOp(
                            match=0,
                            value=[1, 0],
                            dtype=_type_result_0,
                        ),
                        location=None,
                    )
                    print(_autodiff_wrapper_lhs_0.dump())

                    # build the transpose op
                    # %a_1 = "tosa.transpose" (%b_activation, %a_0) : (tensor<3x4xf32>, tensor<2xi32>) -> tensor<4x3xf32>
                    _rhs_type_transposed = RankedTensorType(
                        dimensions=[
                            _rhs_type.dimensions[1],
                            _rhs_type.dimensions[0],
                        ],
                        element_type=_rhs_type.element_type,
                    )
                    _autodiff_lhs_intermediate_result_1 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_lhs_operand.value+"_1", op_no=None), count=None)]
                    _autodiff_wrapper_lhs_1 = astnodes.Operation(
                        result_list=_autodiff_lhs_intermediate_result_1,
                        op=ATIR_TransposeOp(
                            match=0,
                            operand_a=_SsaId_right_activation,
                            operand_b=_autodiff_lhs_intermediate_result_0,
                            dtype=FunctionType(argument_types=[_rhs_type, _type_result_0], result_types=[_rhs_type_transposed])
                        ),
                        location=None,
                    )
                    print(_autodiff_wrapper_lhs_1.dump())


                    _autodiff_result_lhs = [astnodes.OpResult(value=_SsaId_lhs_operand, count=None)]
                    _autodiff_lhs_op = ATIR_MatmulOp(
                            match=0,
                            operand_a=MlirSsaId(value=node.targets[0].id),
                            operand_b=_autodiff_lhs_intermediate_result_1,
                            dtype=FunctionType(argument_types=[_result_types[0], _rhs_type_transposed], result_types=[_lhs_type]),
                    )
                    _autodiff_wrapper_lhs_2 = astnodes.Operation(result_list=_autodiff_result_lhs, op=_autodiff_lhs_op, location=None)
                    print(_autodiff_wrapper_lhs_2.dump())
                    _autodiff_wrapper_lhs = [
                        _autodiff_wrapper_lhs_2,
                        _autodiff_wrapper_lhs_1,
                        _autodiff_wrapper_lhs_0,
                    ]
                    setattr(node, "mast_node_autodiff_lhs", _autodiff_wrapper_lhs)

                    # build the const shape as transpose operand
                    # %a_0 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
                    _type_result_0 = RankedTensorType(dimensions=[Dimension(2)], element_type=astnodes.SignlessIntegerType(32))
                    _autodiff_rhs_intermediate_result_0 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_rhs_operand.value+"_0", op_no=None), count=None)]
                    _autodiff_wrapper_rhs_0 = astnodes.Operation(
                        result_list=_autodiff_rhs_intermediate_result_0,
                        op=ATIR_ConstShapeOp(
                            match=0,
                            value=[1, 0],
                            dtype=_type_result_0,
                        ),
                        location=None,
                    )
                    print(_autodiff_wrapper_rhs_0.dump())

                    # build the transpose op
                    # %a_1 = "tosa.transpose" (%b_activation, %a_0) : (tensor<3x4xf32>, tensor<2xi32>) -> tensor<4x3xf32>
                    _lhs_type_transposed = RankedTensorType(
                        dimensions=[
                            _lhs_type.dimensions[1],
                            _lhs_type.dimensions[0],
                        ],
                        element_type=_lhs_type.element_type,
                    )
                    _autodiff_rhs_intermediate_result_1 = [astnodes.OpResult(value=MlirSsaId(value=_SsaId_rhs_operand.value+"_1", op_no=None), count=None)]
                    _autodiff_wrapper_rhs_1 = astnodes.Operation(
                        result_list=_autodiff_rhs_intermediate_result_1,
                        op=ATIR_TransposeOp(
                            match=0,
                            operand_a=_SsaId_left_activation,
                            operand_b=_autodiff_rhs_intermediate_result_0,
                            dtype=FunctionType(argument_types=[_lhs_type, _type_result_0], result_types=[_lhs_type_transposed])
                        ),
                        location=None,
                    )
                    print(_autodiff_wrapper_rhs_1.dump())

                    _autodiff_result_rhs = [astnodes.OpResult(value=_SsaId_rhs_operand, count=None)]
                    _autodiff_rhs_op = ATIR_MatmulOp(
                            match=0,
                            operand_a=_autodiff_rhs_intermediate_result_1,
                            operand_b=MlirSsaId(value=node.targets[0].id),
                            dtype=FunctionType(argument_types=[_lhs_type_transposed, _result_types[0]], result_types=[_rhs_type]),
                    )
                    _autodiff_wrapper_rhs_2 = astnodes.Operation(result_list=_autodiff_result_rhs, op=_autodiff_rhs_op, location=None)
                    print(_autodiff_wrapper_rhs_2.dump())

                    _autodiff_wrapper_rhs = [
                        _autodiff_wrapper_rhs_2,
                        _autodiff_wrapper_rhs_1,
                        _autodiff_wrapper_rhs_0,
                    ]
                    setattr(node, "mast_node_autodiff_rhs", _autodiff_wrapper_rhs)

                elif _call_method == "conv2d":
                    # TODO + WORKAROUND + HARDCODE, need to renaming
                    _assign_op = ATIR_Conv2DChannelFirstOp(
                        match=0,
                        activation=_SsaId_lhs_operand,
                        kernel=_SsaId_rhs_operand,
                        dtype=_whole_op_type_def,
                    )
                else:
                    assert 0, "unsupported binary op"

                _assign_op_wrapper = astnodes.Operation(result_list=_result_list, op=_assign_op, location=None)
                setattr(node, "mast_node", _assign_op_wrapper)
                # print("\n ++++ show MLIR node ++++ \n", node.mast_node.dump())
                # print(node.mast_node_autodiff_lhs[0].dump())
                # print(node.mast_node_autodiff_rhs[0].dump())

            else:
                print("Not Support this op conversion from ast.AST -> mlir.astnodes.Node")

            return node

        elif isinstance(node.value, ast.Num):
            assert 0, "Guardian: not verified"
            if isinstance(node.value.n, float):
                _match = 0
                _value = node.value.n

                # _type = astnodes.FloatType(MlirType.f32)
                _type = None

                _assignop = ConstantOperation(match=_match, value=_value, type=_type)

                _result_list = list()

                _value = node.targets[0].id
                _SsaId = MlirSsaId(value=_value, op_no=None)
                _result_list.append(astnodes.OpResult(value=_SsaId, count=None))

                _assignop_wrapper = astnodes.Operation(result_list=_result_list, op=_assignop, location=None)
                # print(">>>>>MLIR Node for Assign:<<<<<\n",
                #       self.pretty_mlir(_assignop_wrapper))
                setattr(node, "mast_node", _assignop_wrapper)
            else:
                assert 0, "found non-float value, not supported"

        elif isinstance(node.value, ast.Constant):
            assert 0, "Guardian: not verified"
            if isinstance(node.value.value, float):
                _match = 0
                _value = node.value.value

                # _type = astnodes.FloatType(MlirType.f32)
                _type = None

                _assignop = ConstantOperation(match=_match, value=_value, type=_type)

                _result_list = list()

                _value = node.targets[0].id
                _SsaId = MlirSsaId(value=_value, op_no=None)
                _result_list.append(astnodes.OpResult(value=_SsaId, count=None))

                _assignop_wrapper = astnodes.Operation(result_list=_result_list, op=_assignop, location=None)
                # print(">>>>>MLIR Node for Assign:<<<<<\n",
                #       self.pretty_mlir(_assignop_wrapper))
                setattr(node, "mast_node", _assignop_wrapper)
            else:
                assert 0, "found non-float value, not supported"

        elif isinstance(node.value, ast.BinOp):
            # TODO tweak this functionality passed by hardcoded,
            # replaced by MAPPING ENUMS of ATIR_OPS in future
            # * binary op have two condition:
            # 1. scalar binary op
            # 2. list binart op, via numpy to implement

            # STEP 1 build SsaID for lhs and rhs
            if isinstance(node.value.left, ast.Call):
                assert 0, "not support CallOp as operand of BinOp"
                _lhs_argname = node.value.left.args[0].id
                _rhs_argname = node.value.right.args[0].id
            else:
                _lhs_argname = node.value.left.id
                _rhs_argname = node.value.right.id
            _SsaId_left = MlirSsaId(value=_lhs_argname, op_no=None)
            _SsaId_right = MlirSsaId(value=_rhs_argname, op_no=None)
            _SsaId_left_activation = MlirSsaId(value=_lhs_argname + "_activation", op_no=None)
            _SsaId_right_activation = MlirSsaId(value=_rhs_argname + "_activation", op_no=None)
            _res_argnames_list = [target.id for target in node.targets]

            # STEP 2 build op arg types
            # TODO make a op_builder to simplify the building process ast.AST => astnodes.Node
            _lhs_type = feed_forward_symbol_table.lookup(_lhs_argname).get_type()
            _rhs_type = feed_forward_symbol_table.lookup(_rhs_argname).get_type()
            _argument_types = [_lhs_type, _lhs_type]
            _result_types = [feed_forward_symbol_table.lookup(_res_argname).get_type() for _res_argname in _res_argnames_list]
            _whole_op_type_def = FunctionType(argument_types=_argument_types, result_types=_result_types)

            # STEP 3 build result symbol
            _result_list = list()
            _result_list.append(astnodes.OpResult(value=MlirSsaId(value=node.targets[0].id, op_no=None), count=None))

            # STEP 4 build op according to py.op
            if isinstance(node.value.op, ast.Add):
                _op = ATIR_AddOp(
                        match=0,
                        operand_a=_SsaId_left,
                        operand_b=_SsaId_right,
                        dtype=_whole_op_type_def,
                        )
                _autodiff_lhs_op = ATIR_IdentityOp(
                        match=0,
                        operand=MlirSsaId(value=node.targets[0].id),
                        type=FunctionType(argument_types=_result_types, result_types=[_lhs_type]),
                        )
                _autodiff_rhs_op = ATIR_IdentityOp(
                        match=0,
                        operand=MlirSsaId(value=node.targets[0].id),
                        type=FunctionType(argument_types=_result_types, result_types=[_rhs_type]),
                        )
            elif isinstance(node.value.op, ast.Sub):
                _op = ATIR_SubOp(
                        match=0,
                        operand_a=_SsaId_left,
                        operand_b=_SsaId_right,
                        dtype=_whole_op_type_def,
                        )
                _autodiff_lhs_op = ATIR_IdentityOp(
                        match=0,
                        operand=MlirSsaId(value=node.targets[0].id),
                        type=FunctionType(argument_types=_result_types, result_types=[_lhs_type]),
                        )
                _autodiff_rhs_op = ATIR_NegateOp(
                        match=0,
                        operand=MlirSsaId(value=node.targets[0].id),
                        type=FunctionType(argument_types=_result_types, result_types=[_rhs_type]),
                        )
            elif isinstance(node.value.op, ast.Mult):
                _op = ATIR_MulOp(
                        match=0,
                        operand_a=_SsaId_left,
                        operand_b=_SsaId_right,
                        dtype=_whole_op_type_def,
                        )
                _autodiff_lhs_op = ATIR_MulOp(
                        match=0,
                        operand_a=MlirSsaId(value=node.targets[0].id),
                        operand_b=_SsaId_right_activation,
                        dtype=FunctionType(argument_types=[_result_types[0], _rhs_type], result_types=[_lhs_type]),
                        )
                _autodiff_rhs_op = ATIR_MulOp(
                        match=0,
                        operand_a=_SsaId_left_activation,
                        operand_b=MlirSsaId(value=node.targets[0].id),
                        dtype=FunctionType(argument_types=[_lhs_type, _result_types[0]], result_types=[_rhs_type]),
                        )
            elif isinstance(node.value.op, ast.Mod):
                assert 0
            elif isinstance(node.value.op, ast.Pow):
                assert 0
            elif isinstance(node.value.op, ast.LShift):
                assert 0
            elif isinstance(node.value.op, ast.RShift):
                assert 0
            elif isinstance(node.value.op, ast.BitOr):
                assert 0
            elif isinstance(node.value.op, ast.BitXor):
                assert 0
            elif isinstance(node.value.op, ast.BitAnd):
                assert 0
            elif isinstance(node.value.op, ast.FloorDiv):
                assert 0
            else:
                assert 0

            # STEP 5 build op_wrapper
            _op_wrapper = astnodes.Operation(result_list=_result_list, op=_op, location=None)
            _autodiff_result_lhs = [astnodes.OpResult(value=_SsaId_left, count=None)]
            _autodiff_result_rhs = [astnodes.OpResult(value=_SsaId_right, count=None)]
            _autodiff_wrapper_lhs = [astnodes.Operation(result_list=_autodiff_result_lhs, op=_autodiff_lhs_op, location=None)]
            _autodiff_wrapper_rhs = [astnodes.Operation(result_list=_autodiff_result_rhs, op=_autodiff_rhs_op, location=None)]

        else:
            assert 0, "found unsupported form of rhs operator"
        # print(">>>>>Converted MLIR Node:<<<<<\n",
        #           self.pretty_mlir(node.mast_node))

        setattr(node, "mast_node", _op_wrapper)
        setattr(node, "mast_node_autodiff_lhs", _autodiff_wrapper_lhs)
        setattr(node, "mast_node_autodiff_rhs", _autodiff_wrapper_rhs)
        # print("\n ++++ show MLIR node ++++ \n", node.mast_node.dump())
        # print(node.mast_node_autodiff_lhs[0].dump())
        # print(node.mast_node_autodiff_rhs[0].dump())

        return node
