import ast
import astunparse
from astunparse import printer
from typing import Tuple, List, Optional
import torch

from chopper.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase
from chopper.pass_manager.symbol_table import global_symbol_table, SymbolTable, SymbolEntry

from mlir import astnodes
from mlir.astnodes import CustomOperation, FunctionType, NamedArgument, Dimension, RankedTensorType, NoneType
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp, TCF_ExpOp
from chopper.scaffold.mlir_dialects.dialect_atir import ATIR_AddOp, ATIR_ExpOp, UnitTensorType

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

        _block = astnodes.Block(label=None, body=[None])
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

                _annotation = node._torch_dsl_arg_annotations[arg_index]
                # print(_annotation)
                if _annotation is None:
                    _type = NoneType()
                    # HARDCODE + WORKAROUND, this is a temperal handle to avoid runtime error in type conversion by IREE Runtime
                    continue
                else:
                    _dtype = astnodes.FloatType(MlirType.f32)
                    _dim = [Dimension(_annotation[0][k]) for k in range(len(_annotation[0]))]
                    _type = RankedTensorType(
                        dimensions=_dim,
                        element_type=_dtype,
                    )
                    global_symbol_table.register_symbol(SymbolEntry(arg.arg, _type))

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

        _result_type = None
        # TODO is this function return is None, keep it until all shapes are fixed
        if node.returns:
            assert 0, "wait for tuning, support pre-annotated function-type-return"
            if hasattr(node.returns, "id") and node.returns.id == "float":
                # TODO(albert) workaround to tensor, should be f32 in future , and do convertion in compiler
                _type = UnitTensorType(element_type=astnodes.FloatType(MlirType.f32))
                _result_type = _type
            elif hasattr(node.returns, "id") and node.returns.id == "list":
                _result_type = astnodes.RankedTensorType(
                    dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32)
                )
            elif (
                isinstance(arg.annotation, ast.Subscript)
                and isinstance(arg.annotation.slice, ast.Index)
                and arg.annotation.value.id == "List"
            ):
                if arg.annotation.slice.value.id == "float":  # TODO: Other type, only support float now
                    _result_type = astnodes.RankedTensorType(
                        dimensions=[Dimension(value=None)], element_type=astnodes.FloatType(MlirType.f32)
                    )
            else:
                # TODO: Other type
                pass
        elif node.returns is None:
            # TODO this path now handles equal shaped binary or single shape
            # assert node._torch_dsl_arg_annotations is List[Optional[ArgAnnotation]]
            # TOP guardian
            _annotations = node._torch_dsl_arg_annotations
            if len(_annotations) == 1:
                # this case only has self as sole arg, not possible to infer out type
                _result_type = None
            elif len(_annotations) == 2:
                # this case is unary function, generally outputs same type
                # special cases TODO handled later
                _sole_annotation = _annotations[1]
                _dtype = astnodes.FloatType(MlirType.f32) if _sole_annotation[1] is torch.float32 else None
                assert _dtype is not None, "guard this condition if dtype not annotated"
                _dim = [Dimension(_sole_annotation[0][k]) for k in range(len(_sole_annotation[0]))]
                _result_type = RankedTensorType(
                    dimensions=_dim,
                    element_type=_dtype,
                )
            elif len(_annotations) == 3:
                _lhs_annotation = _annotations[1]
                _rhs_annotation = _annotations[2]
                # do sanity check and only handles equal shape, guard unsupported situation
                # TODO should provide all equal compare utils
                assert _lhs_annotation == _rhs_annotation
                _dtype = astnodes.FloatType(MlirType.f32) if _lhs_annotation[1] is torch.float32 else None
                assert _dtype is not None, "guard this condition if dtype not annotated"
                _dim = [Dimension(_annotation[0][k]) for k in range(len(_annotation[0]))]
                _result_type = RankedTensorType(
                    dimensions=_dim,
                    element_type=_dtype,
                )

        _attributes = None

        _function = astnodes.Function(
            name=_name,
            args=_args,
            result_types=_result_type,
            region=_region,
            attributes=_attributes,
        )
        _function_wrapper = astnodes.Operation(result_list=[], op=_function, location=None)

        setattr(node, "mast_node", _function_wrapper)
        # print("\n>>>>>MLIR Node for FunctionDef:<<<<<\n",
        #       self.pretty_mlir(node.mast_node))

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

        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=_name, attributes=_attributes, region=_out_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        setattr(node, "mast_node", _mlirfile)
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
            # anchor
            # _type = global_symbol_table.query(_value)
            _op_no = None

            _values.append(MlirSsaId(value=_value, op_no=_op_no))
            _types.append(None)
            _returnop.values = _values
            _returnop.types = _types

        _returnop_wrapper = astnodes.Operation(result_list=None, op=_returnop, location=None)

        setattr(node, "mast_node", _returnop_wrapper)
        # print("\nMLIR Node for Return:<<<<<\n",
        #       self.pretty_mlir(node.mast_node))

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
        def mapping_Assign_Call(_node: ast.AST) -> ast.AST:
            assert _node.value.func.attr == "exp", "assign an unsupported call"
            _SsaId_operand = MlirSsaId(value=node.value.args[0].id, op_no=None)

            # TODO(albert) this is a temporal transfer, use f32 for hardcoded
            _type = UnitTensorType(element_type=astnodes.FloatType(MlirType.f32))
            _assignop = ATIR_ExpOp(match=0, operand=_SsaId_operand, type=_type)

            _result_list = list()
            _result_list.append(astnodes.OpResult(value=MlirSsaId(value=node.targets[0].id, op_no=None), count=None))
            _assignop_wrapper = astnodes.Operation(result_list=_result_list, op=_assignop, location=None)

            setattr(_node, "mast_node", _assignop_wrapper)
            return _node

        _type = None

        if isinstance(node.value, ast.Call):
            assert 0, "Guardian: not verified"
            node = mapping_Assign_Call(node)

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
            _SsaId_left = _SsaId_right = None
            if isinstance(node.value.left, ast.Call):
                _SsaId_left = MlirSsaId(value=node.value.left.args[0].id, op_no=None)
                _SsaId_right = MlirSsaId(value=node.value.left.args[0].id, op_no=None)

            else:
                _SsaId_left = MlirSsaId(value=node.value.left.id, op_no=None)
                _SsaId_right = MlirSsaId(value=node.value.right.id, op_no=None)

            # STEP 2 build op arg types
            # TODO albert hardcode remove
            _dtype = astnodes.FloatType(MlirType.f32)
            _dim = [Dimension(2), Dimension(3)]
            _type = RankedTensorType(
                dimensions=_dim,
                element_type=_dtype,
            )

            _argument_types = [_type, _type]
            _result_types = [_type]
            _op_type = FunctionType(argument_types=_argument_types, result_types=_result_types)

            # STEP 3 build result symbol
            _result_list = list()
            _result_list.append(astnodes.OpResult(value=MlirSsaId(value=node.targets[0].id, op_no=None), count=None))

            # STEP 4 build op according to py.op
            if isinstance(node.value.op, ast.Add):
                _op = ATIR_AddOp(
                    match=0,
                    operand_a=_SsaId_left,
                    operand_b=_SsaId_right,
                    dtype=_op_type,
                )
            elif isinstance(node.value.op, ast.Sub):
                assert 0
            elif isinstance(node.value.op, ast.Mult):
                assert 0
            elif isinstance(node.value.op, ast.Div):
                assert 0
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
            # print(">>>>>MLIR Node for Assign BinOp:<<<<<\n",
            #       self.pretty_mlir(_op_wrapper))
            setattr(node, "mast_node", _op_wrapper)

        else:
            assert 0, "found unsupported form of rhs operator"
        # print(">>>>>Converted MLIR Node:<<<<<\n",
        #           self.pretty_mlir(node.mast_node))

        return node

    # def visit_Name(self, node: ast.AST) -> ast.AST:
    #     """Method that constructs MLIR node via the func return type.

    #     Construct MLIR node by set Name attribute "mast_node".
    #     Name is return expr args.

    #     Args:
    #         node (ast.AST): Name node of python ast.

    #     Returns:
    #         ast.AST: Name node with corresponding MLIR ast node.
    #     """
    #     super().generic_visit(node)
    #     # print(self.__str__(), "visit_Name on node\n", astunparse.dump(node))

    #     _type_wrapper = astnodes.FloatType(MlirType.f32)

    #     # print("_type_wrapper:\n", self.pretty_mlir(_type_wrapper))
    #     setattr(node, "mast_node", _type_wrapper)
    #     return node
