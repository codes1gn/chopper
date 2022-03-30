import ast
import astunparse
from astunparse import printer
from typing import Tuple, List, Optional
import torch
from chopper.scaffold.utils.builders import *

from chopper.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase
from chopper.pass_manager.symbol_table import feed_forward_symbol_table

from mlir import astnodes
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
)
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp, TCF_ExpOp
from chopper.scaffold.mlir_dialects.dialect_atir import (
    ModuleName,
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
        # arguments = forward outs + forward saved-acts
        _arguments = []
        assert len(node.args.args) > 0
        for arg in node.args.args:
            # TODO move this hardcode into base
            # case 1, arguments is float type
            if arg.arg == "self":
                # HARDCODE + WORKAROUND, this is a temperal handle to avoid runtime error in type conversion by IREE Runtime
                continue
            _arguments.append(ValueBuilder.get_value_with_type(arg.arg, mode="backward"))

        _result_types = ValueBuilder.get_func_args_autodiff(mode="type")
        _result_types += ValueBuilder.get_saved_activations(mode="type")
        _function_wrapper = OpBuilder.create_function(func_name=node.name, arguments=_arguments, restypes=_result_types)

        # handle backward
        _autodiff_wrapper = [
            OpBuilder.create_return(
                ValueBuilder.get_func_rets_autodiff(mode="value"), ValueBuilder.get_func_rets_autodiff(mode="type")
            ),
        ]

        setattr(node, "mast_node", _function_wrapper)
        setattr(node, "mast_node_autodiff", _autodiff_wrapper)

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
        _block = astnodes.Block(label=None, body=[None])
        _region = astnodes.Region(body=[_block])
        _module = astnodes.Module(name=astnodes.SymbolRefId(value=unique_module_name.get_forward()), attributes=None, region=_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])

        _block_bp = astnodes.Block(label=None, body=[None])
        _region_bp = astnodes.Region(body=[_block_bp])
        _module_bp = astnodes.Module(name=astnodes.SymbolRefId(value=unique_module_name.get_backward()), attributes=None, region=_region_bp, location=None)
        _mlirfile_bp = astnodes.MLIRFile(definitions=[], modules=[_module_bp])
        setattr(node, "mast_node", _mlirfile)
        setattr(node, "mast_node_autodiff", _mlirfile_bp)

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
        assert isinstance(node.value, ast.Name), "no outs vars"

        _returnop_wrapper = []
        # b-act = identity(b)
        saved_act_values = ValueBuilder.get_saved_activations(mode="value")
        saved_act_types = ValueBuilder.get_saved_activations(mode="type")
        for idx in range(len(saved_act_values)):
            # get vanilla name, a-act => a ; a_0-act => a ; a-act-1_0 => a
            _full_name = saved_act_values[idx].value
            for _sub_str_idx in range(len(_full_name)):
                if _full_name[_sub_str_idx] == "-" or _full_name[_sub_str_idx] == "_":
                    break
            _vanilla_name = _full_name[:_sub_str_idx]

            _returnop_wrapper.append(
                OpBuilder.create_unary(
                    func="identity",
                    graph="forward",
                    retval=saved_act_values[idx],
                    operand=ValueBuilder.get_value(_vanilla_name, mode="forward"),
                )
            )
        total_return_values = [ValueBuilder.get_value(node.value.id)] + saved_act_values
        total_return_types = [ValueBuilder.get_type(node.value.id)] + saved_act_types
        _returnop_wrapper.append(
            OpBuilder.create_return(
                arguments=total_return_values,
                restypes=total_return_types,
            )
        )

        # handle backward
        # Construct the autodiff block for current ReturnOp where ReturnOp <=> FunctionOp
        _autodiff_wrapper = [
            OpBuilder.create_function(
                "bpfunction",
                arguments=[ValueBuilder.get_value_with_type(node.value.id)]
                + ValueBuilder.get_saved_activations(mode="value+type"),
                restypes=ValueBuilder.get_func_rets_autodiff(mode="type"),
            )
        ]

        setattr(node, "mast_node", _returnop_wrapper)
        setattr(node, "mast_node_autodiff", _autodiff_wrapper)
        # print("\n ++++ show MLIR node ++++ \n", node.mast_node.dump())
        # print(node.mast_node_autodiff[0].dump())

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

        if isinstance(node.value, ast.BinOp):
            # TODO tweak this functionality passed by hardcoded,
            # replaced by MAPPING ENUMS of ATIR_OPS in future
            # * binary op have two condition:
            # 1. scalar binary op
            # 2. list binart op, via numpy to implement

            # STEP 1 build SsaID for lhs and rhs
            assert not isinstance(node.value.left, ast.Call)
            assert not isinstance(node.value.right, ast.Call)

            # STEP 2 define SsaID of res + lhs + rhs
            _lhs_argname = node.value.left.id
            _SsaId_lhs_operand = ValueBuilder.get_value(_lhs_argname, mode="forward")

            _rhs_argname = node.value.right.id
            _SsaId_rhs_operand = ValueBuilder.get_value(_rhs_argname, mode="forward")

            _res_argnames_list = [target.id for target in node.targets]
            _res_argname = _res_argnames_list[0]
            _SsaId_outs = ValueBuilder.get_value(_res_argname)

            # STEP 3 build op arg types
            if isinstance(node.value.op, ast.Add):
                _op_wrapper = [
                    OpBuilder.create_binary(
                        func="add",
                        graph="forward",
                        retval=MlirSsaId(value=node.targets[0].id),
                        lhs_operand=_SsaId_lhs_operand,
                        rhs_operand=_SsaId_rhs_operand,
                    )
                ]
                _autodiff_wrapper = [
                    OpBuilder.create_unary(
                        func="identity", graph="backward", retval=_SsaId_lhs_operand, operand=_SsaId_outs
                    ),
                    OpBuilder.create_unary(
                        func="identity", graph="backward", retval=_SsaId_rhs_operand, operand=_SsaId_outs
                    ),
                ]
                setattr(node, "mast_node", _op_wrapper)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                return node

            elif isinstance(node.value.op, ast.Sub):
                _op_wrapper = [
                    OpBuilder.create_binary(
                        func="sub",
                        graph="forward",
                        retval=MlirSsaId(value=node.targets[0].id),
                        lhs_operand=_SsaId_lhs_operand,
                        rhs_operand=_SsaId_rhs_operand,
                    )
                ]
                _autodiff_wrapper = [
                    OpBuilder.create_unary(
                        func="identity", graph="backward", retval=_SsaId_lhs_operand, operand=_SsaId_outs
                    ),
                    OpBuilder.create_unary(
                        func="negate", graph="backward", retval=_SsaId_rhs_operand, operand=_SsaId_outs
                    ),
                ]
                print(_op_wrapper[0].dump())
                print(_autodiff_wrapper[0].dump())
                print(_autodiff_wrapper[1].dump())
                setattr(node, "mast_node", _op_wrapper)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                return node
            elif isinstance(node.value.op, ast.Mult):
                _SsaId_lhs_operand_act = ValueBuilder.get_value(_lhs_argname + "-act", mode="savedact")
                _SsaId_rhs_operand_act = ValueBuilder.get_value(_rhs_argname + "-act", mode="savedact")

                _op_wrapper = [
                    OpBuilder.create_binary(
                        func="mul",
                        graph="forward",
                        retval=MlirSsaId(value=node.targets[0].id),
                        lhs_operand=_SsaId_lhs_operand,
                        rhs_operand=_SsaId_rhs_operand,
                    )
                ]
                _autodiff_wrapper = [
                    OpBuilder.create_binary(
                        func="mul",
                        graph="backward",
                        retval=_SsaId_lhs_operand,
                        lhs_operand=_SsaId_outs,
                        rhs_operand=_SsaId_rhs_operand_act,
                    ),
                    OpBuilder.create_binary(
                        func="mul",
                        graph="backward",
                        retval=_SsaId_rhs_operand,
                        lhs_operand=_SsaId_lhs_operand_act,
                        rhs_operand=_SsaId_outs,
                    ),
                ]
                setattr(node, "mast_node", _op_wrapper)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                return node
            else:
                assert 0

        elif isinstance(node.value, ast.Call):
            _call_method = node.value.func.attr
            # split logics for unary and binary
            # TODO change it into pattern match or calltype(node.value) -> unary | binary | cmp or other
            if _call_method == "exp" or _call_method == "tanh":

                _argname = node.value.args[0].id
                _SsaId_operand = ValueBuilder.get_value(_argname, mode="forward")

                _res_argnames_list = [target.id for target in node.targets]
                _res_argname = _res_argnames_list[0]
                _SsaId_outs = ValueBuilder.get_value(_res_argname)

                # build mlir.op according to ast.op
                if _call_method == "tanh":
                    _SsaId_operand_act = ValueBuilder.get_value(_argname + "-act", mode="savedact")
                    _op_wrapper = [
                        OpBuilder.create_unary(func="tanh", graph="forward", retval=_SsaId_outs, operand=_SsaId_operand)
                    ]
                    _tmp_tanh_op, _tmp_tanh_outs = OpBuilder.create_unary_with_retval(
                        func="tanh",
                        graph="backward",
                        retval=_SsaId_operand_act,
                        operand=_SsaId_operand_act,
                        is_replica=False,
                        is_operand_act=True,
                    )
                    """use chain mode to avoid reduce on replicas, this is a tmp value"""
                    _tmp_mul_op, _tmp_mul_outs = OpBuilder.create_binary_with_retval(
                        func="mul",
                        graph="backward",
                        retval=_SsaId_operand,
                        lhs_operand=_tmp_tanh_outs,
                        rhs_operand=_tmp_tanh_outs,
                        is_replica=False,
                    )
                    _tmp_const_op, _tmp_const_outs = OpBuilder.create_const(retval=_SsaId_operand, literal=1.0)
                    _tmp_sub_op, _tmp_sub_outs = OpBuilder.create_binary_with_retval(
                        func="sub",
                        graph="backward",
                        retval=_SsaId_operand,
                        lhs_operand=_tmp_const_outs,
                        rhs_operand=_tmp_mul_outs,
                        is_replica=False,
                    )
                    _tmp_grad_mul_op = OpBuilder.create_binary(
                        func="mul",
                        graph="backward",
                        retval=_SsaId_operand,
                        lhs_operand=_SsaId_outs,
                        rhs_operand=_tmp_sub_outs
                    )
                    _autodiff_wrapper = [
                        _tmp_grad_mul_op,
                        _tmp_sub_op,
                        _tmp_const_op,
                        _tmp_mul_op,
                        _tmp_tanh_op,
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node

                elif _call_method == "exp":
                    _SsaId_operand_act = ValueBuilder.get_value(_argname + "-act", mode="savedact")
                    _op_wrapper = [
                        OpBuilder.create_unary(func="exp", graph="forward", retval=_SsaId_outs, operand=_SsaId_operand)
                    ]
                    _tmp_exp_op, _tmp_exp_outs = OpBuilder.create_unary_with_retval(
                        func="exp",
                        graph="backward",
                        retval=_SsaId_operand_act,
                        operand=_SsaId_operand_act,
                        is_replica=False,
                        is_operand_act=True,
                    )
                    _tmp_grad_mul_op = OpBuilder.create_binary(
                        func="mul",
                        graph="backward",
                        retval=_SsaId_operand,
                        lhs_operand=_SsaId_outs,
                        rhs_operand=_tmp_exp_outs
                    )
                    _autodiff_wrapper = [
                        _tmp_grad_mul_op,
                        _tmp_exp_op,
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node
                else:
                    assert 0, "unsupported unary op"

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
                _SsaId_lhs_operand = ValueBuilder.get_value(_lhs_argname, mode="forward")

                _rhs_argname = node.value.args[1].id
                _SsaId_rhs_operand = ValueBuilder.get_value(_rhs_argname, mode="forward")

                _res_argnames_list = [target.id for target in node.targets]
                _res_argname = _res_argnames_list[0]
                _SsaId_outs = ValueBuilder.get_value(_res_argname)

                # STEP 3 build op arg types
                if _call_method == "add":
                    _op_wrapper = [
                        OpBuilder.create_binary(
                            func="add",
                            graph="forward",
                            retval=MlirSsaId(value=node.targets[0].id),
                            lhs_operand=_SsaId_lhs_operand,
                            rhs_operand=_SsaId_rhs_operand,
                        )
                    ]
                    _autodiff_wrapper = [
                        OpBuilder.create_unary(
                            func="identity", graph="backward", retval=_SsaId_lhs_operand, operand=_SsaId_outs
                        ),
                        OpBuilder.create_unary(
                            func="identity", graph="backward", retval=_SsaId_rhs_operand, operand=_SsaId_outs
                        ),
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node

                elif _call_method == "sub":
                    _op_wrapper = [
                        OpBuilder.create_binary(
                            func="sub",
                            graph="forward",
                            retval=MlirSsaId(value=node.targets[0].id),
                            lhs_operand=_SsaId_lhs_operand,
                            rhs_operand=_SsaId_rhs_operand,
                        )
                    ]
                    _autodiff_wrapper = [
                        OpBuilder.create_unary(
                            func="identity", graph="backward", retval=_SsaId_lhs_operand, operand=_SsaId_outs
                        ),
                        OpBuilder.create_unary(
                            func="negate", graph="backward", retval=_SsaId_rhs_operand, operand=_SsaId_outs
                        ),
                    ]
                    print(_op_wrapper[0].dump())
                    print(_autodiff_wrapper[0].dump())
                    print(_autodiff_wrapper[1].dump())
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node
                elif _call_method == "mul":
                    _SsaId_lhs_operand_act = ValueBuilder.get_value(_lhs_argname + "-act", mode="savedact")
                    _SsaId_rhs_operand_act = ValueBuilder.get_value(_rhs_argname + "-act", mode="savedact")

                    _op_wrapper = [
                        OpBuilder.create_binary(
                            func="mul",
                            graph="forward",
                            retval=MlirSsaId(value=node.targets[0].id),
                            lhs_operand=_SsaId_lhs_operand,
                            rhs_operand=_SsaId_rhs_operand,
                        )
                    ]
                    _autodiff_wrapper = [
                        OpBuilder.create_binary(
                            func="mul",
                            graph="backward",
                            retval=_SsaId_lhs_operand,
                            lhs_operand=_SsaId_outs,
                            rhs_operand=_SsaId_rhs_operand_act,
                        ),
                        OpBuilder.create_binary(
                            func="mul",
                            graph="backward",
                            retval=_SsaId_rhs_operand,
                            lhs_operand=_SsaId_lhs_operand_act,
                            rhs_operand=_SsaId_outs,
                        ),
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node
                elif _call_method == "linear" or _call_method == "matmul":
                    _SsaId_lhs_operand_act = ValueBuilder.get_value(_lhs_argname + "-act", mode="savedact")
                    _SsaId_rhs_operand_act = ValueBuilder.get_value(_rhs_argname + "-act", mode="savedact")

                    _op_wrapper = [
                        OpBuilder.create_binary(
                            func="matmul",
                            graph="forward",
                            retval=MlirSsaId(value=node.targets[0].id),
                            lhs_operand=_SsaId_lhs_operand,
                            rhs_operand=_SsaId_rhs_operand,
                        )
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    print(node.mast_node[0].dump())
                    _const_op, _transpose_const = OpBuilder.create_const(retval=_SsaId_lhs_operand, literal=[1, 0])
                    _lhs_transpose_op, _tmp_lhs_act_transposed = OpBuilder.create_binary_with_retval(
                        func="transpose",
                        graph="backward",
                        retval=_SsaId_lhs_operand_act,
                        lhs_operand=_SsaId_lhs_operand_act,
                        rhs_operand=_transpose_const,
                    )
                    _rhs_transpose_op, _tmp_rhs_act_transposed = OpBuilder.create_binary_with_retval(
                        func="transpose",
                        graph="backward",
                        retval=_SsaId_rhs_operand_act,
                        lhs_operand=_SsaId_rhs_operand_act,
                        rhs_operand=_transpose_const,
                    )
                    _autodiff_wrapper = [
                        OpBuilder.create_binary(
                            func="matmul",
                            graph="backward",
                            retval=_SsaId_lhs_operand,
                            lhs_operand=_SsaId_outs,
                            rhs_operand=_tmp_rhs_act_transposed,
                        ),
                        OpBuilder.create_binary(
                            func="matmul",
                            graph="backward",
                            retval=_SsaId_rhs_operand,
                            lhs_operand=_tmp_lhs_act_transposed,
                            rhs_operand=_SsaId_outs,
                        ),
                        _lhs_transpose_op,
                        _rhs_transpose_op,
                        _const_op,
                    ]
                    setattr(node, "mast_node", _op_wrapper)
                    setattr(node, "mast_node_autodiff", _autodiff_wrapper)
                    return node

                else:
                    assert 0

                """
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
                """
            else:
                assert 0, "Not Support this op conversion from ast.AST -> mlir.astnodes.Node"

            return node
        else:
            assert 0, "RHS of AssignOp expect BinOp or Call"

        return node
