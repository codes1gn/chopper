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
    "AutodiffMergeReplicas",
]


class AutodiffMergeReplicas(NodeTransformerBase):
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
            _SsaId_left = ValueBuilder.get_value(_lhs_argname)

            _rhs_argname = node.value.right.id
            _SsaId_right = ValueBuilder.get_value(_rhs_argname)

            _res_argnames_list = [target.id for target in node.targets]
            _res_argname = _res_argnames_list[0]
            _SsaId_outs = ValueBuilder.get_value(_res_argname)

            # STEP 3 build op arg types
            _autodiff_wrapper = node.mast_node_autodiff
            extra_autodiff_ops = OpBuilder.create_replica_merges(operand=_SsaId_outs)

            setattr(node, "mast_node_autodiff", _autodiff_wrapper + extra_autodiff_ops)
            return node

        elif isinstance(node.value, ast.Call):
            _call_method = node.value.func.attr

            if _call_method == "exp" or _call_method == "tanh":
                _argname = node.value.args[0].id
                _SsaId_operand = ValueBuilder.get_value(_argname, mode="forward")

                _res_argnames_list = [target.id for target in node.targets]
                _res_argname = _res_argnames_list[0]
                _SsaId_outs = ValueBuilder.get_value(_res_argname)
                _SsaId_operand_act = ValueBuilder.get_value(_argname + "-act", mode="savedact")
                # STEP 3 build op arg types
                _autodiff_wrapper = node.mast_node_autodiff
                extra_autodiff_ops = OpBuilder.create_replica_merges(operand=_SsaId_outs)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper + extra_autodiff_ops)
                return node

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
                _SsaId_left = ValueBuilder.get_value(_lhs_argname, mode="forward")

                _rhs_argname = node.value.args[1].id
                _SsaId_right = ValueBuilder.get_value(_rhs_argname, mode="forward")

                _res_argnames_list = [target.id for target in node.targets]
                _res_argname = _res_argnames_list[0]
                _SsaId_outs = ValueBuilder.get_value(_res_argname)

                # STEP 3 build op arg types
                _autodiff_wrapper = node.mast_node_autodiff
                extra_autodiff_ops = OpBuilder.create_replica_merges(operand=_SsaId_outs)

                setattr(node, "mast_node_autodiff", _autodiff_wrapper + extra_autodiff_ops)
                return node
            elif _call_method == "sample":
                sample_fn = node.value.args[1]
                arg0 = sample_fn.args[0].id  # loc in Normal, minval in Uniform
                arg1 = sample_fn.args[1].id # scale in Normal, maxval in Uniform
                _SsaId_operand0 = ValueBuilder.get_value(arg0)
                _SsaId_operand1 = ValueBuilder.get_value(arg1)
                _res_argnames_list = [target.id for target in node.targets]
                _res_argname = _res_argnames_list[0]
                _SsaId_outs = ValueBuilder.get_value(_res_argname)
                _autodiff_wrapper = node.mast_node_autodiff
                extra_autodiff_ops = OpBuilder.create_replica_merges(operand=_SsaId_outs)
                setattr(node, "mast_node_autodiff", _autodiff_wrapper + extra_autodiff_ops)
                return node
            else:
                assert 0, "Not support op type in Autodiff_merger_replicas pass"
        else:
            assert 0

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Method that constructs the FunctionDef's corresponding MLIR node.

        Construct MLIR node by set FunctionDef attribute "mast_node".

        Args:
            node (ast.AST): FunctionDef node of python stmt.

        Returns:
            ast.AST: FunctionDef node with corresponding MLIR ast node.
        """
        super().generic_visit(node)

        _operands = ValueBuilder.get_func_rets_autodiff(mode="value")

        _extra_operations = []
        for _operand in _operands:
            _ = OpBuilder.create_replica_merges(_operand)
            _extra_operations += _

        # handle backward
        _autodiff_wrapper = node.mast_node_autodiff
        setattr(node, "mast_node_autodiff", _extra_operations + _autodiff_wrapper)

        return node
