from typing import Tuple, List, Optional
import ast
import astunparse

from mlir import astnodes
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
    Dimension,
    RankedTensorType,
    NoneType,
    FloatTypeEnum,
    FloatType,
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

from chopper.pass_manager.symbol_table import *

__all__ = [
    "OpBuilder",
    "ValueBuilder",
    "TypeBuilder",
]


class OpBuilder(object):
    @classmethod
    def build(cls, shape: List[int], elem_ty: astnodes.Type):
        return


class ValueBuilder(object):
    @classmethod
    def build(cls, shape: List[int], elem_ty: astnodes.Type):
        return


class TypeBuilder(object):
    @classmethod
    def build(cls, shape: List[int], elem_ty: astnodes.Type):
        return
