""" Implemented classes of Tensor Computation Flow Dialect. """

import inspect
import sys

from mlir import parse_string
from dataclasses import dataclass
import mlir.astnodes as mast
from mlir.astnodes import Node, dump_or_value, TensorType, Dimension, FloatType, IntegerType, ComplexType, VectorType
from mlir.dialect import Dialect, DialectOp, DialectType, UnaryOperation, BinaryOperation, is_op

from typing import Union, Optional, List

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]


__all__ = [
    "DIALECT_ATIR",
]


##############################################################################
# Dialect Types


# this workaround tensortype is to replace the scalar types
@dataclass
class UnitTensorType(TensorType):
    element_type: Union[IntegerType, FloatType, ComplexType, VectorType]

    def dump(self, indent: int = 0) -> str:
        return "tensor<%s>" % (self.element_type.dump(indent))


@dataclass
class ATIR_ConstOp(DialectOp):
    """AST node for an operation with an optional value."""

    value: float
    dtype: mast.Type

    _opname_ = "tosa.const"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    # _syntax_ = [
    #     '"tosa.const"() \{value = dense< {value.float} > : {dtype.type} \} : () -> {dtype.type}',
    # ]

    def dump(self, indent: int = 0) -> str:
        return (
            '"tosa.const"() {value = dense<'
            + self.value.__str__()
            + "> : "
            + self.dtype.dump()
            + "} : () -> "
            + self.dtype.dump()
        )


##############################################################################
# Dialect Operations


@dataclass
class ATIR_AddOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.add"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.add {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_SubOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.sub"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.sub {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_MulOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.mul"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.mul {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_DivOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.div"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.div {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_MaxOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType
    _opname_ = "atir.max"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.max {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_MatmulOp(DialectOp):
    """AST node for an operation with an optional value."""

    operand_a: SsaUse
    operand_b: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.matmul"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.matmul {operand_a.ssa_use} , {operand_b.ssa_use} : {dtype.function_type}",
    ]


@dataclass
class ATIR_Conv2DChannelFirstOp(DialectOp):
    """AST node for an operation with an optional value."""

    activation: SsaUse
    kernel: SsaUse
    dtype: mast.FunctionType

    _opname_ = "atir.conv_2d_cfirst"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.conv_2d_cfirst {activation.ssa_use} , {kernel.ssa_use} : {dtype.function_type}",
    ]


class ATIR_TanhOp(UnaryOperation):
    _opname_ = "atir.tanh"


class ATIR_ExpOp(UnaryOperation):
    _opname_ = "atir.exp"


class ATIR_IdentityOp(UnaryOperation):
    _opname_ = "atir.identity"


class ATIR_NegateOp(UnaryOperation):
    _opname_ = "atir.negate"


##############################################################################
# Dialect

DIALECT_ATIR = Dialect(
    "atir",
    ops=[m[1] for m in inspect.getmembers(sys.modules[__name__], lambda obj: is_op(obj, __name__))],
    types=[],
    preamble="",
    transformers=None,
)
