""" Implemented classes of Tensor Computation Flow Dialect. """

import inspect
import sys

from mlir import parse_string
from dataclasses import dataclass
import mlir.astnodes as mast
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType, UnaryOperation, BinaryOperation, is_op

from typing import Union, Optional, List

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

##############################################################################
# Dialect Types

__all__ = [
    "DIALECT_ATIR",
]

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

    _opname_ = "atir.conv_2d_nchw"

    # TODO in syntax, between string_literals and non-terminals, must be
    # seperated with whitespace
    _syntax_ = [
        "atir.conv_2d_nchw {activation.ssa_use} , {kernel.ssa_use} : {dtype.function_type}",
    ]


class ATIR_TanhOp(UnaryOperation):
    _opname_ = "atir.tanh"



class ATIR_ExpOp(UnaryOperation):
    _opname_ = "atir.exp"


##############################################################################
# Dialect

DIALECT_ATIR = Dialect(
    "atir",
    ops=[
        m[1] for m in inspect.getmembers(sys.modules[__name__],
                                         lambda obj: is_op(obj, __name__))
    ],
    types=[],
    preamble="",
    transformers=None,
)
