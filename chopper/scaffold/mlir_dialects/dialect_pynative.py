""" Implemented classes of NativePython Dialect. """

import inspect
import sys

from mlir import parse_string
from dataclasses import dataclass
import mlir.astnodes as mast
from mlir.dialect import Dialect, DialectOp, DialectType, is_op
from typing import Union, Optional, List

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

##############################################################################
# Dialect Types

__all__ = [
    "DIALECT_PYNATIVE",
]

##############################################################################
# Dialect Operations


@dataclass
class PYNATIVE_ConstantOp(DialectOp):
    """AST node for an operation with an optional value."""

    arg: Union[mast.ElementsAttr, mast.FloatAttr, mast.IntegerAttr]

    _syntax_ = [
        "pynative.constant {arg.elements_attribute}",
        "pynative.constant {arg.float_attribute}",
        "pynative.constant {arg.integer_attribute}",
    ]


##############################################################################
# Dialect

DIALECT_PYNATIVE = Dialect(
    "pynative",
    ops=[
        m[1] for m in inspect.getmembers(sys.modules[__name__],
                                         lambda obj: is_op(obj, __name__))
    ])
