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

__all__ = [
    "DIALECT_SHAPE",
]

##############################################################################
# Dialect Types


@dataclass
class ShapeTensorType(DialectType):
    """
    AST node class for the shape tensor type.
    """

    dimensions: List[mast.Dimension]
    element_type: mast.IndexType

    _syntax_ = "shape.ttensor < {dimensions.dimension_list_ranked} {element_type.index_type} > "

    # Custom MLIR serialization implementation
    def dump(self, indent: int = 0) -> str:
        return "shape.ttensor<%sx%s>" % (
            "x".join(dump_or_value(d, indent) for d in self.dimensions),
            dump_or_value(self.element_type, indent),
        )


##############################################################################
# Dialect Operations


@dataclass
class SHAPE_ConstShape(DialectOp):
    """AST node for an operation with an optional value."""

    shape_literal: mast.ArrayAttr
    shape_type: mast.TensorType

    _opname_ = "shape.const_shape"

    _syntax_ = [
        "shape.const_shape {shape_literal.array_attribute} : {shape_type.tensor_type}",
    ]


##############################################################################
# Dialect

DIALECT_SHAPE = Dialect(
    "shape",
    ops=[SHAPE_ConstShape],
    # ops=[m[1] for m in inspect.getmembers(sys.modules[__name__], lambda obj: is_op(obj, __name__))],
    types=[ShapeTensorType],
    # preamble="""
    # shape_tensor_type : "tensor" "<" dimension_list_ranked index_type ">"
    # """,
)
