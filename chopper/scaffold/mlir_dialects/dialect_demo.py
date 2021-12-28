""" Dialect classes that create and custom dialect as example. """

from mlir import parse_string
from mlir.astnodes import Node, dump_or_value
from mlir.dialect import Dialect, DialectOp, DialectType

##############################################################################
# Dialect Types

__all__ = [
    "DIALECT_DEMO",
]


class RaggedTensorType(DialectType):
    """
    AST node class for the example "toy" dialect representing a ragged tensor.
    """

    _syntax_ = (
        "toy.ragged < {implementation.toy_impl_list} , {dims.dimension_list_ranked} {type.tensor_memref_element_type} >"
    )

    # Custom MLIR serialization implementation
    def dump(self, indent: int = 0) -> str:
        return "!toy.ragged<%s, %sx%s>" % (
            dump_or_value(self.implementation, indent),
            "x".join(dump_or_value(d, indent) for d in self.dims),
            dump_or_value(self.type, indent),
        )


class ToyImplementation(Node):
    """Base "toy" implementation AST node. Corresponds to a "+"-separated list
    of sparse tensor types.
    """

    _fields_ = ["values"]

    def __init__(self, node=None, **fields):
        self.values = node
        super().__init__(None, **fields)

    def dump(self, indent: int = 0) -> str:
        return "+".join(dump_or_value(v, indent) for v in self.values)


##############################################################################
# Dialect Operations


class DensifyOp(DialectOp):
    """AST node for an operation with an optional value."""

    _syntax_ = [
        "toy.densify {arg.ssa_id} : {type.tensor_type}",
        "toy.densify {arg.ssa_id} , {pad.constant_literal} : {type.tensor_type}",
    ]


##############################################################################
# Dialect

DIALECT_DEMO = Dialect(
    "toy",
    ops=[DensifyOp],
    types=[RaggedTensorType],
    preamble="""
// Exclamation mark in Lark means that string tokens will be preserved upon parsing
!toy_impl_type : "coo" | "csr" | "csc" | "ell"
toy_impl_list   : toy_impl_type ("+" toy_impl_type)*
                     """,
    transformers=dict(
        toy_impl_list=ToyImplementation,
        # Will convert every instance to its contents
        toy_impl_type=lambda v: v[0],
    ),
)
