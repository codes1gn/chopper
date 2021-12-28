import ast

from chopper_frontend.scaffold.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    "IdenticalPastTransformer",
]


class IdenticalPastTransformer(NodeTransformerBase):
    """This is class that identical transformer python native Call astnode.

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self):
        """initialize the IdenticalPastTransformer class that inherit NodeTransformerBase.
        """
        super(self.__class__, self).__init__()

    def visit_Call(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Call's corresponding MLIR node.

        Args:
            node (ast.AST): python native python astnode.

        Returns:
            ast.AST: python native python astnode with mast_node attribution.
        """
        return node