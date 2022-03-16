import ast
import astunparse

from chopper.scaffold.utils import *
from .node_visitor_base import NodeVisitorBase

from mlir import astnodes
from mlir.dialects.standard import ReturnOperation
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId

__all__ = [
    "StmtConversionReadyCheckVisitor",
]


class StmtConversionReadyCheckVisitor(NodeVisitorBase):
    """This is Check Class that to make sure you convert MLIR astnode successful.

    The function contained is the same as the StmtNodeMappingTransformer class。

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self):
        """Initialize StmtConversionReadyCheckVisitor class via inherit NodeVisitorBase."""

        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Check Whether the FunctionDef conversion is successful.

        Args:
            node (ast.AST): FunctionDef with corresponding mlir astnode attribution.

        Returns:
            ast.AST: FunctionDef with corresponding mlir astnode attribution.
        """

        super().generic_visit(node)
        assert node.mast_node is not None
        # print("\n##### Check FunctionDef ######\n",
        #       self.pretty_mlir(node.mast_node))

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """Check Whether the Module conversion is successful.

        Args:
            node (ast.AST): Module with corresponding mlir astnode attribution.

        Returns:
            ast.AST: Module with corresponding mlir astnode attribution.
        """

        super().generic_visit(node)
        assert node.mast_node is not None
        # print("\n###### Check Module ######\n",
        #       self.pretty_mlir(node.mast_node))

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Check Whether the Return conversion is successful.

        Args:
            node (ast.AST): Return with corresponding mlir astnode attribution.

        Returns:
            ast.AST: Return with corresponding mlir astnode attribution.
        """

        super().generic_visit(node)
        assert node.mast_node is not None
        # print("\n##### Check ReturnOp #####\n",
        #       self.pretty_mlir(node.mast_node))

        return node

    def visit_Assign(self, node: ast.AST) -> ast.AST:
        """Check Whether the Assign astnode conversion is successful。

        Args:
            node (ast.AST): Assign astnode with corresponding mlir astnode attributions.

        Returns:
            ast.AST: Assign pyast with corresponding mlir astnode attributions
        """

        super().generic_visit(node)
        assert node.mast_node is not None
        # print("\n##### Check AssignOp ######\n", self.pretty_mlir(node.mast_node))

        return node

    # def visit_Name(self, node: ast.AST) -> ast.AST:
    #     """Check Whether the Name conversion is successful.

    #     Args:
    #         node (ast.AST): Name with corresponding mlir astnode attribution.

    #     Returns:
    #         ast.AST: Name with corresponding mlir astnode attribution.
    #     """
    #     super().generic_visit(node)
    #     assert node.mast_node is not None
    #     print("\n Check Type = \n", self.pretty_mlir(node.mast_node))

    #     return node
