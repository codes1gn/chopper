import ast
import astunparse
from astunparse.printer import Printer

from chopper.scaffold.utils import *
from chopper.pass_manager.symbol_table import *
from .node_transformer_base import NodeTransformerBase
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from mlir.astnodes import CustomOperation, FunctionType, NamedArgument
from mlir import astnodes
from mlir.dialects.standard import *
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId

__all__ = [
    "StmtFixDependencyTransformer",
]


class StmtFixDependencyTransformer(NodeTransformerBase):
    """This is fix dependcy Transformer defined in StmtNodeMappingTransformer class

    We map single python astnode to mlir astnode in StmtNodeMappingTransformer class,
    will consolidate all single node transformer to generate final mlir astnode.

    Attributtes:
        None.
    """

    __slots__ = []

    def __init__(self):
        """Initialize StmtFixDependencyTransformer class via inherit NodeTransformerBase."""

        super().__init__()

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Method that constructs the FunctionDef in python_native dialect

        Args:
            node (ast.AST): python native astnode with mast_node attributions.

        Returns:
            ast.AST: python native astnode with mast_node attributions.
        """

        super().generic_visit(node)
        # print(self.__str__(),
        #       "Fix Transformer::handling visit_FunctionDef on node\n")
        # print("***** Python FunctionDef Node *****\n", astunparse.dump(node))

        # print("***** MLIR Node fot FunctionDef *****\n",
        #       self.pretty_mlir(node.mast_node))

        # TODO Fix body elements in function region's block
        """
        Region: body (consist of a series Block)
        Need set ReturnOp Type according Assign op when Pass the Return assigned variable.
        """
        _blocks = node.mast_node.op.region.body
        operations = node.body
        if operations:
            for i in range(len(_blocks)):
                _blocks[i].body.clear()
                for _, operation in enumerate(operations):
                    _blocks[i].body.append(operation.mast_node)

        # handle autodiff logics
        _returnop = node.mast_node_autodiff
        from collections import deque

        _autodiff_op_stack = deque()
        # print(_returnop.dump())
        _autodiff_root = None
        if operations:
            for _, operation in enumerate(operations):
                if hasattr(operation, "mast_node_autodiff"):
                    _autodiff_op_stack.append(operation.mast_node_autodiff)
                else:
                    _autodiff_op_stack.append(operation.mast_node_autodiff_rhs)
                    _autodiff_op_stack.append(operation.mast_node_autodiff_lhs)
        while _autodiff_op_stack:
            _op = _autodiff_op_stack.pop()
            # print(self.pretty_mlir(_op))
            if isinstance(_op.op, astnodes.Function):
                _autodiff_root = _op
                continue
            else:
                assert _autodiff_root is not None, "should has value"
                # print(_autodiff_root.op.region.body[0].body)
                _autodiff_root.op.region.body[0].body.append(_op)
                # print(_autodiff_root.op.region.body[0].body)
        _autodiff_root.op.region.body[0].body.append(_returnop)
        # print(_autodiff_root.dump())
        global_symbol_table.reset_autodiff_graph()
        global_symbol_table.set_autodiff_graph(_autodiff_root)

        return node

    def visit_Module(self, node: ast.AST) -> ast.AST:
        """Method that constructs the Module in python_native dialect

        Module is the tops of python native ast, should traverses all nodes in Module,
        the final MLIR astnode is constructs according to the mast_node attribute of each node.


        Args:
            node (ast.AST): python native astnode with all mast_node attributions.

        Returns:
            ast.AST: python native astnode with final MLIR astnode.
        """

        super().generic_visit(node)
        # print(self.__str__(), "Fix handling visit_Module on node\n",
        #       astunparse.dump(node))

        for _module in node.mast_node.modules:
            for _block in _module.region.body:
                for index in range(len(_block.body)):
                    _block.body[index] = node.body[index].mast_node

        # handle autodiff mastnodes, since its bodies are FunctionDef, they all
        # set with mast_node_autodiff

        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """Method that constructs the ReturnOperation in python_native dialect

        Args:
            node (ast.AST): python native astnode with mast_node attributions.

        Returns:
            ast.AST: python native astnode with mast_node attributions.
        """

        super().generic_visit(node)
        # print(self.__str__(), "Fix handling visit_Return on node\n",
        #       astunparse.dump(node))

        # fix returnop value
        # node.mast_node.op.values = node.value

        # print(self.pretty_mlir(node.mast_node))

        return node
