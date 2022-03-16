import ast
import astunparse
from astunparse.printer import Printer

from chopper.scaffold.utils import *
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
        # * obtain func argument type to List()
        # * if no arguments: argument_type=None
        # * if arguments : argumets_type = [arg_type, ...]
        argument_type = None
        if node.mast_node.op.args:
            argument_type = []
            for nameargument in node.mast_node.op.args:
                argument_type.append(nameargument.type)
        else:
            argument_type = [None]

        op_type = node.mast_node.op.result_types
        return_type = node.mast_node.op.result_types

        # * if there no argument, set argument_type == return_type
        if not argument_type[0]:
            argument_type = [return_type]
        """
        Set the Operation Type based on the Argument Type and Return Type at the time the function was defined
        """
        # TODO(albert), eliminate this hardcoded of type constraints
        for operation in operations:
            _OP = operation.mast_node.op
            if isinstance(_OP, ReturnOperation):
                for i in range(len(operation.mast_node.op.types)):
                    _OP.types[i] = return_type
            if isinstance(_OP, ConstantOperation):
                _OP.type = argument_type[0]

            if isinstance(_OP, CustomOperation):
                # * BinOp -> add/sub/mul/matmul/div
                if (
                    (
                        _OP.name == "add"
                        or _OP.name == "sub"
                        or _OP.name == "mul"
                        or _OP.name == "matmul"
                        or _OP.name == "div"
                        or _OP.name == "mod"
                        or _OP.name == "pow"
                        or _OP.name == "lshift"
                        or _OP.name == "rshift"
                        or _OP.name == "bitor"
                        or _OP.name == "bitxor"
                        or _OP.name == "bitand"
                        or _OP.name == "floordiv"
                    )
                    and isinstance(_OP.type.argument_types, list)
                    and isinstance(_OP.type.result_types, list)
                ):
                    for i in range(len(_OP.type.argument_types)):
                        _OP.type.argument_types[i] = argument_type[0]

                    for i in range(len(_OP.type.result_types)):
                        _OP.type.result_types[i] = return_type
                else:
                    # TODO: add more BinOp type
                    pass

        if operations:
            for i in range(len(_blocks)):
                _blocks[i].body.clear()
                for _, operation in enumerate(operations):
                    _blocks[i].body.append(operation.mast_node)

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

        # print(self.pretty_mlir(node.mast_node))

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

    # def visit_Name(self, node: ast.AST) -> ast.AST:
    #     """
    #     Method that constructs the ReturnOperation in python_native dialect
    #     """
    #     super().generic_visit(node)
    #     print(self.__str__(), "Fix handling visit_Name on node\n",
    #           astunparse.dump(node))

    #     return node
