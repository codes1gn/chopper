import ast
import torch

from chopper.scaffold.utils import *
from .node_visitor_base import NodeVisitorBase
from chopper.pass_manager.symbol_table import global_symbol_table, SymbolTable, SymbolEntry
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
    Dimension,
    RankedTensorType,
    NoneType,
    FloatTypeEnum,
    FloatType,
)

__all__ = [
    "AnnotateTypesVisitor",
]


class AnnotateTypesVisitor(NodeVisitorBase):
    """This is class that identical transformer python native Call astnode.

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self, arg_annotation: ArgAnnotation):
        """Initialize StmtConversionReadyCheckVisitor class via inherit NodeVisitorBase."""

        super().__init__()
        self.arg_annotation = arg_annotation

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Check Whether the FunctionDef conversion is successful.

        Args:
            node (ast.AST): FunctionDef with corresponding mlir astnode attribution.

        Returns:
            ast.AST: FunctionDef with corresponding mlir astnode attribution.
        """

        super().generic_visit(node)
        # only handles arguments > 0
        assert len(node.args.args) > 0
        # WORKAROUND
        setattr(node, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME, self.arg_annotation)
        func_args = node.args.args
        for arg_index in range(len(func_args)):
            _argname = func_args[arg_index].arg
            # TODO move this hardcode into base
            # case 1, arguments is float type
            if global_symbol_table.query(_argname) is not None:
                continue

            if self.arg_annotation[arg_index] is None:
                _type = NoneType()
            else:
                _dtype = FloatType(FloatTypeEnum.f32)
                _dim = [
                    Dimension(self.arg_annotation[arg_index][0][k])
                    for k in range(len(self.arg_annotation[arg_index][0]))
                ]
                _type = RankedTensorType(
                    dimensions=_dim,
                    element_type=_dtype,
                )
            global_symbol_table.register_symbol(SymbolEntry(_argname, _type))
            print(global_symbol_table)

        return node
