import ast

from chopper.pass_manager.transformers import *
from chopper.pass_manager.passes.pass_base import PassBase
from chopper.scaffold.utils import *
from chopper.pass_manager.symbol_table import global_symbol_table, SymbolTable, SymbolEntry


__all__ = [
    "AnnotateCompletionPass",
]


class AnnotateCompletionPass(PassBase):
    """Convert python AST nodes to MLIR AST nodes

    this pass converts all python statements into relevant mlir astnodes
    will find all statements nodes, and set its mast_node value,

    the last step of this pass will be the check pass that checks all nodes
    that belongs to statements.

    should impl is_stmt and is_conversion_ready methods

    follow EBNF rules as:

    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns,
                             string? type_comment)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body, string? type_comment)
          | AsyncWith(withitem* items, stmt* body, string? type_comment)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    Attributes:
        solvers: the list of transformers that map stmt nodes to mlir astnodes, check mlir astnodes and
                 fix dependency respectively。
    """

    __slots__ = [
        "solvers",
    ]

    def __init__(self, arg_annotation: ArgAnnotation):
        """initialize the AnnotateCompletionPass class, and all attributes pass in with args.

        solvers is a list contains Transformers class to convert python native to MLIR astnode.
        """
        super().__init__()
        self.solvers = []
        self.solvers.append(AnnotateTypesVisitor)
        self.arg_annotation = arg_annotation

    def run_pass(self, ast_root: ast.AST) -> ast.AST:
        """Run this pass to convert astnode.

        Args:
            ast_root (ast.AST): python astnode.

        Returns:
            ast.AST: the converted astnode after run the pass.
        """

        print("\n====== enter AnnotateCompletionPass =====\n")
        global_symbol_table.pass_again = True
        for _solver in self.solvers:
            while global_symbol_table.pass_again == True:
                global_symbol_table.pass_again = False
                ast_root = _solver(self.arg_annotation).visit(ast_root)

        return ast_root
