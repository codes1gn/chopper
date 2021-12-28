import ast

from chopper_frontend.pass_manager.transformers import IdenticalPastTransformer
from chopper_frontend.pass_manager.passes.pass_base import PassBase

__all__ = [
    "IdenticalPastPass",
]


class IdenticalPastPass(PassBase):
    """IndeticalPasyPass inherit PassBase.

    Pass the IdenticalPastTransformer.

    Attributions:
        solvers (list) : the transformer involved in this pass.
    """
    __slots__ = [
        "solvers",
    ]

    def __init__(self):
        """Initialize the all attributions.
        """
        super(IdenticalPastPass, self).__init__()
        self.solvers = []
        self.solvers.append(IdenticalPastTransformer)

    def run_pass(self, ast_root: ast.AST) -> ast.AST:
        """Run this pass to convert native astnode.

        Args:
            ast_root (ast.AST): the native astnode.

        Returns:
            ast.AST: the converted astnode after run the pass.
        """
        for _solver in self.solvers:
            ast_root = _solver().visit(ast_root)
            ast.fix_missing_locations(ast_root)

        return ast_root
