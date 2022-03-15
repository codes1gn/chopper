from chopper.pass_manager.pass_manager_base import PassManagerBase
from chopper.pass_manager.passes import *

import ast

__all__ = [
    "ArgAnnotationPassManager",
]


class ArgAnnotationPassManager(PassManagerBase):
    """The class inherit PassManagerBase that to register passes.

    Attributions:
        None.
    """

    def __init__(self, arg_annotation):
        self.arg_annotation = arg_annotation
        super().__init__()

    def register_passes(self):
        """Register passes via add_pass func in PassManagerBase.
        """
        self.add_pass(AnnotateCompletionPass)

        return

    # override this function to support args
    def run_pass(self, pass_class: object, code_node: ast.AST):
        """Run pass instance.

        Args:
            pass_class (object): the pass instance.
            code_node (ast.AST): the native python astnode.
        """
        # this pass has special case of arg num to support arg passing
        cpass = pass_class(self.arg_annotation)
        self._concrete_pass.append(cpass)
        code_node = cpass.run_pass(code_node)
        return
