from typing import Any, List

import ast

from mlir import astnodes
from .pass_registry import PassRegistry


class PassManagerBase(object):
    """[summary]

    Attributes:
        _pass_vector  : the list of passes.
        _pass_id_list : the list of pass ids.
        _concrete_pass: the list of pass instance object.
    """

    __slots__ = [
        "_pass_vector",
        "_pass_id_list",
        "_concrete_pass",
    ]

    def __init__(self):
        """Initialize PassManagerBase attributions.
        """
        self._pass_vector = {}
        self._pass_id_list = []
        self._concrete_pass = []

    @property
    def pass_vector(self):
        """Wrapper pass_vector as PassManagerBase instance's attribution. 

        Returns:
            list : the list of pass_vector.
        """
        return self._pass_vector

    @property
    def pass_id_list(self):
        """Wrapper pass_id_list as PassManagerBase instance's attribution. 

        Returns:
            list : the list of pass_id_list.
        """
        return self._pass_id_list

    @property
    def concrete_pass(self):
        """Wrapper concrete_pass as PassManagerBase instance's attribution. 

        Returns:
            list : the list of concrete_pass.
        """
        return self._concrete_pass

    def add_pass(self, pass_class: Any) -> None:
        """Add pass to construct _pass_vector and _pass_id_list。

        Args:
            pass_class (Any): The Pass class
        
        Returns:
            None.
        """
        id = PassRegistry.pass_table[pass_class.__name__]
        assert isinstance(id, str)
        self._pass_vector[id] = pass_class
        self._pass_id_list.append(id)
        return

    def schedule_passes(self) -> list:
        # TODO(albert) keep dummy for now
        print("pass_manager_base::schedule_passes")
        return self._pass_id_list

    def run_pass(self, pass_class: object, code_node: ast.AST):
        """Run pass instance.

        Args:
            pass_class (object): the pass instance.
            code_node (ast.AST): the native python astnode.
        """
        cpass = pass_class()
        self._concrete_pass.append(cpass)
        code_node = cpass.run_pass(code_node)
        return

    def run(self, code_node: ast.AST):
        """Inference method to generate textual IR.

        Args:
            code_node (ast.AST): the native python astnode.
        """
        order_list = self.schedule_passes()
        for idx in order_list:
            pass_class = self._pass_vector[idx]
            # lazy_load pass_obj
            self.run_pass(pass_class, code_node)
        pass
