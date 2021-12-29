""" Contains jit runner class for compilation and execution """
import sys
import inspect
import textwrap
import copy
import traceback
import types
import ast
import astunparse

from typing import Callable
from imp import new_module

from mlir import parse_path
from mlir import astnodes
from chopper.scaffold.mlir_dialects import *
from chopper.scaffold.utils import *

MlirNode = astnodes.Node
MlirSsaId = astnodes.SsaId


def _pretty(self: MlirNode) -> str:
    result = self.dump_ast()
    lines = [""]
    indent = 0
    for index in range(len(result)):
        char = result[index]
        indent_word = "  "

        if char == " ":
            continue

        if char == "[" and result[index + 1] == "]":
            indent += 1
            lines[-1] += char
            continue

        if char == ",":
            lines[-1] += char
            lines.append(indent * indent_word)
            continue

        if char == "[":
            indent += 1
            lines[-1] += char
            lines.append(indent * indent_word)
            continue
        if char == "]":
            indent -= 1

        if char == "(":
            indent += 1
            lines[-1] += char
            lines.append(indent * "  ")
            continue
        if char == ")":
            indent -= 1

        if char != "\n":
            lines[-1] += char
        if char == "\n":
            lines.append(indent * indent_word)

    return "\n".join(lines)


MlirNode.pretty = _pretty

__all__ = [
    "PythonRunner",
]


class PythonRunner:
    """
    PythonRunner class that is a compiler supports jit functionalities for numpy DSL.

    Returns:
        PythonRunner: returns the instance of this class
    """

    __slots__ = ["pass_manager"]

    def __init__(self):
        """
        Initializes the PythonRunner
        """
    @classmethod
    def dump_mlir(cls, _ast: MlirNode) -> str:
        """dump mlir string.

        Args:
            _ast(MlirNode): mlir ast.

        Returns:
            str: mlir string.
        """
        dump_str = ""
        dump_str += ColorPalette.FAIL
        dump_str += "\n*****dumping mlir ast*****\n"
        dump_str += _ast.pretty()
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\ndumping mlir IR\n"
        dump_str += _ast.dump()
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************\n"
        return dump_str
    
    @classmethod
    def dump_python(cls, _ast: ast.AST) -> str:
        """dump python ast and corresponding source code.

        Args:
            _ast(ast.AST): python ast.

        Returns:
            str: python ast and corresponding source code string.
        """
        dump_str = ""
        dump_str += ColorPalette.FAIL
        dump_str += "\n*****dumping python ast****\n"
        dump_str += astunparse.dump(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += ColorPalette.HEADER
        dump_str += "\n*****dumping python code*****\n"
        dump_str += astunparse.unparse(_ast)
        dump_str += ColorPalette.ENDC
        dump_str += "\n*******************\n"
        return dump_str

    @classmethod
    def parse_mlir(cls, code_path: str) -> MlirNode:
        """
        Parses the code by providing its path
        :param
        """
        return parse_path(code_path, dialects=CHOPPER_DIALECTS)

    @classmethod
    def parse_python(cls, func: Callable) -> ast.AST:
        """parse python source code to python ast node.

        Args:
            func (Callable): python source code.

        Returns:
            ast.AST: python astnode.
        """        
        code_file = inspect.getsourcefile(func)
        code_lines, start_lineno = inspect.getsourcelines(func)
        code = "".join(code_lines)
        code = textwrap.dedent(code)
        pyast = ast.parse(code, filename=code_file)
        ast.increment_lineno(pyast, n=start_lineno - 1)
        return pyast

    @classmethod
    def convert_python_to_mlir(cls, pyast: ast.AST) -> MlirNode:
        """The mian inference that convert python ast to mlir ast in frontend

        Args:
            pyast (ast.AST): python astnode

        Returns:
            MlirNode: mlir astnode that generated via according python astnode 
        """
        from chopper.pass_manager import PastToMlirPassManager

        pass_manager = PastToMlirPassManager()
        pass_manager.register_passes()
        pass_manager.run(pyast)
        
        return pyast.mast_node
