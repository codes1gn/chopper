""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper_frontend.python import PythonRunner
from typing import Callable


def analyse(the_func: Callable) -> None:

    # TODO wrapper this logics into functions
    pyast = PythonRunner.parse_python(the_func)
    print(PythonRunner.dump_python(pyast))
    mlast = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(mlast))


def test_constant():

    # define test function
    def constant1():
        return

    def constant2() -> float:
        return 1.0

    def constant3() -> float:
        arg0 = 1.0
        return arg0
    
    analyse(constant3)


if __name__ == "__main__":
    test_constant()
