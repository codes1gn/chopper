# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable


def analyse(the_func: Callable) -> None:

    # TODO wrapper this logics into functions
    pyast = PythonRunner.parse_python(the_func)
    print(PythonRunner.dump_python(pyast))
    mlast = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(mlast))


if __name__ == "__main__":
    # TODO(albert) remove these invalid case, they can pass frontend convertion but fail at compiler checks
    """
    def constant_test_1():
        return

    def constant_test_2() -> float:
        return 1.0
    """

    def constant_test() -> float:
        arg0 = 1.4
        return arg0

    analyse(constant_test)
    # CHECK: func @constant_test -> f32 {
    # CHECK-NEXT: %arg0 = constant 1.4 : f32
    # CHECK-NEXT: return %arg0 : f32
