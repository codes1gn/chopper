# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable




if __name__ == "__main__":
    # TODO(albert) remove these invalid case, they can pass frontend convertion but fail at compiler checks
    """
    def constant_test_1():
        return

    def constant_test_2() -> float:
        return 1.0
    """

    # TODO(albert) support zero arguments call
    def constant_test(arg0) -> float:
        ret = 1.4
        return ret

    expected_mlir_text = """
    module {
      func @constant_test(%arg0: tensor<f32>) -> tensor<f32> {
        %ret = constant dense<1.4> : tensor<f32>
        return %ret : tensor<f32>
      }
    }
    """

    pyast = PythonRunner.parse_python(constant_test)
    atir = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(atir))
    # TODO(albert), not pass the new settings

    # CHECK: func @constant_test(%arg0: tensor<f32>) -> tensor<f32> {
    # CHECK-NEXT: %ret = constant 1.4 : tensor<f32>
    # CHECK-NEXT: return %ret : tensor<f32>
