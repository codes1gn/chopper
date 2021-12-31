# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable




if __name__ == "__main__":
    # TODO(albert) remove these invalid case, they can pass frontend convertion but fail at compiler checks

    # TODO(albert) remove the floating type noting
    def identity_test(arg0) -> float:
        return arg0

    expected_mlir_text = """
    module {
      func @identity_test(%arg0: tensor<f32>) -> tensor<f32> {
        return %arg0 : tensor<f32>
      }
    }
    """

    # TODO(albert) SUPPORTING CURRYIGN AND FUNCTIONAL DATAFLOW PROGRAMMING
    pyast = PythonRunner.parse_python(identity_test)
    atir = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(atir))

    # CHECK: func @identity_test(%arg0: tensor<f32>) -> tensor<f32> {
    # CHECK-NEXT: return %arg0 : tensor<f32>
