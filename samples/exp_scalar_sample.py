# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable

import math


if __name__ == "__main__":
    # STEP 1: convert python function object to atir ast
    def exp_trial_run(arg0) -> float:
        ret = math.exp(arg0)
        return ret

    expected_textual_atir = """module {
  func @exp_trial_run(%arg0: tensor<f32>) -> tensor<f32> {
    %ret = atir.exp %arg0 : tensor<f32>
    return %ret : tensor<f32>
  }
}"""

    # TODO(albert): 
    # 0. moving debug print to verbose
    # 1. dump_mlir is not pure dump, but show verbose
    # 2. supporting dataflow style prog, like convert_python_to_mlir().dump()
    pyast = PythonRunner.parse_python(exp_trial_run)
    atir = PythonRunner.convert_python_to_mlir(pyast)
    textual_atir = atir.dump()
    print(textual_atir)
    assert textual_atir == expected_textual_atir, "Conversion in frontend not match expected"



    # STEP LAST ref
    _input = 2.3
    print("------ REF -------")
    print(exp_trial_run(_input))
    # CHECK: 9.9741824548

