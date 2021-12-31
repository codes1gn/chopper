# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable

import math

# refback python module
import chopper_compiler

def launch_and_execute(ir: str, target: str, _input: float) -> float:
    if target == "refbackend":
        # TODO
        # 1. support text
        # 2. make value rets
        _ = chopper_compiler.load_and_execute([
                "placeholder",
                "/root/project/chopper/tests/Compiler/chopper-compiler-runmlir/identity.mlir",
                "-invoke",
                "identity",
                "-arg-value=dense<1.3> : tensor<f32>",
                "-shared-libs=/root/project/chopper/build/lib/libCHOPPERCompilerRuntimeShlib.so"
            ])

    return 0.0

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
    print(PythonRunner.dump_python(pyast))
    atir = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(atir))
    textual_atir = atir.dump()
    print(textual_atir)
    assert textual_atir == expected_textual_atir, "Conversion in frontend not match expected"

    # STEP 2 show the reference result
    _input = 2.3
    print("------ REF RESULTS in CPU -------")
    print(exp_trial_run(_input))

    # STEP 3 run on llvm-X86 backend
    launch_and_execute(textual_atir, 'refbackend', _input)

