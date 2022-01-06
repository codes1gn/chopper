# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable

import math
import numpy as np

# refback python module
import chopper_compiler
import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

TMP_FILE_ATIR = "/tmp/atir.0"
TMP_FILE_TOSA = "/tmp/tosa.0"
_INPUT = 1.3

def compile_it():
    _args = [
            "placeholder",
            TMP_FILE_ATIR,
            "-convert-atir-to-tosa",
            "-o",
            TMP_FILE_TOSA,
            ]
    return chopper_compiler.compile(_args)

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
        return _
    elif target == "vulkan":
        print("vulkan backend inited")
        # test scalar on vulkan
        binary_vulkan_scalar = ireecc.tools.compile_file(
            TMP_FILE_TOSA,
            input_type="tosa",
            target_backends=["vulkan-spirv"]
        )
        vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
        config = ireert.Config(driver_name="vulkan")
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        _callable = ctx.modules.module["exp_trial_run"]
        arg0 = np.array(_INPUT, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
        result = _callable(arg0)
        return result
    else:
        assert 0, "Not Supported backend option"

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
    print("------ PYTHON SRC -------")
    print(PythonRunner.dump_python(pyast))
    print("------ ATIR IR -------")
    print(textual_atir)
    assert textual_atir == expected_textual_atir, "Conversion in frontend not match expected"

    atir_file = open(TMP_FILE_ATIR, "w")
    atir_file.write(textual_atir)
    atir_file.close()

    # STEP 2 compile atir ir into tosa ir and saved to TMP_FILE_TOSA
    exit_code_compilation = compile_it()
    if exit_code_compilation:
        raise AssertionError("---- compilation failed ----")
    tosa_file = open(TMP_FILE_TOSA, "r")
    print("------ TOSA IR -------")
    print(tosa_file.read())

    # STEP 3 run on llvm-X86 backend
    print("------ RESULTS in VULKAN GPU -------")
    # TODO(albert), crash may caused by scope, that ctx of iree not freed automatically
    # result = launch_and_execute(textual_atir, 'vulkan', _INPUT)
    print("vulkan backend inited")
    # test scalar on vulkan
    binary_vulkan_scalar = ireecc.tools.compile_file(
        TMP_FILE_TOSA,
        input_type="tosa",
        target_backends=["vulkan-spirv"]
    )
    vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
    config = ireert.Config(driver_name="vulkan")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    _callable = ctx.modules.module["exp_trial_run"]
    arg0 = np.array(_INPUT, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
    result = _callable(arg0)
    print(result)

    # STEP 2 show the reference result
    print("------ REF RESULTS in CPU -------")
    ref_result = exp_trial_run(_INPUT)
    print(ref_result)


