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


def compute_it():
    tosa_file = open(TMP_FILE_TOSA, "r")
    print("------ TOSA IR -------")
    print(tosa_file.read())

    # STEP 3 run on llvm-X86 backend
    print("------ RESULTS in VULKAN GPU -------")
    # TODO(albert), crash may caused by scope, that ctx of iree not freed automatically
    # result = launch_and_execute(textual_atir, 'vulkan', _INPUT)
    print("vulkan backend inited")
    # test scalar on vulkan
    binary_vulkan_scalar = ireecc.tools.compile_file(TMP_FILE_TOSA, input_type="tosa", target_backends=["vulkan-spirv"])

    # clean up the tmp files
    subprocess.run(["rm", tmp_file_atir, tmp_file_tosa])

    # debugging now
    import chopper.iree.runtime.binding as iree_binding

    _avail_devs = iree_binding.HalDriver.query()
    print(_avail_devs)
    _driver = iree_binding.HalDriver.create("vulkan")
    print(_driver)
    _rt_instance = iree_binding.VmInstance()
    print(_rt_instance)
    _hal_device = _driver.create_default_device()
    print(_hal_device)
    _hal_module = iree_binding.create_hal_module(_hal_device)
    print(_hal_module)
    # build VmModule
    _vm_module = iree_binding.VmModule.from_flatbuffer(binary_vulkan_scalar)
    print(_vm_module)
    _vm_context = iree_binding.VmContext(instance=_rt_instance, modules=[_hal_module, _vm_module])
    print(_vm_context)
    _vm_func = _vm_module.lookup_function("exp_trial_run")
    print(_vm_func)

    _arg_list = iree_binding.VmVariantList(1)
    _ret_list = iree_binding.VmVariantList(1)
    print(_arg_list, _ret_list)

    # np.float32   ->   HalElementType.FLOAT_32
    _arg_list.push_buffer_view(_hal_device, np.asarray(1.3), iree_binding.HalElementType.FLOAT_32)
    _vm_context.invoke(_vm_func, _arg_list, _ret_list)

    # extract vm seq to python
    _value = _ret_list.get_variant(0)
    print(_value)

    # STEP 2 show the reference result
    print("------ REF RESULTS in CPU -------")
    ref_result = exp_trial_run(_INPUT)
    print(ref_result)


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

    # TODO, must be scoped with function, since the release call are binded in iree/bindings/python/iree/runtime/binding.h
    # you may see Release() call in deconstruct function, bad

    compute_it()
