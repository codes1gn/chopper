# RUN: python %s 2>&1 | FileCheck %s --check-prefix=CHECK-DYLIB
# RUN: python %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN

import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

import numpy as np

scalar_exp_tosa = """
module {
  func @scalar_exp_tosa(%arg0 : tensor<f32>) -> tensor<f32> {
    %0 = "tosa.exp"(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
"""

tensor_exp_tosa = """
module {
  func @tensor_exp_tosa(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tosa.exp"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            return self._iree_module[function_name](*args)
        return invoke

binary_dylib_tensor = ireecc.tools.compile_str(
        tensor_exp_tosa,
        input_type="tosa",
        target_backends=["dylib"]
        )

binary_vulkan_tensor = ireecc.tools.compile_str(
        tensor_exp_tosa,
        input_type="tosa",
        target_backends=["vulkan-spirv"]
        )

binary_dylib_scalar = ireecc.tools.compile_str(
        scalar_exp_tosa,
        input_type="tosa",
        target_backends=["dylib"]
        )

binary_vulkan_scalar = ireecc.tools.compile_str(
        scalar_exp_tosa,
        input_type="tosa",
        target_backends=["vulkan-spirv"]
        )

# test scalar on dylib
vm_module = ireert.VmModule.from_flatbuffer(binary_dylib_scalar)
config = ireert.Config(driver_name="dylib")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
_callable = ctx.modules.module["scalar_exp_tosa"]
arg0 = np.array(3.3, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array(7.1, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
result = _callable(arg0)
print("result: ", result)
# CHECK-DYLIB: 27.112652

# test scalar on vulkan
vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
config = ireert.Config(driver_name="vulkan")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
_callable = ctx.modules.module["scalar_exp_tosa"]
arg0 = np.array(1.5, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array(2.2, dtype=np.float32) # np.array([1., 2., 3., 4.], dtype=np.float32)
result = _callable(arg0)
print("result: ", result)
# CHECK-VULKAN: 4.481689

# test tensor on dylib
vm_module = ireert.VmModule.from_flatbuffer(binary_dylib_tensor)
config = ireert.Config(driver_name="dylib")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
_callable = ctx.modules.module["tensor_exp_tosa"]
arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array([1., 2., 3., 4.], dtype=np.float32)
result = _callable(arg0)
print("result: ", result)
# CHECK-DYLIB: [ 2.7182815 7.3890533 20.085512 54.598175 ]

# test scalar on vulkan
vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_tensor)
config = ireert.Config(driver_name="vulkan")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
_callable = ctx.modules.module["tensor_exp_tosa"]
arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array([1., 2., 3., 4.], dtype=np.float32)
result = _callable(arg0)
print("result: ", result)
# CHECK-VULKAN: [ 2.7182817 7.389056 20.085535 54.59815 ]
