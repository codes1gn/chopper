import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert
import torch

VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vmvx"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="dylib"))

TMP_FILE_TOSA = "/home/zp/chopper/output/add_tosa.mlir"
# TMP_FILE_MHLO = "/home/zp/chopper/output/stateless_random_uniform_mhlo.mlir"
TMP_FILE_MHLO = "/home/zp/chopper/output/mhlo_preprocess_by_iree.mlir"

#llvm-cpu
#vulkan-spirv
#dylib-llvm-aot
#vmvx

target_backends_vulkan = "vulkan-spirv"
target_backends_cpu = "vmvx"
target_backends_dylib = "dylib"
# callable_binary = ireecc.tools.compile_file(TMP_FILE_TOSA, input_type="tosa", target_backends=["vulkan-spirv"])
callable_binary = ireecc.tools.compile_file(TMP_FILE_MHLO, input_type="mhlo", target_backends=[target_backends_vulkan])


vm_module = ireert.VmModule.from_flatbuffer(callable_binary)
VKCTX.add_vm_module(vm_module)
print(vm_module)
#forward_57e61f4343a64aed8f54e69d11d27266
# _forward_callable = VKCTX.modules["forward_57e61f4343a64aed8f54e69d11d27266"]["forward"]
# a_inference_random_normal_494__.23
_forward_callable = VKCTX.modules["module"]["rng_normal"]


print(_forward_callable)

lhs = torch.empty(7, 5).uniform_()
rhs = torch.empty(7, 5).uniform_()
                    
a = lhs.detach().numpy()
b = rhs.detach().numpy()
print("lhs is:", a)
print("rhs is", b)
print("========================")
c = torch.tensor(0.).detach().numpy()
d = torch.tensor(1.).detach().numpy()
outputs = _forward_callable(c, d)
print("result is ",outputs)