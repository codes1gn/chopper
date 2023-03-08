import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert
import torch

VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vmvx"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="dylib"))

RANDOM_NORMAL_MHLO = "random_normal/random_normal_mhlo.mlir"
STATELESS_RANDOM_NORMAL_MHLO = "random_normal/stateless_random_normal_mhlo.mlir"
IREE_PROCESS_RANDOM_MHLO = "/home/zp/chopper/output/random_normal/mhlo_preprocess_by_iree.mlir"
STATELESS_RANDOM_UNIFORM_MHLO="/home/zp/chopper/output/random_uniform/stateless_random_uniform_mhlo.mlir"
TOSA_RANDOM_NORMAL = "/home/zp/chopper/output/random_normal/random_normal_tosa.mlir"
MY_TOSA_RANDOM_NORMAL = "/home/zp/chopper/tmp/tosa.6ae71d0ed986480fadc32fcaf62d6d30"

TMP_FILE_MHLO = IREE_PROCESS_RANDOM_MHLO
TMP_FILE_TOSA = MY_TOSA_RANDOM_NORMAL


target_backends_vulkan = "vulkan-spirv"
target_backends_cpu = "vmvx"
target_backends_dylib = "dylib"

def init_iree_vm(file: str, type: str):
    callable_binary = ireecc.tools.compile_file(file, input_type=type, target_backends=[target_backends_vulkan])
    vm_module = ireert.VmModule.from_flatbuffer(callable_binary)
    VKCTX.add_vm_module(vm_module)
    _forward_callable = VKCTX.modules["module"]["main"]
    return _forward_callable
                    

mu = torch.zeros(3,5).detach().numpy()
sigma = torch.ones(3,5).detach().numpy()

_forward_callable = init_iree_vm(TMP_FILE_TOSA, "tosa")
res = _forward_callable(mu, sigma)[0]
print(res)