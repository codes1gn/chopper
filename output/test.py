import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert
import torch

VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vmvx"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="dylib"))

TMP_FILE_TOSA = "/home/zp/chopper/output/add_tosa.mlir"
# TMP_FILE_MHLO = "/home/zp/chopper/output/stateless_random_uniform_mhlo.mlir"
BERT_MHLO = "/home/zp/code/torch-mlir/bert_tiny_mhlo.mlir"
RANDOM_NORMAL_MHLO = "random_normal/random_normal_mhlo.mlir"
STATELESS_RANDOM_NORMAL_MHLO = "random_normal/stateless_random_normal_mhlo.mlir"
IREE_PROCESS_RANDOM_MHLO = "/home/zp/chopper/output/random_normal/mhlo_preprocess_by_iree.mlir"
TMP_FILE_MHLO = RANDOM_NORMAL_MHLO

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
# print(vm_module)
_forward_callable = VKCTX.modules["module"]["main"]


# print(_forward_callable)

# lhs = torch.empty(2, 3).uniform_()
# rhs = torch.empty(2, 3).uniform_()
                    
# a = lhs.detach().numpy()
# b = rhs.detach().numpy()
# print("lhs is:", a)
# print("rhs is", b)
# print("========================")
for _ in range(10):
    res = _forward_callable() #torch.tensor(0.0).numpy(), torch.tensor(1.0).numpy())
    print(res, "\n")
    