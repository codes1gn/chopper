import torch

import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vmvx"))
# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="dylib"))


target_backends_vulkan = "vulkan-spirv"
target_backends_cpu = "vmvx"
target_backends_dylib = "dylib"
target_backends_cuda = "cuda"

def init_iree_vm(file: str, type: str):
    callable_binary = ireecc.tools.compile_file(file, input_type=type, target_backends=[target_backends_vulkan])
    vm_module = ireert.VmModule.from_flatbuffer(callable_binary)
    VKCTX.add_vm_module(vm_module)
    _forward_callable = VKCTX.modules["module"]["main"]
    return _forward_callable


# ireecc.tools.compile_file can accept linalg dialect, when input_type="none"
# input = torch.empty(1, 3, 224, 224).uniform_(-1, 1).detach().numpy()
mu = torch.empty(2,3).uniform_(1, 10).detach().numpy()
sigma = torch.empty(2,3).uniform_(1, 10).detach().numpy()
_forward_callable = init_iree_vm("/home/zp/chopper/output/random_normal/random_normal_mhlo.mlir", "mhlo")

res = _forward_callable()
print(res)


import matplotlib.pyplot as plt
plt.style.use('default')

res = torch.from_numpy(res)
res = torch.mul(res, 100)
res = res.int()

cnt = [0] * 10000

for i in range(len(res)):
    for j in range(len(res[i])):
            cnt[res[i][j]] += 1
            
axis_x = list(range(1,10001))
plt.plot(axis_x, cnt)
# plt.margins(y=0)
# plt.ylim([0,700])
# for i in range(100):
    # plt.bar(i, cnt[i])
    
    
plt.show()

