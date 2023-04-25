import torch
import numpy as np

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
import seaborn as sns
plt.style.use('default')

# plt.ylim([0,700])
data = np.array(res).flatten().tolist()

data1 = np.array(res[0:64,:]).flatten().tolist()
data2 = np.array(res[64:128,:]).flatten().tolist()

# plt.hist(data, bins=50)
# plt.hist(data1,bins=50)
# plt.hist(data2,bins=50)

sns.distplot(data2, hist=True, kde=True, bins=100, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

plt.show()

