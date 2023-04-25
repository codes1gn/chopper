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


# _forward_callable = init_iree_vm("/home/zp/chopper/output/random_normal/random_normal_mhlo.mlir", "mhlo")

# res = _forward_callable()

exp = torch.distributions.Exponential(torch.tensor(1.0))
shape = (128, 128)
normal = np.random.random(shape)
normal_tensor = torch.from_numpy(normal)
res = exp.icdf(normal_tensor)
print(res)


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

data = np.array(res).flatten().tolist()

sns.distplot(data, hist=True, kde=False, bins=100, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
plt.show()

