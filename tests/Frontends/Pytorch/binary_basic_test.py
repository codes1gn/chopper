# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np
from typing import Callable, List

from chopper.pytorch import *


# pyast = PythonRunner.parse_python(the_func)
# mlast = PythonRunner.convert_python_to_mlir(pyast)


class ElementwiseBinaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @compile_callable
    @annotate_arguments(
        [
            None,
            ([2, 3], torch.float32),
            ([2, 3], torch.float32),
        ]
    )
    def forward(self, a, b):
        c = a + b
        return c


lhs_input = torch.empty(2, 3).uniform_(0.0, 1.0)
print(lhs_input)
rhs_input = torch.empty(2, 3).uniform_(0.0, 1.0)
arg0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
arg1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
heh = ElementwiseBinaryModule()
print("init module")
out = heh.forward(lhs_input.numpy(), arg0, arg1)
print(out)
# USE __CALL__, NEED TO ADD NEW ARG
# out = ElementwiseBinaryModule()(lhs_input, rhs_input)
# out = ElementwiseBinaryModule().forward(lhs_input, rhs_input)

print("test pass")
# CHECK: test pass
