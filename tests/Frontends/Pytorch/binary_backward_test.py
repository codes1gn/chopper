# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *
from torch.autograd import gradcheck


class ElementwiseBinaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @backend("IREE")
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


class HighLevelBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ewsb_module1 = ElementwiseBinaryModule()
        self.ewsb_module2 = ElementwiseBinaryModule()

    def forward(self, a, b):
        x = self.ewsb_module1(a, b)
        return x


lhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
rhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)

module1 = ElementwiseBinaryModule()
out = module1(lhs_input, rhs_input)
loss = torch.sum(out)
loss.backward()
print("lhs grad = {}".format(lhs_input.grad))
print("rhs grad = {}".format(rhs_input.grad))
print("single module test passed!")
# CHECK: single module test passed

module2 = HighLevelBlock()
out2 = module2(lhs_input, rhs_input)
print("nested modules test passed!")
loss = torch.sum(out2)
loss.backward()

# test = gradcheck(module2, (lhs_input, rhs_input))
# print(test)

# TODO use autograd verify utils

# CHECK: nested modules test passed
