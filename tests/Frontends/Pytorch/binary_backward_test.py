# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *


class OriginalElementwiseBinaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        c = a + b
        d = a + c
        return d


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
        d = a + c
        return d


class HighLevelBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ewsb_module1 = ElementwiseBinaryModule()
        self.ewsb_module2 = ElementwiseBinaryModule()

    def forward(self, a, b):
        x = self.ewsb_module1(a, b)
        y = self.ewsb_module2(a, x)
        return y


lhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
rhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
label = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)

# case 1
"""
class MyFunc(Function):
    @staticmethod
    def forward(ctx, lhs, rhs):
        ctx.save_for_backward(lhs, rhs)
        return lhs * 2 + rhs * 3

    @staticmethod
    def backward(ctx, grad_output):
        lhs, rhs = ctx.saved_tensors
        print("debug", lhs)
        print("debug", rhs)
        return 1.1 * grad_output, 1.3 * grad_output


class ModuleWithFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        d = MyFunc.apply(a, b)
        return d

"""
# output = ModuleWithFunc()(lhs_input, rhs_input)
# loss = torch.sum(output)
# loss.backward()
# print(output)
# print(lhs_input.grad)
# print(rhs_input.grad)
# assert 0


module1 = ElementwiseBinaryModule()
out = module1(lhs_input, rhs_input)
print("output = ", out)

# ANCHOR switch if enable this compile decorator
# TODO fix func signiture first then, automate the backward()
"""
orig_module = OriginalElementwiseBinaryModule()
orig_out = orig_module(lhs_input, rhs_input)
print(orig_out)
print(orig_out.grad_fn)
assert 0
"""
loss = torch.sum(out)
print(loss)
loss.backward()
print("lhs grad = {}".format(lhs_input.grad))
print("rhs grad = {}".format(rhs_input.grad))
print("single module test passed!")

# CHECK: 63.
# CHECK: single module test passed

module2 = HighLevelBlock()
out2 = module2(lhs_input, rhs_input)
loss2 = torch.sum(out2)
print(loss2)
loss2.backward()
print("lhs grad = {}".format(lhs_input.grad))
print("rhs grad = {}".format(rhs_input.grad))
print("nested modules test passed!")

# CHECK: 105.
# CHECK: nested module test passed
