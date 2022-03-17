# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *


class ElementwiseBinaryModule2(torch.nn.Module):
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
        d = b + c
        return d


"""
"""


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
        return x


lhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
rhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)

# ANCHOR switch if enable this compile decorator
# TODO fix func signiture first then, automate the backward()
out = ElementwiseBinaryModule()(lhs_input, rhs_input)
out2 = ElementwiseBinaryModule2()(lhs_input, rhs_input)
print("single module test passed!")
module2 = HighLevelBlock()
out2 = module2(lhs_input, rhs_input)
print(out2)
print("hierachical modules test passed!")
# CHECK: test pass
