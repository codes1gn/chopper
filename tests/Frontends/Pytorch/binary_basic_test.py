# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *


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


lhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
rhs_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
label = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)

heh = ElementwiseBinaryModule()
print("create module instance")
# ANCHOR switch if enable this compile decorator
# TODO fix func signiture first then, automate the backward()
# out = heh(arg0, arg1)
out = heh(lhs_input, rhs_input)
# loss = torch.nn.functional.binary_cross_entropy_with_logits(label, out)
print(out)
# loss.backward()
# print(lhs_input.grad)
# USE __CALL__, NEED TO ADD NEW ARG
# out = ElementwiseBinaryModule()(lhs_input, rhs_input)
# out = ElementwiseBinaryModule().forward(lhs_input, rhs_input)

print("test pass")
# CHECK: test pass
