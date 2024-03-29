# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *


def do_test(shape1, shape2):
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def ref_forward(self, a, b):
            c = torch.nn.functional.conv2d(a, b)
            return c

        @backend("IREE")
        @annotate_arguments(
            [
                None,
                (shape1, torch.float32),
                (shape2, torch.float32),
            ]
        )
        def forward(self, a, b):
            c = torch.nn.functional.conv2d(a, b)
            return c

    input1 = torch.empty(shape1, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    input2 = torch.empty(shape2, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    # input3 = torch.empty(shape3, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    # e = torch.matmul(d, c)
    # (shape3, torch.float32),

    test_module = TestModule()
    ref_out = test_module.ref_forward(input1, input2)
    real_out = test_module(input1, input2)
    # TENSOR EQUAL
    print("reference result =\n", ref_out)
    print("actual result =\n", real_out)
    print("TEST RESULT =", torch.allclose(ref_out, real_out))
    # CHECK: TEST RESULT = True
    return


# NHWC, HWCO, 1444, 1148
do_test((4, 4, 4, 4), (1, 1, 4, 4))
# do_test((2, 3), (3, 5), (5, 4))
