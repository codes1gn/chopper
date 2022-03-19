# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

import torch
import numpy as np

from chopper.pytorch import *


def do_test(shape):
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        @backend("IREE")
        @annotate_arguments(
            [
                None,
                (shape, torch.float32),
                (shape, torch.float32),
            ]
        )
        def forward(self, a):
            b = torch.tanh(a)
            return b

        def ref_forward(self, a):
            b = torch.tanh(a)
            return b

    sole_input = torch.empty(shape, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)

    test_module = TestModule()
    ref_out = test_module.ref_forward(sole_input)
    real_out = test_module(sole_input)
    # TENSOR EQUAL
    print("reference result =\n", ref_out)
    print("actual result =\n", real_out)
    print("TEST RESULT =", torch.allclose(ref_out, real_out))
    # CHECK: TEST RESULT = True
    return


# TODO move it into utils
do_test((2, 3))
do_test((3, 3))
do_test((7, 5))
