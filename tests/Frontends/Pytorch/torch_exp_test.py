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
            ]
        )
        def forward(self, a):
            b = torch.exp(a)
            return b

        def ref_forward(self, a):
            b = torch.exp(a)
            return b

    unary_input_ref = torch.empty(shape, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    unary_input_act = unary_input_ref.clone().detach().requires_grad_(True)

    test_module = TestModule()
    ref_out = test_module.ref_forward(unary_input_ref)
    act_out = test_module(unary_input_act)
    # TENSOR EQUAL
    print("reference result =\n", ref_out)
    print("actual result =\n", act_out)
    print("FF TEST RESULT =", torch.allclose(ref_out, act_out))
    ref_loss = torch.sum(ref_out)
    ref_loss.backward()
    act_loss = torch.sum(act_out)
    act_loss.backward()
    print("reference grad = ", unary_input_ref.grad)
    print("actual grad = ", unary_input_act.grad)
    print("BP UNARY TEST RESULT =", torch.allclose(unary_input_act.grad, unary_input_ref.grad))
    return


do_test((2, 3))
# CHECK: FF TEST RESULT = True
# CHECK: BP UNARY TEST RESULT = True
do_test((3, 3))
# CHECK: FF TEST RESULT = True
# CHECK: BP UNARY TEST RESULT = True
do_test((7, 5))
# CHECK: FF TEST RESULT = True
# CHECK: BP UNARY TEST RESULT = True
