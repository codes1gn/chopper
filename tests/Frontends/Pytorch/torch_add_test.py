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
        def forward(self, a, b):
            c = torch.add(a, b)
            return c

        def ref_forward(self, a, b):
            c = torch.add(a, b)
            return c

    lhs_input_ref = torch.empty(shape, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    rhs_input_ref = torch.empty(shape, dtype=torch.float32).uniform_().clone().detach().requires_grad_(True)
    lhs_input_act = lhs_input_ref.clone().detach().requires_grad_(True)
    rhs_input_act = rhs_input_ref.clone().detach().requires_grad_(True)

    test_module = TestModule()
    ref_out = test_module.ref_forward(lhs_input_ref, rhs_input_ref)
    act_out = test_module(lhs_input_act, rhs_input_act)
    # TENSOR EQUAL
    print("reference result =\n", ref_out)
    print("actual result =\n", act_out)
    print("FF TEST RESULT =", torch.allclose(ref_out, act_out))
    ref_loss = torch.sum(ref_out)
    ref_loss.backward()
    act_loss = torch.sum(act_out)
    act_loss.backward()
    print("reference grad = ", lhs_input_ref.grad)
    print("actual grad = ", lhs_input_act.grad)
    print("BP LHS TEST RESULT =", torch.allclose(lhs_input_act.grad, lhs_input_ref.grad))
    print("BP RHS TEST RESULT =", torch.allclose(rhs_input_act.grad, rhs_input_ref.grad))
    return


do_test((2, 3))
# CHECK: FF TEST RESULT = True
# CHECK: BP LHS TEST RESULT = True
# CHECK: BP RHS TEST RESULT = True
do_test((3, 3))
# CHECK: FF TEST RESULT = True
# CHECK: BP LHS TEST RESULT = True
# CHECK: BP RHS TEST RESULT = True
do_test((7, 5))
# CHECK: FF TEST RESULT = True
# CHECK: BP LHS TEST RESULT = True
# CHECK: BP RHS TEST RESULT = True
