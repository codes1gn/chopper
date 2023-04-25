import pyro
import torch
import uuid
import subprocess

import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

from chopper.pytorch import *
from chopper.scaffold.utils import *
from chopper.pass_manager.symbol_table import feed_forward_symbol_table

def random_uniform_test(shape):
    class Test(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
        
        @backend("IREE")
        @set_target_ir("")
        @annotate_arguments([
            None,
            (shape, torch.float32),
            (shape, torch.float32)
        ])
        def forward(self, minval, maxval):
            x = pyro.sample("my_sample", pyro.distributions.Uniform(minval, maxval))
            return x


    test = Test()
    operand1 = torch.zeros(shape).detach()
    operand2 = torch.ones(shape).detach()
    
    # operand1 = torch.empty(shape).uniform_(1, 10).detach()
    # operand2 = torch.empty(shape).uniform_(1, 10).detach()
    
    act_out = test(operand1, operand2)
    ref_out = test.ref_forward(operand1, operand2)
    
    print(f"input = \n{operand1} \n {operand2}")
    print("actual result = ", act_out)
    print("reference res = ", ref_out)


random_uniform_test((2, 3))