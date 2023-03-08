import pyro
import torch
import uuid
import subprocess

import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

from chopper.pytorch import *
from chopper.scaffold.utils import *
from chopper.pass_manager.symbol_table import feed_forward_symbol_table


uid = "6ae71d0ed986480fadc32fcaf62d6d30"
TMP_FILE_ATIR = "/home/zp/chopper/tmp/atir." + uid
TMP_FILE_TOSA = "/home/zp/chopper/tmp/tosa." + uid
    
    
def random_normal_test(shape):
    class Test(torch.nn.Module):
        @annotate_arguments([
            None,
            (shape, torch.float32),
            (shape, torch.float32)
        ])
        def forward(self, loc, scale):
            x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale), sample_shape = [2])
            return x

    fn = Test.forward
    # STAGE 1 :: python src => mlir atir dialects
    torch_jit_compiler = TorchJitCompiler()
    src_ast = torch_jit_compiler.parse_callable(fn)
    # print(torch_jit_compiler.dump_python(src_ast))

    unique_module_name.set_forward("forward_" + uid)
    unique_module_name.set_backward("backward_" + uid)
    feed_forward_symbol_table.reset_symbol_table()

    ast_source = torch_jit_compiler.annotate_function(src_ast, fn._torch_dsl_arg_annotations)
    mlir_dialect, autodiff_mlir_dialect = torch_jit_compiler.to_mlir_dialect(ast_source)
    print(mlir_dialect.dump())
    print(autodiff_mlir_dialect.dump())
    # print(torch_jit_compiler.dump_mlir(mlir_dialect))
    
    #STAGE2 :: mlir atir dialects to tosa dialect
    subprocess.run(
        [
            "tool-opt",
            TMP_FILE_ATIR,
            "-convert-atir-to-tosa",
            "-o",
            TMP_FILE_TOSA,
        ]
    )
    
    tosa_file = open(TMP_FILE_TOSA, "r")
    print("------ TOSA IR -------")
    textual_tosa = tosa_file.read()
    print(textual_tosa)
    tosa_file.close()
    
random_normal_test((3,5))
    

def sample_grad():
    loc = torch.tensor([0.0,1.0, 2.0, 3.0]).requires_grad_(True)
    scale = torch.tensor([1.0, 1.0, 1.0, 1.0]).requires_grad_(True)
    z = torch.zeros(4)
    o = torch.ones(4)
    a = pyro.sample("my_sample", pyro.distributions.Normal(z, o))
    # a = torch.distributions.normal.Normal(z, o).sample()
    print(a)
    
    a = scale * a + loc
    print(a)

    a.sum().backward()
    print(f"after backward, loc.grad = {loc.grad} and scale.grad = {scale.grad}")
    # loc.grad.zero_()
    # scale.grad.zero_()


# sample_grad()

def sample_batch_test():
    # loc = 0.0
    # scale = 1.0
    # x = pyro.distributions.Normal(loc, scale)
    # print(x)
    # print(type(x))
    
    seed = 0
    torch.manual_seed(seed)   

    a = pyro.sample("my_sample1", pyro.distributions.Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0])))
    b = pyro.sample("my_sample2", pyro.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])))
    c = pyro.sample("my_sample3", pyro.distributions.Normal(torch.tensor([1.0]), torch.tensor([1.0])))
    d = pyro.sample("my_sample4", pyro.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0])))
    e = pyro.sample("my_sample5", pyro.distributions.Normal(torch.tensor([[0.0, 1.0, 2.0],[1.0, 2.0, 3.0]]), torch.tensor([1.0])))
    f = pyro.sample("my_sample6", pyro.distributions.Normal(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([[1.0, 2.0, 3.0],[1.0, 2.0, 3.0]])))

    print("a = ", a)
    print("b = ", b)
    print("c = ", c)
    print("d = ", d)
    print("e = ", e)
    print("f = ", f)
    print("====================================")
    
