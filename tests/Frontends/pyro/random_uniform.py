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
        @annotate_arguments([
            None,
            (shape, torch.float32),
            (shape, torch.float32)
        ])
        def forward(self, minval, maxval):
            x = pyro.sample("my_sample", pyro.distributions.Uniform(minval, maxval))
            return x

    fn = Test.forward
    torch_jit_compiler = TorchJitCompiler()
    src_ast = torch_jit_compiler.parse_callable(fn)
    print(torch_jit_compiler.dump_python(src_ast))

    uid = "6ae71d0ed986480fadc32fcaf62d6d30"
    TMP_FILE_ATIR = "/home/zp/chopper/tmp/atir." + uid
    TMP_FILE_TOSA = "/home/zp/chopper/tmp/tosa." + uid
    unique_module_name.set_forward("forward_" + uid)
    unique_module_name.set_backward("backward_" + uid)
    feed_forward_symbol_table.reset_symbol_table()

    ast_source = torch_jit_compiler.annotate_function(src_ast, fn._torch_dsl_arg_annotations)
    mlir_dialect, autodiff_mlir_dialect = torch_jit_compiler.to_mlir_dialect(ast_source)
    print(mlir_dialect.dump())
random_uniform_test((2, 3))