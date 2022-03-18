from typing import Callable, List, Optional, Tuple, NamedTuple

import numpy as np
import uuid
import subprocess
import torch
import functools
from torch.autograd import Function
import chopper_compiler
import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

from .torch_jit_compiler import *
from chopper.scaffold.utils import *
from chopper.pass_manager.symbol_table import global_symbol_table

__all__ = [
    "annotate_arguments",
    "backend",
]

# VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))

# todo(albert) refactoring
# this part of code snippets are borrows from torch-mlir to ensure the util
# that torch.nn.module.forward function has consistent function annotation forms
# to simplify the thin frontend importer compiler.

# todo: replace with py3 extended argument annotations when available.
# see https://www.python.org/dev/peps/pep-0593/
def annotate_arguments(annotations: List[Optional[ArgAnnotation]]):
    """decorator that tells the torch-mlir compiler information about arguments.

    the `annotations` should be a list of the same length as the number of
    argument to the method (including `self`). each list entry is either:
    - none, corresponding to providing the compiler with no information.
    - a 2-tuple consisting of a shape and a dtype, such as
      `([2, 3, 4], torch.float32)`. a dimension with an unknown size can be
      indicated by using `-1` as the size. this provides the compiler a
      guarantee that the argument will always dynamically have the described
      shape and dtype.
    """

    # todo: check the number of arguments matches the number of arg annotations.
    def decorator(fn: callable) -> callable:
        setattr(fn, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME, annotations)
        return fn

    return decorator


def backend(backend_name: str):
    def compiled_callable(fn: callable) -> callable:

        # import this Callable
        # use parse Module to inspect this Module.forward method into ast.AST while keep original ret
        #

        # STAGE 1 :: python src => mlir atir dialects
        tjcompiler = TorchJitCompiler()
        ast_source = tjcompiler.parse_callable(fn)
        print("------ PYTHON SRC -------")
        print(tjcompiler.dump_python(ast_source))

        # reset symbol table
        # TODO avoid this action here, make it lazy_load and add scope support,
        # bind the lifetime with the whole compiler
        global_symbol_table.reset_symbol_table()

        ast_source = tjcompiler.annotate_function(ast_source, fn._torch_dsl_arg_annotations)
        mlir_dialect = tjcompiler.to_mlir_dialect(ast_source)
        print("------ ATIR IR -------")
        textual_atir = mlir_dialect.dump()
        print(textual_atir)

        # STAGE 2 :: mlir atir dialects => TOSA
        uid = uuid.uuid4().hex
        TMP_FILE_ATIR = "/tmp/atir." + uid
        TMP_FILE_TOSA = "/tmp/tosa." + uid

        # write atir to tmp file
        atir_file = open(TMP_FILE_ATIR, "w")
        atir_file.write(textual_atir)
        atir_file.close()
        # compile atir ir into tosa ir and saved to TMP_FILE_TOSA
        """
        # TODO(fix): USE CMD SHELL TOOL to avoid core dump, may caused by initialization once but not closed before run twice
        def compile_it():
            _args = [
                "placeholder",
                TMP_FILE_ATIR,
                "-convert-atir-to-tosa",
                "-o",
                TMP_FILE_TOSA,
            ]
            return chopper_compiler.compile(_args)
        exit_code_compilation = compile_it()
        if exit_code_compilation:
            raise AssertionError("---- compilation failed ----")
        """

        # subprocess.run("tool-opt " + TMP_FILE_ATIR + " -convert-atir-to-tosa " + "-o " + TMP_FILE_TOSA)
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
        print(tosa_file.read())
        tosa_file.close()

        # STAGE 3 IREE branch :: TOSA => spirv-module and executable on IREE
        print("------ RESULTS in VULKAN GPU -------")
        print("vulkan backend inited")
        # test scalar on vulkan
        binary_vulkan_scalar = ireecc.tools.compile_file(
            TMP_FILE_TOSA, input_type="tosa", target_backends=["vulkan-spirv"]
        )
        vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
        # clean up the tmp files after all compilation done
        subprocess.run(["rm", TMP_FILE_ATIR])
        subprocess.run(["rm", TMP_FILE_TOSA])

        # TODO mock with arg0 as self, anyway this are not used
        # result = _callable(arg0, arg0, arg1)

        # STAGE 3 CRT branch :: TOSA => CRT bytecode

        # STAGE 4 CRT branch :: CRT bytecode => execute on CRT

        # print("------ REF RESULTS in CPU -------")
        # ref_result = add_trial_run(_INPUT_LHS, _INPUT_RHS)
        # print(ref_result)
        # @functools.wraps(_callable)
        def wrapper(*args, **kwargs):
            # REMOVE FIRST MODULE ARGS BEFORE CALLING THE COMPILED CALLABLE
            # args = [args[k + 1].detach().numpy() for k in range(len(args) - 1)]
            # ret_tensor = torch.tensor(_callable(*args, **kwargs), requires_grad=True)

            class _Callable_Func(Function):
                @staticmethod
                def forward(ctx, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
                    ctx.save_for_backward(lhs, rhs)
                    lhs_data = lhs.detach().numpy()
                    rhs_data = rhs.detach().numpy()

                    # TODO considering change the lifetime of CTX into higher level and let
                    # some entity of the chopper instance to manage it, but has to avoid
                    # duplicate naming of function entries.
                    # shall support uid or hashing for mangling functions
                    VKCTX = ireert.SystemContext(config=ireert.Config(driver_name="vulkan"))
                    VKCTX.add_vm_module(vm_module)
                    # this part to be replaced by dyn naming
                    _callable = VKCTX.modules.module["forward"]
                    return torch.tensor(_callable(lhs_data, rhs_data), requires_grad=True)

                @staticmethod
                def backward(ctx, grad_output):
                    lhs, rhs = ctx.saved_tensors
                    return 1.1 * grad_output, 1.3 * grad_output

            # return ret_tensor
            return _Callable_Func.apply(*args[1:], **kwargs)

        return wrapper

        # return _callable

    return compiled_callable


"""
# LEGACY way to return new callable
@functools.wraps(_callable)
def wrapper(*args, **kwargs):
    # REMOVE FIRST MODULE ARGS BEFORE CALLING THE COMPILED CALLABLE
    # args = [args[k + 1].detach().numpy() for k in range(len(args) - 1)]
    # ret_tensor = torch.tensor(_callable(*args, **kwargs), requires_grad=True)
    class _Callable_Func(Function):
        @staticmethod
        def forward(ctx, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            print(ctx)
            print(lhs)
            print(rhs)
            ctx.save_for_backward(lhs, rhs)
            lhs_data = lhs.detach().numpy()
            rhs_data = rhs.detach().numpy()
            return torch.tensor(_callable(lhs_data, rhs_data), requires_grad=True)

        @staticmethod
        def backward(ctx, grad_output):
            lhs, rhs = ctx.saved_tensors
            return 1.1 * grad_output, 1.3 * grad_output

    def new_forward(self, a, b):
        print(self)
        print(a)
        print(b)
        return _Callable_Func.apply(a, b)

    return new_forward(*args, **kwargs)

return wrapper
"""
