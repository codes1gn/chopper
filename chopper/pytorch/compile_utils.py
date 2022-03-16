from typing import Callable, List, Optional, Tuple, NamedTuple

import numpy as np
import torch
import functools
import chopper_compiler
import chopper.iree.compiler as ireecc
import chopper.iree.runtime as ireert

from .torch_jit_compiler import *
from chopper.scaffold.utils import *

__all__ = [
    "annotate_arguments",
    "compile_callable",
    "dummy_compile_callable",
    "backend",
]


# TODO(albert) refactoring
# this part of code snippets are borrows from torch-mlir to ensure the util
# that torch.nn.Module.forward function has consistent function annotation forms
# to simplify the thin frontend importer compiler.

# TODO: Replace with py3 extended argument annotations when available.
# See https://www.python.org/dev/peps/pep-0593/
def annotate_arguments(annotations: List[Optional[ArgAnnotation]]):
    """Decorator that tells the torch-mlir compiler information about arguments.

    The `annotations` should be a list of the same length as the number of
    argument to the method (including `self`). Each list entry is either:
    - None, corresponding to providing the compiler with no information.
    - A 2-tuple consisting of a shape and a dtype, such as
      `([2, 3, 4], torch.float32)`. A dimension with an unknown size can be
      indicated by using `-1` as the size. This provides the compiler a
      guarantee that the argument will always dynamically have the described
      shape and dtype.
    """

    # TODO: Check the number of arguments matches the number of arg annotations.
    def decorator(fn: Callable) -> Callable:
        setattr(fn, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME, annotations)
        return fn

    return decorator


def compile_callable(fn: Callable) -> Callable:

    # import this Callable
    # use parse Module to inspect this Module.forward method into ast.AST while keep original ret
    #

    # STAGE 1 :: python src => mlir atir dialects
    tjcompiler = TorchJitCompiler()
    ast_source = tjcompiler.parse_callable(fn)
    print("------ PYTHON SRC -------")
    print(tjcompiler.dump_python(ast_source))

    ast_source = tjcompiler.annotate_function(ast_source, fn._torch_dsl_arg_annotations)
    mlir_dialect = tjcompiler.to_mlir_dialect(ast_source)
    print("------ ATIR IR -------")
    textual_atir = mlir_dialect.dump()
    print(textual_atir)

    # STAGE 2 :: mlir atir dialects => TOSA
    TMP_FILE_ATIR = "/tmp/atir.0"
    TMP_FILE_TOSA = "/tmp/tosa.0"

    def compile_it():
        _args = [
            "placeholder",
            TMP_FILE_ATIR,
            "-convert-atir-to-tosa",
            "-o",
            TMP_FILE_TOSA,
        ]
        return chopper_compiler.compile(_args)

    # write atir to tmp file
    atir_file = open(TMP_FILE_ATIR, "w")
    atir_file.write(textual_atir)
    atir_file.close()
    # compile atir ir into tosa ir and saved to TMP_FILE_TOSA
    exit_code_compilation = compile_it()
    if exit_code_compilation:
        raise AssertionError("---- compilation failed ----")
    tosa_file = open(TMP_FILE_TOSA, "r")
    print("------ TOSA IR -------")
    print(tosa_file.read())

    # STAGE 3 IREE branch :: TOSA => spirv-module and executable on IREE
    print("------ RESULTS in VULKAN GPU -------")
    print("vulkan backend inited")
    # test scalar on vulkan
    binary_vulkan_scalar = ireecc.tools.compile_file(TMP_FILE_TOSA, input_type="tosa", target_backends=["vulkan-spirv"])
    vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
    config = ireert.Config(driver_name="vulkan")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    # this part to be replaced by dyn naming
    _callable = ctx.modules.module["forward"]
    # TODO mock with arg0 as self, anyway this are not used
    # result = _callable(arg0, arg0, arg1)

    # STAGE 3 CRT branch :: TOSA => CRT bytecode

    # STAGE 4 CRT branch :: CRT bytecode => execute on CRT

    # print("------ REF RESULTS in CPU -------")
    # ref_result = add_trial_run(_INPUT_LHS, _INPUT_RHS)
    # print(ref_result)
    return _callable
    # return fn


def dummy_compile_callable(*args, **kwargs):
    def _callable_after_args_canonicaliser(fn: Callable) -> Callable:
        print("lalala")
        return fn(*args, **kwargs)

    return _callable_after_args_canonicaliser


def backend(backend_name: str):
    def compiled_callable(fn: Callable) -> Callable:
        # import this Callable
        # use parse Module to inspect this Module.forward method into ast.AST while keep original ret
        #

        # STAGE 1 :: python src => mlir atir dialects
        tjcompiler = TorchJitCompiler()
        ast_source = tjcompiler.parse_callable(fn)
        # print("------ PYTHON SRC -------")
        # print(tjcompiler.dump_python(ast_source))

        ast_source = tjcompiler.annotate_function(ast_source, fn._torch_dsl_arg_annotations)
        mlir_dialect = tjcompiler.to_mlir_dialect(ast_source)
        print("------ ATIR IR -------")
        textual_atir = mlir_dialect.dump()
        print(textual_atir)

        # STAGE 2 :: mlir atir dialects => TOSA
        TMP_FILE_ATIR = "/tmp/atir.0"
        TMP_FILE_TOSA = "/tmp/tosa.0"

        def compile_it():
            _args = [
                "placeholder",
                TMP_FILE_ATIR,
                "-convert-atir-to-tosa",
                "-o",
                TMP_FILE_TOSA,
            ]
            return chopper_compiler.compile(_args)

        # write atir to tmp file
        atir_file = open(TMP_FILE_ATIR, "w")
        atir_file.write(textual_atir)
        atir_file.close()
        # compile atir ir into tosa ir and saved to TMP_FILE_TOSA
        exit_code_compilation = compile_it()
        if exit_code_compilation:
            raise AssertionError("---- compilation failed ----")
        tosa_file = open(TMP_FILE_TOSA, "r")
        print("------ TOSA IR -------")
        print(tosa_file.read())

        # STAGE 3 IREE branch :: TOSA => spirv-module and executable on IREE
        # TODO device init logics should be removed into other part, or make it lazy_load
        # print("------ RESULTS in VULKAN GPU -------")
        # print("vulkan backend inited")
        # test scalar on vulkan
        binary_vulkan_scalar = ireecc.tools.compile_file(
            TMP_FILE_TOSA, input_type="tosa", target_backends=["vulkan-spirv"]
        )
        vm_module = ireert.VmModule.from_flatbuffer(binary_vulkan_scalar)
        config = ireert.Config(driver_name="vulkan")
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        # this part to be replaced by dyn naming
        _callable = ctx.modules.module["forward"]
        # TODO mock with arg0 as self, anyway this are not used
        # result = _callable(arg0, arg0, arg1)

        # STAGE 3 CRT branch :: TOSA => CRT bytecode

        # STAGE 4 CRT branch :: CRT bytecode => execute on CRT

        # print("------ REF RESULTS in CPU -------")
        # ref_result = add_trial_run(_INPUT_LHS, _INPUT_RHS)
        # print(ref_result)
        print("compile callable on {}".format(backend_name))

        @functools.wraps(_callable)
        def wrapper(*args, **kwargs):
            # REMOVE FIRST MODULE ARGS BEFORE CALLING THE COMPILED CALLABLE
            args = [args[k + 1].detach().numpy() for k in range(len(args) - 1)]
            return torch.tensor(_callable(*args, **kwargs))

        return wrapper

    return compiled_callable
