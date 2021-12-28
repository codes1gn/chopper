# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "CHOPPER"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".pymlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.chopper_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP", "PYTHONPATH"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.chopper_obj_root, "test")
config.chopper_tools_dir = os.path.join(config.chopper_obj_root, "bin")
config.iree_tools_dir = os.path.join(config.iree_obj_root, "bin")
config.iree_env1 = os.path.join(config.iree_obj_root, "bindings/python")
config.iree_env2 = os.path.join(config.iree_obj_root, "compiler-api/python_package")

# Tweak the PATH and PYTHONPATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)


config.chopper_runtime_shlib = os.path.join(
    config.chopper_obj_root, "lib", "libCHOPPERCompilerRuntimeShlib" + config.llvm_shlib_ext
)

tool_dirs = [config.iree_tools_dir, config.chopper_tools_dir, config.llvm_tools_dir]
tools = [
    "chopper-opt",
    "chopper-translate",
    "chopper-compiler-runmlir",
    "iree-opt",
    "iree-run-module",
    "iree-run-mlir",
    ToolSubst("%chopper_runtime_shlib", config.chopper_runtime_shlib),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
