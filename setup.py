# -*- coding: utf-8 -*-
# from Chopper.chopper import python
import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

__TOP_DIR_PATH__ = os.path.abspath(os.path.dirname(__file__))
if not __TOP_DIR_PATH__.endswith(os.path.sep):
    __TOP_DIR_PATH__ += os.path.sep

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        self._build_chopper_extension_module(ext)
        self._attach_iree_compiler(ext)
        self._attach_iree_runtime(ext)

    def _attach_iree_compiler(self, ext):
        build_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not build_dir.endswith(os.path.sep):
            build_dir += os.path.sep

        new_dir = build_dir +"chopper/iree"

        mkdir_cmd_str = "mkdir -p " + new_dir
        subprocess.call(
            mkdir_cmd_str,
            shell=True,
        )

        cp_cmd_str = "cp -r " + os.getcwd() + "/iree_build/compiler-api/python_package/iree/compiler" + " " + new_dir
        subprocess.call(
            cp_cmd_str,
            shell=True,
        )

    def _attach_iree_runtime(self, ext):
        build_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not build_dir.endswith(os.path.sep):
            build_dir += os.path.sep

        new_dir = build_dir +"chopper/iree"

        mkdir_cmd_str = "mkdir -p " + new_dir
        subprocess.call(
            mkdir_cmd_str,
            shell=True,
        )

        cp_cmd_str = "cp -r " + os.getcwd() + "/iree_build/bindings/python/iree/runtime/" + " " + new_dir
        subprocess.call(
            cp_cmd_str,
            shell=True,
        )

    def _build_chopper_extension_module(self, ext):
        cmake_generator = "-GNinja"

        build_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not build_dir.endswith(os.path.sep):
            build_dir += os.path.sep

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            # TODO(albert) make it none hardcode
            "-DMLIR_DIR={}".format(__TOP_DIR_PATH__ + "mlir_build/install_dir/lib/cmake/mlir"),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DLLVM_EXTERNAL_LIT={}".format(__TOP_DIR_PATH__ + "mlir_build/bin/llvm-lit"),
            # not used on MSVC, but no harm
            "-DCMAKE_BUILD_TYPE={}".format("DEBUG"),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(build_dir),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
        ]
        build_args = []
        # build_args += ["-j8"]
        # build_args += ["--verbose"]
        # build_args += ["--clean-first"]
        build_args += ["--target", "chopper_compiler"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)

        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        print("pwd:", os.getcwd())
        """
        subprocess.check_call(
            [
                "copy",
                os.getcwd() + "/build/compiler/tools/chopper-compiler/*.so",
                build_dir,
            ],
            cwd=self.build_temp,
        )

        """
        command_str = "cp " + os.getcwd() + "/build/compiler/tools/chopper-compiler/*.so" + " " + build_dir
        print("copy command = ", command_str)
        subprocess.call(
            "cp " + os.getcwd() + "/build/compiler/tools/chopper-compiler/*.so" + " " + build_dir,
            shell=True,
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="chopper",
    version="0.2.0",
    author="Albert Shi, Tianyu Jiang",
    author_email="codefisheng@gmail.com",
    description="Composite AI Compiler Experiment Platform",
    long_description="",
    ext_modules=[
        CMakeExtension("chopper_compiler"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.6.12",
    entry_points={
        # "console_scripts": [
        #     "chopper_runner=chopper.bin.chopper_runner:main",
        # ]
    },
)
