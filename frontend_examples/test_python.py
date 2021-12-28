""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper_frontend.python import PythonRunner
from typing import Callable

# TODO
# test with function decorator as inputs
# test with python source code inputs
MLIR_INPUTS = [
    # "constant.py",
    # "identity.py",
    # "basic.py",
    # "constant-add.py",
    # "constant-add-scalar.py",
    # "scalar.py",
    # "broadcast.py",
    # "conv_2d_nchw.py",
    # "elementwise.py",
    # "matmul.py",
    # "invalid-broadcast.py",
    # "invalid-conv_2d_nchw.py",
    # "invalid-matmul.py",
    # "invalid-num-inputs.py",
    # "multi-output.py",
    # "mixed-rank.py",
    # "multiple-ops.py",
    # "pad.py",
    # "control-flow-basic.py",
]


def test_constant_add_scalar():
    py_runner = PythonRunner()

    def constant_add_scalar() -> float:
        _ = 1.0
        return _

    _ = py_runner.parse_python(constant_add_scalar)
    print(py_runner.dump_python(_))


def analyse(the_func: Callable) -> None:

    # get python default result
    golden = the_func(3.3)
    print("golden = ", golden)

    # TODO wrapper this logics into functions
    py_runner = PythonRunner()
    pyast = py_runner.parse_python(the_func)
    print(py_runner.dump_python(pyast))
    mlast = py_runner.convert_python_to_mlir(pyast)
    print(py_runner.dump_mlir(mlast))


def test_scalar_add_float():

    # define test function
    def scalar_add_float(arg0: float) -> float:
        _ = arg0 + arg0
        return _

    analyse(scalar_add_float)


def test_python_runner():
    test_scalar_add_float()
    # test_constant_add_scalar()


if __name__ == "__main__":
    test_python_runner()
