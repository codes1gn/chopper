""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper_frontend.python import PythonRunner

MLIR_INPUTS = [
    "constant.mlir",
    "identity.mlir",
    "basic.mlir",
    "constant-add.mlir",
    "constant-add-scalar.mlir",
    "scalar.mlir",
    "broadcast.mlir",
    "conv_2d_nchw.mlir",
    "elementwise.mlir",
    "matmul.mlir",
    "invalid-broadcast.mlir",
    "invalid-conv_2d_nchw.mlir",
    "invalid-matmul.mlir",
    "invalid-num-inputs.mlir",
    "multi-output.mlir",
    "mixed-rank.mlir",
    "multiple-ops.mlir",
    # "pad.mlir",
    # "control-flow-basic.mlir",
]


def test_mlir_parser():
    py_runner = PythonRunner()
    for input_file in MLIR_INPUTS:
        print("*****************************************************************")
        print("testing " + input_file + "\n")
        _ = py_runner.parse_mlir(os.path.join(os.path.dirname(__file__), input_file))
        print(py_runner.dump_mlir(_))
        print("")
        print("*****************************************************************")
        print("")
        print("")


def test_python_parser():
    py_runner = PythonRunner()

    def func_a():
        b = 1 + 1
        return b

    _ = py_runner.parse_python(func_a)
    print(py_runner.dump_python(_))


if __name__ == "__main__":
    test_mlir_parser()
    test_python_parser()
