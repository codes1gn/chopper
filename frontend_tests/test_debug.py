""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper_frontend.python import PythonRunner

DEBUG_MLIR_INPUTS = [
    "return_op_without_operand.mlir",
    # "return_op_with_operand.mlir",
]


def test_debug():
    py_runner = PythonRunner()
    for input_file in DEBUG_MLIR_INPUTS:
        print("*****************************************************************")
        print("testing " + input_file + "\n")
        _ = py_runner.parse_mlir(os.path.join(os.path.dirname(__file__), input_file))
        print(py_runner.dump_mlir(_))
        print("")
        print("*****************************************************************")
        print("")
        print("")


if __name__ == "__main__":
    test_debug()
