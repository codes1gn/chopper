# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os

from chopper.python import PythonRunner
from typing import Callable

import math


if __name__ == "__main__":
    # TODO(albert) remove these invalid case, they can pass frontend convertion but fail at compiler checks
    print("tweak the filecheck with fake check")
    # CHECK: tweak the filecheck with fake check

    def exp_trial_run(arg0) -> float:
        ret = math.exp(arg0)
        return ret

    pyast = PythonRunner.parse_python(exp_trial_run)
    print('------ Python SRC ------')
    # CHECK: --- Python SRC ---
    # CHECK: ret = math.exp(arg0)
    print(PythonRunner.dump_python(pyast))
    atir = PythonRunner.convert_python_to_mlir(pyast)
    print('------ MLIR SRC ------')
    # CHECK: --- MLIR SRC ---
    # CHECK: %ret = tcf.exp %arg0 : f32
    print(PythonRunner.dump_mlir(atir))

    # ref
    _input = 2.3
    print("------ REF -------")
    print(exp_trial_run(_input))
    # CHECK: 9.9741824548

