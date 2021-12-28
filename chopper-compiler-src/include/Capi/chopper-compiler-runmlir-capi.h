//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility binary for compiling and running code through the chopper
// compiler/runtime stack.
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_PYRUNNER_CHOPPER_RUNNER_BACKEND_H
#define CHOPPER_PYRUNNER_CHOPPER_RUNNER_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

int chopperrun(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif // CHOPPER_PYRUNNER_CHOPPER_RUNNER_BACKEND_H
