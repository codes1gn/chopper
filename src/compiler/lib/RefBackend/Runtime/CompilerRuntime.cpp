//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbols referenced only by the compiler and which will be compiled into a
// shared object that a JIT can load to provide those symbols.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdlib>
#include <iostream>

#include "CompilerDataStructures.h"
#include "RefBackend/Runtime/UserAPI.h"

using namespace refbackrt;

extern "C" void __chopper_compiler_rt_abort_if(bool b, const char *msg) {
  if (b) {
    std::fprintf(stderr, "CHOPPER: aborting: %s\n", msg);
    std::exit(1);
  }
}
