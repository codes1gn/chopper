//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REFBACKRT_BASE
#define REFBACKRT_BASE

include "mlir/IR/OpBase.td"

def Refbackrt_Dialect : Dialect {
  let name = "refbackrt";
  let cppNamespace = "::mlir::CHOPPER::refbackrt";
  let description = [{
The `refbackrt` dialect is the IR manifestation for interaction with the
reference backend runtime. It primarily serves as a layer that enapsulates the
data structures and functions available in the runtime, and faciliates
conversion to those conventions, such as by providing utilities for being
lowered to the llvm dialect.
  }];
}

#endif // #ifndef REFBACKRT_BASE
