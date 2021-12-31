//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CTIR_DIALECT
#define CTIR_DIALECT

include "mlir/IR/OpBase.td"

def Ctir_Dialect : Dialect {
  let name = "ctir";
  let cppNamespace = "::mlir::CHOPPER::ctir";
  let description = [{
The `ctir` dialect is the gateway to MLIR's code generation infrastructure.
It is also a great place to do algebraic transformations making use of
semantically-charged named ops.

Features:
- Requires ranked tensors (except for a handful of special ops).
- No implicit broadcasting.
- Performance-critical parameters like convolution window sizes are represented
  with attributes.
- Attention to detail modeling ops that are logically "pure" but have
  preconditions.

Together these features allow a relatively large class of "common-sense"
optimizations to be done with only modestly complex considerations.
// TODO: consider having these ops take a "witness" argument
// that makes them truly NoSideEffect?
// Or have a totally pure "ctir.island" op?
// Figure it out when doing the atir to ctir lowering.
  }];
}


class Ctir_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Ctir_Dialect, mnemonic, traits> {
}

#endif // CTIR_DIALECT
