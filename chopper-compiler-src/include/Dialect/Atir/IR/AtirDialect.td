//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATIR_DIALECT
#define ATIR_DIALECT

include "mlir/IR/OpBase.td"

def Atir_Dialect : Dialect {
  let name = "atir";
  let cppNamespace = "::mlir::CHOPPER::atir";
  let description = [{
The `atir` dialect is namely the abstract-tensor-intermediate-representation that
  serves as the entry language into the MLIR eco-system.

It abstracts the commons formalisation of high-level dynamic languages with the operational rules defined on top of the 'tensor' primitive datatype.

Follow design of torch-mlir projects, Some of its key features are:
- Ops that safely report errors, such as mismatching sizes for a matrix
  multiplication.
- Parameters controlling op behavior are dynamic operands, such as
  convolution window sizes.
- Support for a rank-dynamic programming model.
- Support for implicit broadcasting, following the industry-standard numpy
  broadcasting rules.

These features make this dialect interoperate well with highly-dynamic
programming models as are common in many frontends.

This dialect is optimized for compiler analysis and transformation, especially
lowering to lower levels of abstraction in the compiler.
Tensor programs, as represented in this dialect, are not necessarily represented
in the most efficient way for op-by-op execution.
  }];
}

class Atir_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Atir_Dialect, mnemonic, traits> {
}


#endif // #ifndef ATIR_DIALECT
