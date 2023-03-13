//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHLO_DIALECT
#define MHLO_DIALECT

include "mlir/IR/OpBase.td"

def Mhlo_Dialect : Dialect {
    let name = "mhlo";
    let cppNamespace = "::mlir::CHOPPER::mhlo";
    let description = [{
        Custom mhlo dialect to match IREE, which only accepts tosa and mhlo dialect.
        Mainly support functions about rng: rng_unform, rng_normal, rng_bonuli, ... , and so on.
        To support ops above, some auxiliary operations need to be implemented, such as constant, add, mul.

        Op list:
        - constant
        - add 
        - multiply
        - rng_normal
        - rng_uniform
        - rng_other_distribution
    }];
}

class Mhlo_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Mhlo_Dialect, mnemonic, traits> {
}

#endif //ifndef MHLO_DIALECT