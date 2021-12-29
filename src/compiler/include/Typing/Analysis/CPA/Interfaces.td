//===- CPAInterfaces.td - Interfaces to augment CPA --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_TYPING_ANALYSIS_CPA_CPA_INTERFACES
#define CHOPPER_TYPING_ANALYSIS_CPA_CPA_INTERFACES

include "mlir/IR/OpBase.td"

// TODO: Check upstream: there doesn't seem to be a way to define interfaces
// outside of the ::mlir root namespace.
def CHOPPER_TypingCPATypeMapInterface : TypeInterface<
    "CHOPPERTypingTypeMapInterface"> {
  let methods = [
    InterfaceMethod<"Maps an MLIR Type to a CPA::TypeNode.",
      /*retTy=*/ "::mlir::CHOPPER::Typing::CPA::TypeNode *",
      /*methodName=*/ "mapToCPAType",
      /*args=*/ (ins
          "::mlir::CHOPPER::Typing::CPA::Context &":$context)>,
  ];
}

def CHOPPER_CPATypeInferenceOpInterface : OpInterface<
    "CHOPPERTypingCPATypeInferenceOpInterface"> {
  let methods = [
    InterfaceMethod<"Adds CPA constraints for the op.",
      /*retTy=*/ "void",
      /*methodName=*/ "addCPAConstraints",
      /*args=*/ (ins
          "::mlir::CHOPPER::Typing::CPA::Context &":$context)>,
  ];
}

#endif // CHOPPER_TYPING_ANALYSIS_CPA_CPA_INTERFACES
