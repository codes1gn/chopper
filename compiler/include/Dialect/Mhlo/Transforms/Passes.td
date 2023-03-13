//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_MHLO_PASSES
#define CHOPPER_MHLO_PASSES

include "mlir/Pass/PassBase.td"

def MhloCustomPass : Pass<"mhlo-custom-pass", "FuncOp"> {
  let summary = "Mhlo Custom Pass";
  let constructor = "mlir::CHOPPER::mhlo::createMhloCustomPass()";
}

#endif // CHOPPER_MHLO_PASSES
