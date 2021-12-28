//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_ATIR_TRANSFORMS_PASSES_H
#define CHOPPER_DIALECT_ATIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace CHOPPER {
namespace atir {

std::unique_ptr<OperationPass<FuncOp>> createShapeRefinementPass();

} // namespace atir

/// Registers all Atir transformation passes.
void registerAtirPasses();

} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_DIALECT_ATIR_TRANSFORMS_PASSES_H
