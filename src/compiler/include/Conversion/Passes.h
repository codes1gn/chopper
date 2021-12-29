//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_CONVERSION_PASSES_H
#define CHOPPER_CONVERSION_PASSES_H

namespace mlir {
namespace CHOPPER {

// Registers all CHOPPER conversion passes.
void registerConversionPasses();

} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_CONVERSION_PASSES_H
