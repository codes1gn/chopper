//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_CONVERSION_BASICPYTOSTD_PATTERNS_H
#define CHOPPER_CONVERSION_BASICPYTOSTD_PATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
namespace CHOPPER {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

void populateBasicpyToStdPrimitiveOpPatterns(RewritePatternSet &patterns);

} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_CONVERSION_BASICPYTOSTD_PATTERNS_H
