//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_CONVERSION_ATIRTOLINALG_ATIRTOLINALG_H
#define CHOPPER_CONVERSION_ATIRTOLINALG_ATIRTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace CHOPPER {
std::unique_ptr<OperationPass<FuncOp>> createConvertAtirToLinalgPass();
}
} // namespace mlir

#endif // CHOPPER_CONVERSION_ATIRTOLINALG_ATIRTOLINALG_H
