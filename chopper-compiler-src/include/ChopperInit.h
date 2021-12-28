//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_INITALL_H
#define CHOPPER_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace CHOPPER {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_INITALL_H
