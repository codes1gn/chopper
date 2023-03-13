//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_MHLO_TRANSFORMS_PASSDETAIL_H
#define CHOPPER_DIALECT_MHLO_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace CHOPPER {
namespace mhlo {

#define GEN_PASS_CLASSES
#include "Dialect/Mhlo/Transforms/Passes.h.inc"

} // namespace mhlo
} // namespace CHOPPER
} // end namespace mlir

#endif // CHOPPER_DIALECT_MHLO_TRANSFORMS_PASSDETAIL_H
