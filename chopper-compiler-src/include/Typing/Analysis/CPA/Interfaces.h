//===- Interfaces.h - Interfaces for IR types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_TYPING_ANALYSIS_CPA_INTERFACES_H
#define CHOPPER_TYPING_ANALYSIS_CPA_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "Typing/Analysis/CPA/Types.h"

namespace mlir {

#include "Typing/Analysis/CPA/OpInterfaces.h.inc"
#include "Typing/Analysis/CPA/TypeInterfaces.h.inc"

} // namespace mlir

#endif // CHOPPER_TYPING_ANALYSIS_CPA_INTERFACES_H
