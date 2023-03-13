//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_MHLO_IR_MHLOOPS_H
#define CHOPPER_DIALECT_MHLO_IR_MHLOOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Mhlo/IR/MhloOps.h.inc"


#endif // CHOPPER_DIALECT_MHLO_IR_MHLOOPS_H