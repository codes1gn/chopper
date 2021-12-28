//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_REFBACKRT_IR_REFBACKRTOPS_H
#define CHOPPER_DIALECT_REFBACKRT_IR_REFBACKRTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "Dialect/Refbackrt/IR/RefbackrtOps.h.inc"

#endif // CHOPPER_DIALECT_REFBACKRT_IR_REFBACKRTOPS_H
