//===- NumpyOps.h - Core numpy dialect ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_NUMPY_IR_NUMPY_OPS_H
#define CHOPPER_DIALECT_NUMPY_IR_NUMPY_OPS_H

#include "Typing/Analysis/CPA/Interfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Numpy/IR/NumpyOps.h.inc"

#endif // CHOPPER_DIALECT_NUMPY_IR_NUMPY_OPS_H