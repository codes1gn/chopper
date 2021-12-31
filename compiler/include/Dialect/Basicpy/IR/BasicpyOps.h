//===- BasicPyOps.h - Basic python ops --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_BASICPY_IR_BASICPY_OPS_H
#define CHOPPER_DIALECT_BASICPY_IR_BASICPY_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Basicpy/IR/BasicpyOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Basicpy/IR/BasicpyOps.h.inc"

#endif // CHOPPER_DIALECT_BASICPY_IR_BASICPY_OPS_H
