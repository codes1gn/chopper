//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ctir/IR/CtirOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::CHOPPER;
using namespace mlir::CHOPPER::ctir;

#define GET_OP_CLASSES
#include "Dialect/Ctir/IR/CtirOps.cpp.inc"