//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Mhlo/IR/MhloDialect.h"
#include "Dialect/Mhlo/IR/MhloOps.h"

using namespace mlir;
using namespace mlir::CHOPPER::mhlo;

void MhloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Mhlo/IR/MhloOps.cpp.inc"
      >();
}
