//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ctir/IR/CtirDialect.h"
#include "Dialect/Ctir/IR/CtirOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::CHOPPER::ctir;

//===----------------------------------------------------------------------===//
// CtirDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct CtirInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void CtirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Ctir/IR/CtirOps.cpp.inc"
      >();
  addInterfaces<CtirInlinerInterface>();
}
