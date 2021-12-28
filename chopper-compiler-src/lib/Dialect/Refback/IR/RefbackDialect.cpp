//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Refback/IR/RefbackDialect.h"
#include "Dialect/Refback/IR/RefbackOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::CHOPPER::refback;

//===----------------------------------------------------------------------===//
// RefbackDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct RefbackInlinerInterface : public DialectInlinerInterface {
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

void RefbackDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Refback/IR/RefbackOps.cpp.inc"
      >();
  addInterfaces<RefbackInlinerInterface>();
}
