//===- ShapeRefinement.cpp - Shape refinement pass ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "Dialect/Mhlo/IR/MhloDialect.h"
#include "Dialect/Mhlo/IR/MhloOps.h"
#include "Dialect/Mhlo/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::CHOPPER;
using namespace mlir::CHOPPER::mhlo;

namespace {

class MhloCustomPass
    : public MhloCustomPassBase<MhloCustomPass> {
  void runOnOperation() override {
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::mhlo::createMhloCustomPass() {
  return std::make_unique<MhloCustomPass>();
}
