//===- ShapeRefinement.cpp - Shape refinement pass ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "Dialect/Atir/IR/AtirDialect.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "Dialect/Atir/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::CHOPPER;
using namespace mlir::CHOPPER::atir;

namespace {

class ShapeRefinementPass
    : public AtirShapeRefinementBase<ShapeRefinementPass> {
  void runOnOperation() override {
    auto func = getOperation();
    // TODO: Implement for real.
    func.walk([](atir::AddOp addOp) {
      auto lhsType = addOp.lhs().getType();
      auto rhsType = addOp.rhs().getType();
      if (lhsType == rhsType) {
        addOp.result().setType(lhsType);
      } else {
        addOp.emitError() << "args has different shape";
      }
    });

    // If the change cascaded to any returns, need to update the function
    // signature.
    Optional<ReturnOp> firstReturnOp;
    func.walk([&](ReturnOp returnOp) {
      if (!firstReturnOp) {
        firstReturnOp = returnOp;
      } else {
        if (returnOp.getOperandTypes() != firstReturnOp->getOperandTypes()) {
          returnOp.emitError() << "after refining shapes, different "
                                  "terminators have different types";
          signalPassFailure();
        }
      }
    });

    assert(firstReturnOp && "function lacks a terminator");
    auto funcType = func.getType();
    SmallVector<Type, 4> resultTypes(firstReturnOp->getOperandTypes().begin(),
                                     firstReturnOp->getOperandTypes().end());
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   resultTypes));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::atir::createShapeRefinementPass() {
  return std::make_unique<ShapeRefinementPass>();
}
