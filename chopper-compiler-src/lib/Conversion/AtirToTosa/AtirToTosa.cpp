//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/AtirToTosa/AtirToTosa.h"

#include "../PassDetail.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::CHOPPER;



namespace {
class ConvertExpOp : public OpRewritePattern<atir::ExpOp> {
public:
  using OpRewritePattern<atir::ExpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::ExpOp op,
                                PatternRewriter &rewriter) const override {
    // way 2, explicit replace with replaceOp
    auto loc = op->getLoc();
    auto elementTy = op->getOperand(0).getType();
    auto tosa_exp = rewriter.create<tosa::ExpOp>(loc, elementTy, op->getOperand(0));
    rewriter.replaceOp(op, tosa_exp.getResult());

    // way 1, use replaceOpWithNewOp
    // rewriter.replaceOpWithNewOp<tosa::ExpOp>(op, elementTy, op->getOperand(0));

    return success();
  }
};

class ConvertTanhOp : public OpRewritePattern<atir::TanhOp> {
public:
  using OpRewritePattern<atir::TanhOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::TanhOp op,
                                PatternRewriter &rewriter) const override {
    // way 2, explicit replace with replaceOp
    auto loc = op->getLoc();
    auto elementTy = op->getOperand(0).getType();
    auto tosa_tanh = rewriter.create<tosa::TanhOp>(loc, elementTy, op->getOperand(0));
    rewriter.replaceOp(op, tosa_tanh.getResult());

    // way 1, use replaceOpWithNewOp
    // rewriter.replaceOpWithNewOp<tosa::TanhOp>(op, elementTy, op->getOperand(0));

    return success();
  }
};
} // namespace

namespace {
class ConvertAtirToTosa : public ConvertAtirToTosaBase<ConvertAtirToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, tosa::TosaDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    // NOTE: We are keeping this pass around, even though it currently does
    // nothing, in order to avoid having to reintroduce the same
    // boilerplate.
    // change OwningRewritePatternList into RewritePatternSet
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertExpOp>(context);
    patterns.add<ConvertTanhOp>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertAtirToTosaPass() {
  return std::make_unique<ConvertAtirToTosa>();
}
