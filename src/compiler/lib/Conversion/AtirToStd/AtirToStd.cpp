//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/AtirToStd/AtirToStd.h"

#include "../PassDetail.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "Dialect/Ctir/IR/CtirDialect.h"
#include "Dialect/Ctir/IR/CtirOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::CHOPPER;

static RankedTensorType getExtentTensorType(Builder &builder) {
  return RankedTensorType::get({ShapedType::kDynamicSize},
                               builder.getIndexType());
}

// Non-templated version of the body of ConvertBinaryElementwise to keep things
// simple.
static LogicalResult
matchAndRewriteBinaryElementwise(Operation *op, PatternRewriter &rewriter) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Location loc = op->getLoc();
  Value result = op->getResult(0);

  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType)
    return rewriter.notifyMatchFailure(op, "requires ranked tensors");

  Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
  Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);

  // Create the constraints, and the assuming region.
  Value witness =
      rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
  auto assuming = rewriter.create<shape::AssumingOp>(
      loc, ArrayRef<Type>{result.getType()}, witness);

  // Start building the region body.
  rewriter.createBlock(&assuming.doRegion());
  Value broadcastedShape = rewriter.create<shape::BroadcastOp>(
      loc, getExtentTensorType(rewriter), lhsShape, rhsShape,
      /*error=*/nullptr);

  // TODO: It's annoying to do the dynamic broadcast above then
  // do the static transfer function here. Would be nice if they could
  // somehow be unified.
  SmallVector<int64_t, 6> broadcastedStaticShape;
  OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                     broadcastedStaticShape);
  auto resultType =
      RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
  Value lhsBroadcasted = rewriter.create<ctir::BroadcastToOp>(
      loc, resultType, lhs, broadcastedShape);
  Value rhsBroadcasted = rewriter.create<ctir::BroadcastToOp>(
      loc, resultType, rhs, broadcastedShape);
  Value binaryOpResult;
  if (isa<atir::AddOp>(op)) {
    binaryOpResult = rewriter.create<AddFOp>(loc, result.getType(),
                                             lhsBroadcasted, rhsBroadcasted);
  } else if (isa<atir::MaxOp>(op)) {
    // XXX: remove Ctir dep
    // XXX: remove Ctir ops from Ctir
    auto pred = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhsBroadcasted,
                                        rhsBroadcasted);
    binaryOpResult =
        rewriter.create<SelectOp>(loc, pred, lhsBroadcasted, rhsBroadcasted);
  } else if (isa<atir::MulOp>(op)) {
    binaryOpResult = rewriter.create<MulFOp>(loc, result.getType(),
                                             lhsBroadcasted, rhsBroadcasted);
  } else {
    op->dump();
    llvm::report_fatal_error(
        "unhandled op (see dump above): Atir->Std binary elementwise");
  }
  rewriter.create<shape::AssumingYieldOp>(loc, binaryOpResult);

  // Finally, replace with the results of the shape.assuming
  rewriter.replaceOp(op, assuming.getResults());
  return success();
}

namespace {
template <typename SourceOp>
class ConvertBinaryElementwise : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteBinaryElementwise(op, rewriter);
  }
};
} // namespace

static LogicalResult
matchAndRewriteUnaryElementwise(Operation *op, PatternRewriter &rewriter) {
  if (isa<atir::ExpOp>(op)) {
    rewriter.replaceOpWithNewOp<math::ExpOp>(op, op->getOperand(0));
  } else if (isa<atir::TanhOp>(op)) {
    rewriter.replaceOpWithNewOp<math::TanhOp>(op, op->getOperand(0));
  } else {
    op->dump();
    llvm::report_fatal_error(
        "unhandled op (see dump above): Atir->Ctir unary elementwise");
  }
  return success();
}

namespace {
template <typename SourceOp>
class ConvertUnaryElementwise : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteUnaryElementwise(op, rewriter);
  }
};
} // namespace

namespace {
class ConvertAtirToStd : public ConvertAtirToStdBase<ConvertAtirToStd> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<math::MathDialect, shape::ShapeDialect, ctir::CtirDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertUnaryElementwise<atir::ExpOp>,
                 ConvertUnaryElementwise<atir::TanhOp>>(context);
    patterns.add<ConvertBinaryElementwise<atir::AddOp>,
                 ConvertBinaryElementwise<atir::MaxOp>,
                 ConvertBinaryElementwise<atir::MulOp>>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertAtirToStdPass() {
  return std::make_unique<ConvertAtirToStd>();
}
