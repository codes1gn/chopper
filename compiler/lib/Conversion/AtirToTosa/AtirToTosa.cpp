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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

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
    auto tosa_exp =
        rewriter.create<tosa::ExpOp>(loc, elementTy, op->getOperand(0));
    rewriter.replaceOp(op, tosa_exp.getResult());

    // way 1, use replaceOpWithNewOp
    // rewriter.replaceOpWithNewOp<tosa::ExpOp>(op, elementTy,
    // op->getOperand(0));

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
    auto tosa_tanh =
        rewriter.create<tosa::TanhOp>(loc, elementTy, op->getOperand(0));
    rewriter.replaceOp(op, tosa_tanh.getResult());

    // way 1, use replaceOpWithNewOp
    // rewriter.replaceOpWithNewOp<tosa::TanhOp>(op, elementTy,
    // op->getOperand(0));

    return success();
  }
};

class ConvertAddOp : public OpRewritePattern<atir::AddOp> {
public:
  using OpRewritePattern<atir::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::AddOp op,
                                PatternRewriter &rewriter) const override {

    // TODO consider if do broadcast explicitly, or let TOSA do it
    // one ref is AtirToStd
    /***
    // get shape of lhs and rhs
    Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);

    // Create the constraints, and the assuming region.
    Value witness =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
    auto assuming = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{result.getType()}, witness);

    // Start building the region body.
    rewriter.createBlock(&assuming.doRegion());
    */
    // Value broadcastedShape = rewriter.create<shape::BroadcastOp>(
    //     loc, getExtentTensorType(rewriter), lhsShape, rhsShape,
    //     /*error=*/nullptr);

    /*
    // It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                       broadcastedStaticShape);
    auto resultType =
        RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
    Value lhsBroadcasted = rewriter.create<ctir::BroadcastToOp>(
        loc, resultType, lhs, broadcastedShape);
    Value rhsBroadcasted = rewriter.create<tosa::BroadcastToOp>(
        loc, resultType, rhs, broadcastedShape);
    binaryOpResult = rewriter.create<tosa::AddOp>(loc, result.getType(),
                                             lhsBroadcasted, rhsBroadcasted);

    rewriter.create<shape::AssumingYieldOp>(loc, binaryOpResult);

    // Finally, replace with the results of the shape.assuming
    rewriter.replaceOp(op, assuming.getResults());
    */

    // auto elementTy = lhs.getType();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    // Value result = op->getResult(0);
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");
    // It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                       broadcastedStaticShape);
    auto resultType =
        RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
    // in TOSA, no need to explicit handle broadcast, just make the ret type
    // become broadcastable compatible with the lhs and rhs
    // check lhs/rhs, who shape is the resulting broadcastable shape

    auto tosa_op = rewriter.create<tosa::AddOp>(loc, resultType, lhs, rhs);
    rewriter.replaceOp(op, tosa_op.getResult());

    return success();
  }
};

class ConvertSubOp : public OpRewritePattern<atir::SubOp> {
public:
  using OpRewritePattern<atir::SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::SubOp op,
                                PatternRewriter &rewriter) const override {
    // auto elementTy = lhs.getType();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    // Value result = op->getResult(0);
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");
    // It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                       broadcastedStaticShape);
    auto resultType =
        RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
    // in TOSA, no need to explicit handle broadcast, just make the ret type
    // become broadcastable compatible with the lhs and rhs
    // check lhs/rhs, who shape is the resulting broadcastable shape

    auto tosa_op = rewriter.create<tosa::SubOp>(loc, resultType, lhs, rhs);
    rewriter.replaceOp(op, tosa_op.getResult());

    return success();
  }
};

class ConvertMulOp : public OpRewritePattern<atir::MulOp> {
public:
  using OpRewritePattern<atir::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::MulOp op,
                                PatternRewriter &rewriter) const override {
    // auto elementTy = lhs.getType();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    // Value result = op->getResult(0);
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");
    // It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                       broadcastedStaticShape);
    auto shiftAttr = rewriter.getI32IntegerAttr(0);
    auto resultType =
        RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
    // in TOSA, no need to explicit handle broadcast, just make the ret type
    // become broadcastable compatible with the lhs and rhs
    // check lhs/rhs, who shape is the resulting broadcastable shape

    auto tosa_op = rewriter.create<tosa::MulOp>(loc, resultType, lhs, rhs, shiftAttr);
    rewriter.replaceOp(op, tosa_op.getResult());

    return success();
  }
};

// TODO tosa not support DivOp currently
// even divop is included here: https://mlir.llvm.org/docs/Dialects/TOSA/
// wait for update
//
/*
class ConvertDivOp : public OpRewritePattern<atir::DivOp> {
public:
  using OpRewritePattern<atir::DivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(atir::DivOp op,
                                PatternRewriter &rewriter) const override {
    // auto elementTy = lhs.getType();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    Value result = op->getResult(0);
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");
    // It's annoying to do the dynamic broadcast above then
    // do the static transfer function here. Would be nice if they could
    // somehow be unified.
    SmallVector<int64_t, 6> broadcastedStaticShape;
    OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                       broadcastedStaticShape);
    auto resultType =
        RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
    // in TOSA, no need to explicit handle broadcast, just make the ret type
    // become broadcastable compatible with the lhs and rhs
    // check lhs/rhs, who shape is the resulting broadcastable shape

    auto tosa_op = rewriter.create<tosa::DivOp>(loc, resultType, lhs, rhs);
    rewriter.replaceOp(op, tosa_op.getResult());

    return success();
  }
};
*/

// PUT ALL CONVERT PASSES ABOVE
} // namespace

namespace {
class ConvertAtirToTosa : public ConvertAtirToTosaBase<ConvertAtirToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<shape::ShapeDialect, tosa::TosaDialect, math::MathDialect>();
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
    patterns.add<ConvertAddOp>(context);
    patterns.add<ConvertSubOp>(context);
    patterns.add<ConvertMulOp>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertAtirToTosaPass() {
  return std::make_unique<ConvertAtirToTosa>();
}
