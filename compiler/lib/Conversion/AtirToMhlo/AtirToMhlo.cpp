//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/AtirToMhlo/AtirToMhlo.h"

#include "../PassDetail.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "Dialect/Mhlo/IR/MhloOps.h"
#include "Dialect/Mhlo/IR/MhloDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <random>
#include <numeric>
#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::CHOPPER;

namespace {
class ConvertRngNormalOp: public OpRewritePattern<atir::RngNormalOp> {
public:
  using OpRewritePattern<atir::RngNormalOp>::OpRewritePattern;

  /**
   * %x = atir.rng_normal %mu , %sigma , %my_sample_shape : (tensor<3x5xf32>, tensor<3x5xf32>, tensor<2xi32>) -> tensor<3x5xf32>
   *   |
   *   |
   *  \ /
   * %0 = mhlo.constant 0.0 : tensor<f32>
   * %1 = mhlo.constant 1.0 : tensor<f32>
   * %2 = mhlo.constant shape : tensor<?xi64>
   * %3 = mhlo.rng_normal(%0, %1, %2)
   * %4 = mhlo.multiply %3, %sigma
   * %5 = mhlo.add %4, %mu
   * return %5
  */
  LogicalResult matchAndRewrite(atir::RngNormalOp op, PatternRewriter &rewriter) const override {
    Value mu = op.mu();
    Value sigma = op.sigma();
    Value result = op.getResult();
    Location loc = op.getLoc();

    RankedTensorType rngArgType = RankedTensorType::get({}, rewriter.getF32Type());
    Value origin_mu = rewriter.create<mhlo::ConstOp>(
      loc, rngArgType, 
      DenseElementsAttr::get(rngArgType, rewriter.getF32FloatAttr(0.0)));
    Value origin_scale = rewriter.create<mhlo::ConstOp>(
      loc, rngArgType, 
      DenseElementsAttr::get(rngArgType, rewriter.getF32FloatAttr(1.0)));
    auto shapeType = RankedTensorType::get({op.getType().dyn_cast<RankedTensorType>().getRank()}, rewriter.getI64Type());
    auto shapeVal = result.getType().dyn_cast<RankedTensorType>().getShape();
    Value shapeArg = rewriter.create<mhlo::ConstOp>(
      loc,shapeType,
      DenseElementsAttr::get(shapeType, shapeVal));
    
    Value res = rewriter.create<mhlo::RngNormalOp>(loc, result.getType(), origin_mu, origin_scale, shapeArg);
    res = rewriter.create<mhlo::MulOp>(loc, result.getType().dyn_cast<RankedTensorType>(), res, sigma);
    res = rewriter.create<mhlo::AddOp>(loc, result.getType().dyn_cast<RankedTensorType>(), res, mu);
    rewriter.replaceOp(op, res);
    return success();
  }
};
} // namespace

namespace {
class ConvertRngUniformOp: public OpRewritePattern<atir::RngUniformOp> {
public:
  using OpRewritePattern<atir::RngUniformOp>::OpRewritePattern;

  /**
   * %x = atir.rng_uniform %minval , %maxval , %my_sample_shape : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
   *    |
   *    |
   *   \ /
   * %0 = mhlo.constant 0.0 : tensor<f32>
   * %1 = mhlo.constant 1.0 : tensor<f32>
   * %2 = mhlo.constant shape : tensor<?xi64>
   * %3 = mhlo.rng_uniform(%0, %1, %2)
   * %4 = mhlo.sub %maxval, %minval
   * %5 = mhlo.multiply %3, %5
   * %6 = mhlo.add %5, %minval
   * return %6
  */
  LogicalResult matchAndRewrite(atir::RngUniformOp op, PatternRewriter &rewriter) const override {
    Value minval = op.minval();
    Value maxval = op.maxval();
    Value result = op.getResult();
    Location loc = op.getLoc();

    RankedTensorType rngArgType = RankedTensorType::get({}, rewriter.getF32Type());
    Value origin_min = rewriter.create<mhlo::ConstOp>(
      loc, rngArgType, 
      DenseElementsAttr::get(rngArgType, rewriter.getF32FloatAttr(0.0)));
    Value origin_max = rewriter.create<mhlo::ConstOp>(
      loc, rngArgType, 
      DenseElementsAttr::get(rngArgType, rewriter.getF32FloatAttr(1.0)));
    auto shapeType = RankedTensorType::get({op.getType().dyn_cast<RankedTensorType>().getRank()}, rewriter.getI64Type());
    auto shapeVal = result.getType().dyn_cast<RankedTensorType>().getShape();
    Value shapeArg = rewriter.create<mhlo::ConstOp>(
      loc,shapeType,
      DenseElementsAttr::get(shapeType, shapeVal));
    
    Value res = rewriter.create<mhlo::RngUniformOp>(loc, result.getType(), origin_min, origin_max, shapeArg);
    Value scale = rewriter.create<mhlo::SubOp>(loc, result.getType(), maxval, minval);

    res = rewriter.create<mhlo::MulOp>(loc, result.getType().dyn_cast<RankedTensorType>(), res, scale);
    res = rewriter.create<mhlo::AddOp>(loc, result.getType().dyn_cast<RankedTensorType>(), res, minval);
    rewriter.replaceOp(op, res);
    return success();
  }
};
} // namespace

namespace {
class ConvertAddOp: public OpRewritePattern<atir::AddOp> {
public:
  using OpRewritePattern<atir::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(atir::AddOp op, PatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace

namespace {
class ConvertMulOp: public OpRewritePattern<atir::MulOp> {
public:
  using OpRewritePattern<atir::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(atir::MulOp op, PatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace

namespace {
class ConvertConstOp: public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp op, PatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace

namespace {
class ConvertIdentityOp: public OpRewritePattern<atir::IdentityOp> {
public:
  using OpRewritePattern<atir::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(atir::IdentityOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    auto res_type = op.result().getType().dyn_cast<RankedTensorType>();
    auto res = rewriter.create<mhlo::ReshapeOp>(loc, res_type, input, rewriter.getI64ArrayAttr(res_type.getShape()));
    
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};
}
namespace {
class ConvertAtirToMhlo : public ConvertAtirToMhloBase<ConvertAtirToMhlo> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<shape::ShapeDialect, mhlo::MhloDialect, tosa::TosaDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternSet getPatterns() {
    // NOTE: We are keeping this pass around, even though it currently does
    // nothing, in order to avoid having to reintroduce the same
    // boilerplate.
    // change OwningRewritePatternList into RewritePatternSet
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertRngNormalOp>(context);
    patterns.add<ConvertRngUniformOp>(context);
    patterns.add<ConvertAddOp>(context);
    patterns.add<ConvertMulOp>(context);
    patterns.add<ConvertConstOp>(context);
    patterns.add<ConvertIdentityOp>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertAtirToMhloPass() {
  return std::make_unique<ConvertAtirToMhlo>();
}
