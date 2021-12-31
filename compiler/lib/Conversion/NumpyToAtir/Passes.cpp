//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/NumpyToAtir/Passes.h"

#include "../PassDetail.h"
#include "Dialect/Atir/IR/AtirDialect.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "Dialect/Numpy/IR/NumpyOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::CHOPPER;

namespace {
template <typename TargetTcfOp>
class ConvertBinaryBuiltinUfuncCallOp
    : public OpRewritePattern<Numpy::BuiltinUfuncCallOp> {
public:
  ConvertBinaryBuiltinUfuncCallOp(MLIRContext *context, StringRef qualifiedName,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), qualifiedName(qualifiedName) {}
  LogicalResult matchAndRewrite(Numpy::BuiltinUfuncCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.qualified_name() != qualifiedName)
      return failure();
    if (op.inputs().size() != 2)
      return failure();

    rewriter.replaceOpWithNewOp<TargetTcfOp>(op, op.getResult().getType(),
                                             op.inputs()[0], op.inputs()[1]);
    return success();
  }

private:
  StringRef qualifiedName;
};
} // namespace

namespace {
class ConvertNumpyToAtir : public ConvertNumpyToAtirBase<ConvertNumpyToAtir> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CHOPPER::atir::AtirDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    // change OwningRewritePatternList into RewritePatternSet
    RewritePatternSet patterns(context);
    patterns.add<ConvertBinaryBuiltinUfuncCallOp<atir::AddOp>>(context,
                                                               "numpy.add");
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertNumpyToAtirPass() {
  return std::make_unique<ConvertNumpyToAtir>();
}
