//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/AtirToCtir/AtirToCtir.h"

#include "../PassDetail.h"
#include "Dialect/Atir/IR/AtirOps.h"
#include "Dialect/Ctir/IR/CtirDialect.h"
#include "Dialect/Ctir/IR/CtirOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::CHOPPER;

namespace {
class ConvertAtirToCtir : public ConvertAtirToCtirBase<ConvertAtirToCtir> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, ctir::CtirDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternSet getPatterns() {
    // NOTE: We are keeping this pass around, even though it currently does
    // nothing, in order to avoid having to reintroduce the same
    // boilerplate.
    // change OwningRewritePatternList into RewritePatternSet
    RewritePatternSet patterns(getOperation().getContext());
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CHOPPER::createConvertAtirToCtirPass() {
  return std::make_unique<ConvertAtirToCtir>();
}
