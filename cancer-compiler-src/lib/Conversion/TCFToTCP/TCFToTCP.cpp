//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/TCFToTCP/TCFToTCP.h"

#include "../PassDetail.h"
#include "Dialect/TCF/IR/TCFOps.h"
#include "Dialect/TCP/IR/TCPDialect.h"
#include "Dialect/TCP/IR/TCPOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::CANCER;

namespace {
class ConvertTCFToTCP : public ConvertTCFToTCPBase<ConvertTCFToTCP> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, tcp::TCPDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    // NOTE: We are keeping this pass around, even though it currently does
    // nothing, in order to avoid having to reintroduce the same
    // boilerplate.
    // change OwningRewritePatternList into RewritePatternSet
    RewritePatternSet patterns(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::CANCER::createConvertTCFToTCPPass() {
  return std::make_unique<ConvertTCFToTCP>();
}
