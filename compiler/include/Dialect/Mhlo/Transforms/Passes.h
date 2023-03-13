#ifndef CHOPPER_DIALECT_MHLO_TRANSFORMS_PASSES_H
#define CHOPPER_DIALECT_MHLO_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>


namespace mlir {
namespace CHOPPER {
namespace mhlo {

std::unique_ptr<OperationPass<FuncOp>> createMhloCustomPass();

} // namespace mhlo

// Register all Mhlo transformation passes
void registerMhloPasses();

} // namespace CHOPPER
} // namespace mlir

#endif