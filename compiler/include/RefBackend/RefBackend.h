//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_REFBACKEND_REFBACKEND_H
#define CHOPPER_REFBACKEND_REFBACKEND_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace CHOPPER {

/// Registers all RefBackend passes.
void registerRefBackendPasses();

// Look in createRefBackendLoweringPipeline for more information about how these
// passes fit together.
//
// Pass summaries are in Passes.td.

std::unique_ptr<OperationPass<FuncOp>> createLowerStructuralToMemrefPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToRefbackrtABIPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerAllocMemRefOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();

std::unique_ptr<Pass> createRestrictedCanonicalizerPass();

struct RefBackendLoweringPipelineOptions
    : public PassPipelineOptions<RefBackendLoweringPipelineOptions> {
  // If this option is true, then perform optimizations.
  // If this option is false, only do the bare minimum for correctness.
  Option<bool> optimize{*this, "optimize", llvm::cl::desc("Do optimizations."),
                        llvm::cl::init(false)};
};

// The main pipeline that encapsulates the full RefBackend lowering.
void createRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options);

// Helper pipeline that runs Atir->Ctir lowering.
//
// For now, just piggy-back on the same set of options since this is such a
// simple set of passes.
//
// TODO: Move this out of RefBackend once the Atir->Ctir conversions
// become more substantial.
void createRefBackendAtirToCtirPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options);

// Helper pipeline that runs Atir->Ctir lowering before invoking
// RefBackendLoweringPipeline.
// For now, just piggy-back on the same set of options since this is such a
// thin wrapper.
// Longer-term, the reference backend should fit into some sort of
// "target interface" and this helper won't be needed.
void createAtirRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options);

} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_REFBACKEND_REFBACKEND_H
