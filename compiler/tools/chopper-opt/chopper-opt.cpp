//===- chopper-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <mlir/Pass/PassRegistry.h>

// #include "Dialect/Chopper/ChopperDialect.h"
// #include "Dialect/Basicpy/IR/BasicpyDialect.h"
// TODO should we use global register???
#include "ChopperInit.h"

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

int main(int argc, char **argv) {

  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::CHOPPER::registerAllDialects(registry);
  mlir::CHOPPER::registerAllPasses();

  // TODO delete demo dialect
  // registry.insert<mlir::StandardOpsDialect>();
  // registry.insert<mlir::chopper::ChopperDialect>();
  // registry.insert<mlir::CHOPPER::Basicpy::BasicpyDialect>();

  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();

  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline, registry,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;

  // return failed(
  //     mlir::MlirOptMain(argc, argv, "Chopper optimizer driver\n", registry));
}
