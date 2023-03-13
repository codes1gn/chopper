//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_CONVERSION_PASSES
#define CHOPPER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// Basicpy conversions
//===----------------------------------------------------------------------===//

def ConvertBasicpyToStd : Pass<"convert-basicpy-to-std", "FuncOp"> {
  let summary = "Convert representable Basicpy ops to std";
  let constructor = "mlir::CHOPPER::createConvertBasicpyToStdPass()";
}

//===----------------------------------------------------------------------===//
// Numpy conversions
//===----------------------------------------------------------------------===//

def ConvertNumpyToAtir : Pass<"convert-numpy-to-atir", "FuncOp"> {
  let summary = "Convert the numpy dialect to supported Atir dialect";
  let constructor = "mlir::CHOPPER::createConvertNumpyToAtirPass()";
}

//===----------------------------------------------------------------------===//
// AtirToCtir
//===----------------------------------------------------------------------===//

def ConvertAtirToLinalg : Pass<"convert-atir-to-linalg", "FuncOp"> {
  let summary = "Convert Atir to Linalg";
  let description = [{
    The intention is for this pass to convert mainly to linalg named ops.

    Because linalg is at the "Ctir" layer of abstraction, this pass has to
    concern itself with generating guards for error cases.
  }];
  let constructor = "mlir::CHOPPER::createConvertAtirToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// AtirToStd
//===----------------------------------------------------------------------===//

def ConvertAtirToStd : Pass<"convert-atir-to-std", "FuncOp"> {
  let summary = "Convert Atir to Std";
  let constructor = "mlir::CHOPPER::createConvertAtirToStdPass()";
}

//===----------------------------------------------------------------------===//
// AtirToCtir
//===----------------------------------------------------------------------===//

def ConvertAtirToCtir : Pass<"convert-atir-to-ctir", "FuncOp"> {
  let summary = "Convert Atir to Ctir";
  let description = [{
    this conversion pass is placeholder
  }];
  let constructor = "mlir::CHOPPER::createConvertAtirToCtirPass()";
}

//===----------------------------------------------------------------------===//
// AtirToTosa
//===----------------------------------------------------------------------===//

def ConvertAtirToTosa : Pass<"convert-atir-to-tosa", "FuncOp"> {
  let summary = "Convert Atir to Tosa";
  let description = [{
    This conversion pass is to lowering the abstract tensor
    operation IR downstream to TOSA dialects, which is the interface
    for temporal use until we refactor the compilation flow to
    a better design
  }];
  let constructor = "mlir::CHOPPER::createConvertAtirToTosaPass()";
}

//===----------------------------------------------------------------------===//
// AtirToMhlo
//===----------------------------------------------------------------------===//

def ConvertAtirToMhlo : Pass<"convert-atir-to-mhlo", "FuncOp"> {
  let summary = "Convert Atir to Mhlo";
  let description = [{
    This conversion pass is to lowering the abstract tensor
    operation IR downstream to MHLO dialects, which is the interface
    for temporal use until we refactor the compilation flow to
    a better design
  }];
  let constructor = "mlir::CHOPPER::createConvertAtirToMhloPass()";
}

#endif // CHOPPER_CONVERSION_PASSES
