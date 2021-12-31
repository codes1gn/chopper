//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Conversion/Passes.h"

#include "Conversion/AtirToCtir/AtirToCtir.h"
#include "Conversion/AtirToLinalg/AtirToLinalg.h"
#include "Conversion/AtirToStd/AtirToStd.h"
#include "Conversion/AtirToTosa/AtirToTosa.h"
#include "Conversion/BasicpyToStd/Passes.h"
#include "Conversion/NumpyToAtir/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
} // end namespace

void mlir::CHOPPER::registerConversionPasses() { ::registerPasses(); }
