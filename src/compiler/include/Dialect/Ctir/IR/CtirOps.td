//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CTIR_OPS
#define CTIR_OPS

include "Dialect/Ctir/IR/CtirDialect.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


def Ctir_BroadcastToOp : Ctir_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
Broadcasts `operand` to the shape `shape`.

It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$operand `,` $shape attr-dict `:` functional-type(operands, results)";
}

def Ctir_SplattedOp : Ctir_Op<"splatted"> {
  let summary = "Creates a tensor filled with a particular scalar value.";
  let description = [{
    Creates a tensor of shape `shape` with all elements filled with `splatVal`.

    This op is somewhat redundant with ctir.broadcast_to. However,
    ctir.broadcast_to handles degenerate "size-1" broadcasting which structurally
    cannot happen with this op. So to avoid losing that information, we keep
    this op separate.

    NOTE: The name "splatted" separates it from std.splat, which currently
    only handles statically shaped memrefs.

    TODO: Improve std.splat to take dynamic shapes.
  }];
  let arguments = (ins AnyType:$splatVal, Shape_ExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$splatVal `,` $shape attr-dict `:` functional-type(operands, results)";
}

def Ctir_PadOp : Ctir_Op<"pad"> {
  let summary = "Pads a tensor with a fill value";
  let description = [{
    Pads a tensor with `fillVal` along the borders of each dimension according
    to `lowerExpansion` and `upperExpansion`. Note that this op is unmanaged,
    meaning that it assumes its operands and their shapes are valid.

    The tensors have dimensions:
    - operand:   [D1, D2, ..., DN]
    - lowerExpansion: [L1, L2, ..., LN]
    - upperExpansion: [U1, U2, ..., UN]
    - fillVal:   scalar
    - result:    [D1+L1+U1, D2+L2+U2, ..., DN+LN+UN]
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ExtentTensorType:$lowerExpansion, Shape_ExtentTensorType:$upperExpansion, AnyType:$fillVal);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$operand `,` $lowerExpansion `,` $upperExpansion `,` $fillVal attr-dict `:` functional-type(operands, results)";
}

#endif // CTIR_OPS
