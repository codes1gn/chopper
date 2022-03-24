//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATIR_OPS
#define ATIR_OPS

include "Dialect/Atir/IR/AtirDialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class BinaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  Atir_Op<mnemonic, traits> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  // let arguments = (
  //     ins
  //     TensorRankOf<[F32, I32], [0, 1, 2, 3, 4]>:$lhs,
  //     TensorRankOf<[F32, I32], [0, 1, 2, 3, 4]>:$rhs);
  // let results = (outs
  //    TensorRankOf<[F32, I32], [0, 1, 2, 3, 4]>:$result);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// we shouldn't constraint the same type/shape/elemty, since we should allow casting, broadcasting rule
def Atir_AddOp : BinaryArithmeticOp<"add", [NoSideEffect]> {
  let summary = "Addition of two tensors.";
  let description = [{
    Addition of two tensors.

    Numpy-style broadcasting is allowed.
  }];
}

def Atir_SubOp : BinaryArithmeticOp<"sub", [NoSideEffect]> {
  let summary = "Substraction of two tensors.";
  let description = [{
    Substraction of two tensors.

    Numpy-style broadcasting is allowed.
  }];
}

def Atir_MulOp : BinaryArithmeticOp<"mul", [NoSideEffect]> {
  let summary = "Multiply an input tensor by a scalar tensor.";
  let description = [{
    Multiplies each element of the input `input` with the scalar `other` and returns a new resulting tensor. The tensor types must match and shapes must be broadcastable.
  }];
}

// TODO tosa not support DivOp currently
// even divop is included here: https://mlir.llvm.org/docs/Dialects/TOSA/
// wait for update
//
/*
def Atir_DivOp : BinaryArithmeticOp<"mul", [NoSideEffect]> {
  let summary = "Divtiply an input tensor by a scalar tensor.";
  let description = [{
    Divtiplies each element of the input `input` with the scalar `other` and returns a new resulting tensor. The tensor types must match and shapes must be broadcastable.
  }];
}
*/

def Atir_MaxOp : BinaryArithmeticOp<"max", [NoSideEffect]> {
  let summary = "Maximum of two tensors.";
  let description = [{
    Maximum of two tensors.

    Numpy-style broadcasting is allowed.
  }];
}

class UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
  Atir_Op<mnemonic,
        !listconcat(traits, [AllTypesMatch<["operand", "result"]>])>,
  AllTypesMatch<["operand", "result"]> {
  let arguments = (
      ins
      TensorRankOf<[F32], [0, 1, 2, 3, 4]>:$operand);
  let results = (outs
      TensorRankOf<[F32], [0, 1, 2, 3, 4]>:$result);
  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

def Atir_ExpOp : UnaryArithmeticOp<"exp", [NoSideEffect,
                                           SameOperandsAndResultType,
                                           SameOperandsAndResultShape,
                                           SameOperandsAndResultElementType]> {
  let summary = "base-e exponential";
  let description = [{
    See atir.exp for more details.
  }];
}

def Atir_TanhOp : UnaryArithmeticOp<"tanh", [NoSideEffect,
                                             SameOperandsAndResultType,
                                             SameOperandsAndResultShape,
                                             SameOperandsAndResultElementType]> {
  let summary = "hyperbolic tangent";
  let description = [{
    See atir.tanh for more details.
  }];
}

def Atir_IdentityOp : UnaryArithmeticOp<"identity", [NoSideEffect,
                                             SameOperandsAndResultType,
                                             SameOperandsAndResultShape,
                                             SameOperandsAndResultElementType]> {
  let summary = "identity function";
  let description = [{
    See atir.identity for more details.
  }];
}

def Atir_NegateOp : UnaryArithmeticOp<"negate", [NoSideEffect,
                                             SameOperandsAndResultType,
                                             SameOperandsAndResultShape,
                                             SameOperandsAndResultElementType]> {
  let summary = "negate function";
  let description = [{
    See atir.negate for more details.
  }];
}

// TODO: Generalize this op appropriately and add more verification.
// For example, an unranked operand probably should be allowed and verified
// dynamically in Atir->Ctir lowering if needed.
def Atir_MatmulOp : Atir_Op<"matmul"> {
  let summary = "Performs a matrix multiplication";
  let description = [{
    Performs a matrix multiplication.

    The tensors have dimensions:
    - lhs: [M, K]
    - rhs: [K, N]
    - result: [M, N]

    If the `K` dimension mismatches between the operands, this op aborts the
    program.
  }];
  let arguments = (ins TensorRankOf<[F32], [2, 3]>:$lhs, TensorRankOf<[F32], [2, 3]>:$rhs);
  let results = (outs TensorRankOf<[F32], [2, 3]>:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Atir_Conv2DCFirstOp : Atir_Op<"conv_2d_cfirst"> {
  let summary = "2-D channel first convolution";
  let description = [{
    Performs 2-D convolution. This op is inspired by PyTorch's Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

    The tensors have dimensions:
    - in:     [N, Cin, H, W]
    - filter: [Cout, Cin, KH, KW]
    - result: [N, Cout, Hout, Wout]

    The tensors must meet the following conditions; otherwise, this op aborts the program.
    - H is greater than or equal to KH
    - W is greater than or equal to KW
    - Cin matches between in and filter
  }];
  let arguments = (ins 4DTensorOf<[F32]>:$in, 4DTensorOf<[F32]>:$filter);
  let results = (outs 4DTensorOf<[F32]>:$result);

  let assemblyFormat = "$in `,` $filter attr-dict `:` functional-type(operands, results)";
}


#endif // #ifndef ATIR_OPS
