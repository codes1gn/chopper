//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHLO_OPS
#define MHLO_OPS

include "Dialect/Mhlo/IR/MhloDialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


// Op-list 
//  - constant
//  - add 
//  - multiply
//  - rng_normal
//  - rng_uniform
class Mhlo_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
    Mhlo_Op<mnemonic, traits> {
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Mhlo_MulOp : Mhlo_BinaryElementwiseOp<"multiply", 
        [Commutative, NoSideEffect, SameOperandsAndResultType, Elementwise]> {
  let summary = "Multiplication operator";
  let description = [{
    Returns `lhs * rhs` element-wise.
  }];
  // let assemblyFormat = "attr-dict $lhs `,` ";
}

def Mhlo_AddOp : Mhlo_BinaryElementwiseOp<"add",
        [Commutative, NoSideEffect, SameOperandsAndResultType, Elementwise]> {
  let summary = "Addition operator";
  let description = [{
    Returns `lhs + rhs` element-wise.
  }];
}

def Mhlo_SubOp : Mhlo_BinaryElementwiseOp<"subtract", 
        [Commutative, NoSideEffect, SameOperandsAndResultType, Elementwise]> {
  let summary = "Subtraction operator";
  let description = [{
    Returns `lhs - rhs` element-wise.
  }];
}

def Mhlo_ConstOp : Mhlo_Op<"constant",
    [NoSideEffect, AllTypesMatch<["value", "output"]>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    AnyTensor:$output
  );

  let assemblyFormat = "attr-dict $value";
}

def Mhlo_ReshapeOp : Mhlo_Op<"reshape", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Reshape operator";
  let description = [{
    Reshape the input tensor.
  }];

  let arguments = (ins 
    TensorRankOf<[F32], [0, 1, 2, 3, 4]>:$operand, 
    I64ArrayAttr:$new_shape);
  let results = (outs  TensorRankOf<[F32], [0, 1, 2, 3, 4]>:$result);
}

def MHLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;
// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def MHLO_SInt : SignlessIntOfWidths<[8, 16, 32, 64]>;
def MHLO_UInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;
def MHLO_Int : AnyTypeOf<[MHLO_SInt, MHLO_UInt]>;
def MHLO_PredIntOrFpTensor : TensorOf<[MHLO_Pred, MHLO_Int, AnyFloat]>;
def MHLO_DimensionValue : AnyTypeOf<[Index, MHLO_Pred, MHLO_Int]>;
// Dynamic representation of a shape vector as a tensor.
def MHLO_DimensionTensor : 1DTensorOf<[MHLO_DimensionValue]>;

//===----------------------------------------------------------------------===//
// MHLO RNG Operators.
//===----------------------------------------------------------------------===//

def MHLO_RngUniformOp : Mhlo_Op<"rng_uniform", [NoSideEffect]> {
  let summary = "RNG with uniform distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the uniform distribution over the interval `[a,b)`. The parameters
    and output element type have to be a boolean type, an integral type or a
    floating point types, and the types have to be consistent.
  }];
  let arguments = (ins
    MHLO_PredIntOrFpTensor:$a,
    MHLO_PredIntOrFpTensor:$b,
    MHLO_DimensionTensor:$shape
  );

  let results = (outs TensorOf<[AnyFloat]>:$output);
  // let assemblyFormat = "$a `,` $b `,` $shape attr-dict `:` functional-type(operands, results)";
}

def MHLO_RngNormalOp : Mhlo_Op<"rng_normal", [NoSideEffect]> {
  let summary = "RNG with normal distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the normal distribution with parameters `mu` and `sigma`. The
    parameters and output shape have to have a floating point elemental type.
    The parameters furthermore have to be scalar valued.
  }];
  let arguments = (ins
    MHLO_PredIntOrFpTensor:$mu,
    MHLO_PredIntOrFpTensor:$sigma,
    MHLO_DimensionTensor:$shape
  );

  let results = (outs TensorOf<[AnyFloat]>:$output);
  // let assemblyFormat = [{
  //   `(` $mu `,` $sigma `,` $shape `:` type($mu) `,` type($sigma) `,` type($shape) `)` attr-dict `to` type(results)
  // }];
  // let assemblyFormat = "$mu `,` $sigma `,` $shape attr-dict `:` functional-type(operands, results)";
}

#endif // MHLO_OPS