//===- IrHelpers.h - Helpers for bridging analysis and IR types -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_TYPING_SUPPORT_CPA_IR_HELPERS_H
#define CHOPPER_TYPING_SUPPORT_CPA_IR_HELPERS_H

#include "Typing/Analysis/CPA/Types.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace CHOPPER {
namespace Typing {
namespace CPA {

/// Creates an array object type with a possibly unknown element type.
/// By convention, arrays have a single type slot for the element type
/// named 'e'.
ObjectValueType *newArrayType(Context &context,
                              ObjectValueType::IrTypeConstructor irCtor,
                              Identifier *typeIdentifier,
                              llvm::Optional<TypeNode *> elementType);

/// Gets the TypeNode associated with the element type for an array allocated
/// via newArrayType.
TypeNode *getArrayElementType(ObjectValueType *arrayType);

/// Creates an ObjectValueType for the given TensorType. The result will
/// reconstruct the original TensorType's structure but with the resolved
/// element type.
ObjectValueType *createTensorLikeArrayType(Context &context,
                                           TensorType tensorType);

/// Creates a default IR type map hook which supports built-in MLIR types
/// that do not implement the analysis interfaces.
Context::IrTypeMapHook createDefaultTypeMapHook();

} // namespace CPA
} // namespace Typing
} // namespace CHOPPER
} // namespace mlir

#endif // CHOPPER_TYPING_SUPPORT_CPA_IR_HELPERS_H
