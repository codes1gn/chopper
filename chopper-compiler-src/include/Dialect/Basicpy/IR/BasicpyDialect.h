//===- BasicPyDialect.h - Basic Python --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHOPPER_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
#define CHOPPER_DIALECT_BASICPY_IR_BASICPY_DIALECT_H

#include "Typing/Analysis/CPA/Interfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace CHOPPER {
namespace Basicpy {

namespace detail {
struct SlotObjectTypeStorage;
} // namespace detail

/// Python 'bool' type (can contain values True or False, corresponding to
/// i1 constants of 0 or 1).
class BoolType : public Type::TypeBase<BoolType, Type, TypeStorage> {
public:
  using Base::Base;
  static BoolType get(MLIRContext *context) { return Base::get(context); }
};

/// The type of the Python `bytes` values.
class BytesType : public Type::TypeBase<BytesType, Type, TypeStorage> {
public:
  using Base::Base;
  static BytesType get(MLIRContext *context) { return Base::get(context); }
};

/// Python 'dict' type.
class DictType : public Type::TypeBase<DictType, Type, TypeStorage> {
public:
  using Base::Base;
  static DictType get(MLIRContext *context) { return Base::get(context); }
};

/// The type of the Python `Ellipsis` value.
class EllipsisType : public Type::TypeBase<EllipsisType, Type, TypeStorage> {
public:
  using Base::Base;
  static EllipsisType get(MLIRContext *context) { return Base::get(context); }
};

/// Python 'list' type.
class ListType : public Type::TypeBase<ListType, Type, TypeStorage> {
public:
  using Base::Base;
  static ListType get(MLIRContext *context) { return Base::get(context); }
};

/// The type of the Python `None` value.
class NoneType : public Type::TypeBase<NoneType, Type, TypeStorage> {
public:
  using Base::Base;
  static NoneType get(MLIRContext *context) { return Base::get(context); }
};

class SlotObjectType : public Type::TypeBase<SlotObjectType, Type,
                                             detail::SlotObjectTypeStorage> {
public:
  using Base::Base;
  static SlotObjectType get(StringAttr className, ArrayRef<Type> slotTypes);
  StringAttr getClassName();
  unsigned getSlotCount();
  ArrayRef<Type> getSlotTypes();

  // Shorthand to check whether the SlotObject is of a given className and
  // arity.
  bool isOfClassArity(StringRef className, unsigned arity) {
    return getClassName().getValue() == className && getSlotCount() == arity;
  }
};

/// The type of the Python `str` values.
class StrType : public Type::TypeBase<StrType, Type, TypeStorage> {
public:
  using Base::Base;
  static StrType get(MLIRContext *context) { return Base::get(context); }
};

/// Python 'tuple' type.
class TupleType : public Type::TypeBase<TupleType, Type, TypeStorage> {
public:
  using Base::Base;
  static TupleType get(MLIRContext *context) { return Base::get(context); }
};

/// An unknown type that could be any supported python type.
class UnknownType : public Type::TypeBase<UnknownType, Type, TypeStorage,
                                          CHOPPERTypingTypeMapInterface::Trait> {
public:
  using Base::Base;
  static UnknownType get(MLIRContext *context) { return Base::get(context); }

  Typing::CPA::TypeNode *mapToCPAType(Typing::CPA::Context &context);
};

} // namespace Basicpy
} // namespace CHOPPER
} // namespace mlir

#include "Dialect/Basicpy/IR/BasicpyOpsDialect.h.inc"

#endif // CHOPPER_DIALECT_BASICPY_IR_BASICPY_DIALECT_H
