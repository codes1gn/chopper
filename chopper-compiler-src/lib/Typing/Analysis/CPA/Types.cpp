//===- Support.cpp - Support types and utilities for CPA ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Typing/Analysis/CPA/Types.h"

#include "Typing/Analysis/CPA/Interfaces.h"
#include "mlir/IR/Operation.h"

using namespace mlir::CHOPPER::Typing::CPA;

ObjectBase::~ObjectBase() = default;

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

const ConstraintSet &ConstraintSet::getEmptySet() {
  static ConstraintSet s;
  return s;
}

const TypeVarSet &TypeVarSet::getEmptySet() {
  static TypeVarSet s;
  return s;
}

const TypeNodeSet &TypeNodeSet::getEmptySet() {
  static TypeNodeSet s;
  return s;
}

const ValueTypeSet &ValueTypeSet::getEmptySet() {
  static ValueTypeSet s;
  return s;
}

//===----------------------------------------------------------------------===//
// Environment
//===----------------------------------------------------------------------===//

Environment::Environment(Context &context) : context(context) {}

TypeNode *Environment::mapValueToType(Value value) {
  TypeNode *&cpaType = valueTypeMap[value];
  if (cpaType)
    return cpaType;

  cpaType = context.mapIrType(value.getType());
  assert(cpaType && "currently every IR type must map to a CPA type");

  // Do accounting for type vars.
  if (auto *tv = llvm::dyn_cast<TypeVar>(cpaType)) {
    typeVars.insert(tv);
    // TODO: Tie to value.
  }

  return cpaType;
}

//===----------------------------------------------------------------------===//
// TypeNode and descendent methods
//===----------------------------------------------------------------------===//

void TypeNode::collectDependentTypeVars(Context &context,
                                        TypeVarSet &typeVars) {}

mlir::Type TypeNode::constructIrType(Context &context,
                                     const TypeVarMap &mapping,
                                     MLIRContext *mlirContext,
                                     llvm::Optional<Location> loc) {
  (void)mlir::emitOptionalError(loc, "base class cannot construct concrete types");
  return {};
}

void TypeVar::collectDependentTypeVars(Context &context, TypeVarSet &typeVars) {
  typeVars.insert(this);
}

mlir::Type TypeVar::constructIrType(Context &context, const TypeVarMap &mapping,
                                    MLIRContext *mlirContext,
                                    llvm::Optional<Location> loc) {
  auto *resolvedTypeNode = mapping.lookup(this);
  if (!resolvedTypeNode) {
    if (loc) {
      mlir::emitError(*loc)
          << "type variable " << getOrdinal() << " was not assigned a type";
    }
    return {};
  }
  return resolvedTypeNode->constructIrType(context, mapping, mlirContext, loc);
}

mlir::Type IRValueType::constructIrType(Context &context,
                                        const TypeVarMap &mapping,
                                        MLIRContext *mlirContext,
                                        llvm::Optional<Location> loc) {
  return irType;
}

void ObjectValueType::collectDependentTypeVars(Context &context,
                                               TypeVarSet &typeVars) {
  for (auto *fieldType : getFieldTypes()) {
    fieldType->collectDependentTypeVars(context, typeVars);
  }
}

mlir::Type ObjectValueType::constructIrType(Context &context,
                                            const TypeVarMap &mapping,
                                            MLIRContext *mlirContext,
                                            llvm::Optional<Location> loc) {
  llvm::SmallVector<mlir::Type, 4> fieldIrTypes;
  for (TypeNode *fieldType : getFieldTypes()) {
    auto irType =
        fieldType->constructIrType(context, mapping, mlirContext, loc);
    if (!irType)
      return {};
    fieldIrTypes.push_back(irType);
  }
  return irCtor(this, fieldIrTypes, mlirContext, loc);
}

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

Context::Context(IrTypeMapHook irTypeMapHook) : irTypeMapHook(irTypeMapHook) {
  environmentStack.emplace_back(std::make_unique<Environment>(*this));
  currentEnvironment = environmentStack.back().get();
}

TypeNode *Context::mapIrType(::mlir::Type irType) {
  // First, see if the type knows how to map itself.
  assert(irType);
  if (auto mapper = irType.dyn_cast<CHOPPERTypingTypeMapInterface>()) {
    auto *cpaType = mapper.mapToCPAType(*this);
    if (cpaType)
      return cpaType;
  }

  if (irTypeMapHook) {
    auto *cpaType = irTypeMapHook(*this, irType);
    if (cpaType)
      return cpaType;
  }

  // Fallback to an IR type.
  return getIRValueType(irType);
}

void Context::addConstraintToGraph(Constraint *c) {
  fwdNodeToConstraintMap[c->getFrom()].insert(c);
  fwdConstraintToNodeMap[c].insert(c->getTo());
  bakNodeToConstraintMap[c->getTo()].insert(c);
  pendingConstraints.insert(c);
  propagateConstraints();
}

void Context::propagateConstraints() {
  // Process pending constraints until converges.
  while (!pendingConstraints.empty()) {
    // Swap for stable iteration.
    assert(pendingConstraintWorklist.empty());
    pendingConstraintWorklist.swap(pendingConstraints);

    for (auto *constraint : pendingConstraintWorklist) {
      ValueTypeSet &fromContents = typeNodeMembers[constraint->getFrom()];
      ValueTypeSet &toContents = typeNodeMembers[constraint->getTo()];

      bool modified = false;
      for (ValueType *fromItem : fromContents) {
        modified = toContents.insert(fromItem).second || modified;
      }
      // If the 'from' is a ValueType, consider it part of its own set.
      if (auto *fromIdentity =
              llvm::dyn_cast<ValueType>(constraint->getFrom())) {
        modified = toContents.insert(fromIdentity).second;
      }

      // If the 'to' item was modified, propagate any of its constraints.
      if (modified) {
        ConstraintSet &toPropagate =
            fwdNodeToConstraintMap[constraint->getTo()];
        for (Constraint *newConstraint : toPropagate) {
          pendingConstraints.insert(newConstraint);
        }
      }
    }
    pendingConstraintWorklist.clear();
  }
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void Identifier::print(raw_ostream &os, bool brief) {
  os << "'" << value << "'";
}

void TypeNode::print(Context &context, raw_ostream &os, bool brief) {
  os << "<BASE CLASS>";
}

void TypeVar::print(Context &context, raw_ostream &os, bool brief) {
  os << "TypeVar(" << ordinal;
  if (!brief) {
    auto &members = context.getMembers(this);
    if (members.empty()) {
      os << " => EMPTY";
    } else {
      os << " => [\n";
      for (ValueType *member : members) {
        os << "      ";
        member->print(context, os, true);
        os << "\n";
      }
      os << "]";
    }
  }
  os << ")";
}

void CastType::print(Context &context, raw_ostream &os, bool brief) {
  os << "cast(" << *typeIdentifier << ", ";
  typeVar->print(context, os, true);
  os << ")";
}

void ReadType::print(Context &context, raw_ostream &os, bool brief) {
  os << "read(";
  type->print(context, os, true);
  os << ")";
}

void WriteType::print(Context &context, raw_ostream &os, bool brief) {
  os << "write(";
  type->print(context, os, true);
  os << ")";
}

void IRValueType::print(Context &context, raw_ostream &os, bool brief) {
  os << "irtype(" << irType << ")";
}

void ObjectValueType::print(Context &context, raw_ostream &os, bool brief) {
  os << "object(" << *typeIdentifier << ",[";
  bool first = true;
  for (auto it : llvm::zip(getFieldIdentifiers(), getFieldTypes())) {
    if (!first)
      os << ", ";
    else
      first = false;
    os << *std::get<0>(it) << ":";
    auto *ft = std::get<1>(it);
    if (ft)
      ft->print(context, os, true);
    else
      os << "NULL";
  }
  os << "])";
}

void Constraint::print(Context &context, raw_ostream &os, bool brief) {
  from->print(context, os, true);
  os << " <: ";
  to->print(context, os, true);
}

void ConstraintSet::print(Context &context, raw_ostream &os, bool brief) {
  for (auto it : llvm::enumerate(*this)) {
    os << it.index() << ":  ";
    it.value()->print(context, os, brief);
    os << "\n";
  }
}

void TypeVarSet::print(Context &context, raw_ostream &os, bool brief) {
  for (auto it : *this) {
    os << it->getOrdinal() << ":  ";
    it->print(context, os, brief);
    os << "\n";
  }
}
