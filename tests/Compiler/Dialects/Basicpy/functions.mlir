// RUN: chopper-opt -split-input-file -verify-diagnostics %s | chopper-opt -canonicalize | FileCheck --dump-input=fail %s

//===----------------------------------------------------------------------===//
// func_template_call
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @positional
func @positional(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // CHECK: basicpy.func_template_call @foobar(%arg0, %arg1) kw []
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw [] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
// CHECK-LABEL: func @kwValid
func @kwValid(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // CHECK: basicpy.func_template_call @foobar(%arg0, %arg1) kw ["second"]
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["second"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
// CHECK-LABEL: func @posArgPack
func @posArgPack(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // CHECK: basicpy.func_template_call @foobar(%arg0, %arg1) kw ["*"]
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["*"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
// CHECK-LABEL: func @kwArgPack
func @kwArgPack(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // CHECK: basicpy.func_template_call @foobar(%arg0, %arg1) kw ["**"]
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["**"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
func @kwOverflow(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // expected-error @+1 {{expected <= kw arg names vs args}}
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["second", "third", "fourth"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
func @badPosArgPack(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // expected-error @+1 {{positional arg pack must be the first kw arg}}
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["*", "*"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

// -----
func @badKwArgPack(%arg0 : !basicpy.UnknownType, %arg1 : !basicpy.UnknownType) -> !basicpy.UnknownType {
  // expected-error @+1 {{kw arg pack must be the last kw arg}}
  %0 = basicpy.func_template_call @foobar(%arg0, %arg1) kw ["**", "next"] : (!basicpy.UnknownType, !basicpy.UnknownType) -> !basicpy.UnknownType
  return %0 : !basicpy.UnknownType
}

//===----------------------------------------------------------------------===//
// func_template
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL: module @valid_template
module @valid_template {
  // CHECK: basicpy.func_template @__global$pkg.foobar attributes {py_bind = ["#abs"]} {
  basicpy.func_template @__global$pkg.foobar attributes {py_bind = ["#abs"]} {
    // CHECK: func @forInts(%arg0: i32) -> i32
    func @forInts(%arg0 : i32) -> i32 {
      return %arg0 : i32
    }
  }
}

// -----
module @invalid_template {
  basicpy.func_template @__global$pkg.foobar {
    // expected-error @+1 {{illegal operation in func_template}}
    module {}
  }
}
