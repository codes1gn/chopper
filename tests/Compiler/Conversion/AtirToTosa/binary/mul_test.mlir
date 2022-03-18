// RUN: chopper-opt <%s -split-input-file -convert-atir-to-tosa | FileCheck %s --dump-input=fail

// this test covers the error check that should be reported while parsing the IR

// CHECK-LABEL: func @atir.mul_test_scalar_noret
// basic op with scalar args and none return
func @atir.mul_test_scalar_noret(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  // CHECK-NOT: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = atir.mul %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// CHECK-LABEL: func @atir.mul_test_scalar_ret
// basic op with scalar args and none return
func @atir.mul_test_scalar_ret(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>{
  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = atir.mul %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @atir.mul_test_tensor_ret
func @atir.mul_test_tensor_ret(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @atir.mul_test_unknown_shape_same_rank
// unknown shape tensor with same rank
func @atir.mul_test_unknown_shape_same_rank(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: "tosa.mul"(%arg2, %arg3) {shift = 0 : i32} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = atir.mul %arg2, %arg3 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @atir.mul_test_same_shape_same_rank
// unknown shape tensor with same rank
func @atir.mul_test_same_shape_same_rank(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<16x2xf32>, %arg3: tensor<16x2xf32>) -> tensor<8xf32>{
  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NOT: "tosa.mul"(%arg2, %arg3) {shift = 0 : i32} : (tensor<16x2xf32>, tensor<16x2xf32>) -> tensor<16x2xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %1 = atir.mul %arg2, %arg3 : (tensor<16x2xf32>, tensor<16x2xf32>) -> tensor<16x2xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @atir.mul_test_lhs_broadcast
// the broadcast is from scalar.
func @atir.mul_test_lhs_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @atir.mul_test_rhs_broadcast
// the broadcast is from scalar.
func @atir.mul_test_rhs_broadcast(%arg0: tensor<f32>, %arg1: tensor<13xf32>) -> tensor<13xf32> {
  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<f32>, tensor<13xf32>) -> tensor<13xf32>
  %0 = atir.mul %arg0, %arg1 : (tensor<f32>, tensor<13xf32>) -> tensor<13xf32>
  return %0 : tensor<13xf32>
}


// TODO casting element types
