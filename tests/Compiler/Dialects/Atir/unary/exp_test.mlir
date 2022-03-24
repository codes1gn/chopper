// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @atir_exp_test_scalar_noret
//func @atir_exp_test_scalar_noret(%arg0: tensor<f32>) {
func @atir_exp_test_scalar_noret(%arg0: tensor<f32>, %arg1: none) {
  // CHECK: atir.exp %arg0 : (tensor<f32>) -> tensor<f32>
  %0 = atir.exp %arg0: (tensor<f32>) -> tensor<f32>
  return
}

// CHECK-LABEL: func @atir_exp_test_scalar_ret
func @atir_exp_test_scalar_ret(%arg0: tensor<f32>) -> tensor<f32>{
  // CHECK: atir.exp %arg0 : (tensor<f32>) -> tensor<f32>
  %0 = atir.exp %arg0: (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// CHECK-LABEL: func @atir_exp_test_tensor_noret
func @atir_exp_test_tensor_noret(%arg0: tensor<?xf32>) {
  // CHECK: atir.exp %arg0 : (tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.exp %arg0: (tensor<?xf32>) -> tensor<?xf32>
  return
}

// CHECK-LABEL: func @atir_exp_test_tensor_ret
func @atir_exp_test_tensor_ret(%arg0: tensor<?xf32>) -> tensor<?xf32>{
  // CHECK: atir.exp %arg0 : (tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.exp %arg0: (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL: func @atir_exp_test_shapedtensor_noret
func @atir_exp_test_shapedtensor_noret(%arg0: tensor<31xf32>) {
  // CHECK: atir.exp %arg0 : (tensor<31xf32>) -> tensor<31xf32>
  %0 = atir.exp %arg0: (tensor<31xf32>) -> tensor<31xf32>
  return
}

// CHECK-LABEL: func @atir_exp_test_shapedtensor_ret
func @atir_exp_test_shapedtensor_ret(%arg0: tensor<17x16xf32>) -> tensor<17x16xf32>{
  // CHECK: atir.exp %arg0 : (tensor<17x16xf32>) -> tensor<17x16xf32>
  %0 = atir.exp %arg0: (tensor<17x16xf32>) -> tensor<17x16xf32>
  return %0 : tensor<17x16xf32>
}

