// RUN: chopper-opt <%s -split-input-file -convert-atir-to-tosa | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @exp_conversion_test_scalar
func @exp_conversion_test_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "tosa.exp"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = atir.exp %arg0: (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @exp_conversion_test_tensor
func @exp_conversion_test_tensor(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "tosa.exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.exp %arg0: (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL: func @exp_conversion_test_shapedtensor
func @exp_conversion_test_shapedtensor(%arg0: tensor<17x16xf32>) -> tensor<17x16xf32>{
  // CHECK: "tosa.exp"(%arg0) : (tensor<17x16xf32>) -> tensor<17x16xf32>
  %0 = atir.exp %arg0: (tensor<17x16xf32>) -> tensor<17x16xf32>
  return %0 : tensor<17x16xf32>
}

