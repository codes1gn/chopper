// RUN: chopper-opt <%s -split-input-file -convert-atir-to-tosa | FileCheck %s --dump-input=fail --check-prefix=PCHECK


//tosa.Tanh not accepting 0D tensor

// CHECK-LABEL: func @tanh_conversion_test_tensor
func @tanh_conversion_test_tensor(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // PCHECK: "tosa.tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = atir.tanh %arg0: (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL: func @tanh_conversion_test_shapedtensor
func @tanh_conversion_test_shapedtensor(%arg0: tensor<17x16xf32>) -> tensor<17x16xf32>{
  // PCHECK: "tosa.tanh"(%arg0) : (tensor<17x16xf32>) -> tensor<17x16xf32>
  %0 = atir.tanh %arg0: (tensor<17x16xf32>) -> tensor<17x16xf32>
  return %0 : tensor<17x16xf32>
}

