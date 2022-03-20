// RUN: chopper-opt <%s -split-input-file -convert-atir-to-tosa | FileCheck %s --dump-input=fail

// this test covers the error check that should be reported while parsing the IR

// CHECK-LABEL: func @matmul_legal_known
func @matmul_legal_known(%arg0: tensor<27x16xf32>, %arg1: tensor<16x21xf32>) -> tensor<27x21xf32> {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<1x27x16xf32>, tensor<1x16x21xf32>) -> tensor<1x27x21xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  return %0 : tensor<27x21xf32>
}


// new 3DTENSOR test -----
// CHECK-LABEL: test_matmul
func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}
// TODO casting element types
