// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @matmul_legal_unknown
func @matmul_legal_unknown(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_legal_known
func @matmul_legal_known(%arg0: tensor<27x16xf32>, %arg1: tensor<16x21xf32>) -> tensor<27x21xf32> {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  return %0 : tensor<27x21xf32>
}

// -----
// CHECK-LABEL: test_matmul_2d
func @test_matmul_2d(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

// new 3DTENSOR test -----
// CHECK-LABEL: test_matmul_3d
func @test_matmul_3d(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}
