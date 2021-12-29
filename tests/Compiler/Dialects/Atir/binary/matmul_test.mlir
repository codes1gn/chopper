// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @matmul_legal_unknown
func @matmul_legal_unknown(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_legal_known
func @matmul_legal_known(%arg0: tensor<27x16xf32>, %arg1: tensor<16x21xf32>) -> tensor<27x21xf32> {
  // CHECK: atir.matmul %arg0, %arg1 : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  %0 = atir.matmul %arg0, %arg1 : (tensor<27x16xf32>, tensor<16x21xf32>) -> tensor<27x21xf32>
  return %0 : tensor<27x21xf32>
}

