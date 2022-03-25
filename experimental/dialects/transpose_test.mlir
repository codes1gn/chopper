// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail


// -----
// CHECK-LABEL: transpose
func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: atir.transpose %arg0, %0 : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  // CHECK: "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  %1 = atir.transpose %arg0, %0 : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %1 : tensor<3x13x21xf32>
}
