// RUN: chopper-opt <%s -split-input-file -convert-atir-to-tosa | FileCheck %s --dump-input=fail


// CHECK-LABEL: conv2d
func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32> {
  // CHECK: "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
  %0 = atir.conv_2d_cfirst %arg0, %arg1 {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}
