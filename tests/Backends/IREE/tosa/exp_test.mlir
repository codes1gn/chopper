// RUN: iree-opt -split-input-file %s | FileCheck %s --dump-input=fail

module  {
  func @atir_exp_test_scalar_noret(%arg0: tensor<f32>) -> tensor<f32> {
    // CHECK: "tosa.exp"
    %0 = "tosa.exp"(%arg0) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

