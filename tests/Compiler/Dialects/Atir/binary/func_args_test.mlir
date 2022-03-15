// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail

// this test covers the error check that should be reported while parsing the IR

// CHECK-LABEL: func @atir_func_arg_test
// basic op with scalar args and none return
func @atir_func_arg_test(%self: none, %arg0: tensor<f32>, %arg1: tensor<f32>) {
  // CHECK: atir.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = atir.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// CHECK-LABEL: func @atir_func_arg_test_with_ret
// basic op with scalar args and none return
func @atir_func_arg_test_with_ret(%self: none, %arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>{
  // CHECK: atir.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = atir.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// CHECK-LABEL: func @forward
func @forward(%self: none, %a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %0 = atir.add %arg1, %arg2 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %c = atir.add %a, %b : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %c : tensor<2x3xf32>
}
