// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @atir_const_test_scalar_noret
func @atir_const_test_scalar_noret(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: constant dense<1.000000e+00> : tensor<f32>
  %cst = constant dense<1.000000e+00> : tensor<f32>
  return %cst : tensor<f32>
}
