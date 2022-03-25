// RUN: chopper-opt <%s | FileCheck %s --dump-input=fail


// -----
// CHECK-LABEL: const
func @test_const(%arg0 : tensor<f32>) -> tensor<f32> {
    %0 = "tosa.const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    return %0 : tensor<f32>
}
