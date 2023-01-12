module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
  func @main() -> tensor<3x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<3x4xf32>
    %1 = mhlo.constant dense<[3, 4]> : tensor<2xi64>
    %2 = mhlo.constant dense<2.000000e+00> : tensor<3x4xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = "mhlo.rng_normal"(%3, %4, %1) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x4xf32>
    %6 = mhlo.multiply %5, %2 : tensor<3x4xf32>
    %7 = mhlo.add %6, %0 : tensor<3x4xf32>
    return %7 : tensor<3x4xf32>
  }
}