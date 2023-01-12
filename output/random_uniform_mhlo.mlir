module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
  func @__inference_random_uniform_88() -> tensor<3x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[3, 4]> : tensor<2xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.rng_uniform"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x4xf32>
    return %3 : tensor<3x4xf32>
  }
}