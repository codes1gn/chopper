// tf.random.uniform(shape, minval, maxval)

// with default arguments , minval = 0.0 and maxval = 1.0
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {device = "", value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.RandomUniform"(%cst) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x3xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
//  ||
//  ||
// \\//
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.rng_uniform"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}

// with not default args and bias(maxval-minval>1)
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.RandomUniform"(%cst_0) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x3xf32>
    %1 = "tf.Mul"(%0, %cst) {device = ""} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
    %2 = "tf.AddV2"(%1, %cst_1) {device = ""} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
//  ||
//  ||
// \\//
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<2x3xf32>
    %1 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %2 = mhlo.constant dense<2.000000e+00> : tensor<2x3xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = "mhlo.rng_uniform"(%3, %4, %1) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    %6 = mhlo.multiply %5, %2 : tensor<2x3xf32>
    %7 = mhlo.add %6, %0 : tensor<2x3xf32>
    return %7 : tensor<2x3xf32>
  }
}