// source code
import tensorflow as tf

@tf.function(jit_compile = True)
def random_normal(shape, mean, stddev):
  return tf.random.normal(shape, mean, stddev)


//==================TF Dialect===================
// tf.mlir.experimental.convert_function(random_normal.get_concrete_function(shape2, 1.0, 2.0))
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {device = "", value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.RandomStandardNormal"(%cst_1) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x3xf32>
    %1 = "tf.Mul"(%0, %cst) {device = ""} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
    %2 = "tf.AddV2"(%1, %cst_0) {device = ""} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}

//===============MHLO Dialect=======================
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<2x3xf32>
    %2 = mhlo.constant dense<2.000000e+00> : tensor<2x3xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = "mhlo.rng_normal"(%3, %4, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    %6 = mhlo.multiply %5, %2 : tensor<2x3xf32>
    %7 = mhlo.add %6, %1 : tensor<2x3xf32>
    return %7 : tensor<2x3xf32>
  }
}

//IREE preprocess mhlo.rng_normal
module  {
  func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x5xf32> {
    %0 = mhlo.constant dense<[1.02256835, 0.553560376, 0.971909821, 1.55706441, 1.61216831, 0.645053744, 1.36710894, 1.4741081]> : tensor<8xf32>
    %1 = mhlo.constant dense<[5.582620e-01, 0.573728263, -0.747536361, 0.937180519, -0.990169346, -0.992096722, 0.514863789, -0.597038865]> : tensor<8xf32>
    %2 = mhlo.constant dense<[-0.829664707, -0.819045722, 0.664220869, 0.348844767, 0.139873877, 0.125475585, -8.572720e-01, -0.802212297]> : tensor<8xf32>
    %3 = "mhlo.broadcast"(%arg1) {broadcast_sizes = dense<8> : tensor<1xi64>} : (tensor<f32>) -> tensor<8xf32>
    %4 = mhlo.multiply %3, %0 : tensor<8xf32>
    %5 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<8> : tensor<1xi64>} : (tensor<f32>) -> tensor<8xf32>
    %6 = mhlo.multiply %4, %1 : tensor<8xf32>
    %7 = mhlo.add %6, %5 : tensor<8xf32>
    %8 = mhlo.multiply %4, %2 : tensor<8xf32>
    %9 = mhlo.add %8, %5 : tensor<8xf32>
    %10 = "mhlo.concatenate"(%7, %9) {dimension = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
    %11 = tensor.extract_slice %10[0] [15] [1] : tensor<16xf32> to tensor<15xf32>
    %12 = "mhlo.reshape"(%11) : (tensor<15xf32>) -> tensor<3x5xf32>
    return %12 : tensor<3x5xf32>
  }
}

//no mean and stddev
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {device = "", value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.RandomStandardNormal"(%cst) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x3xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
//mean and stddev is 2D-tensor
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {device = "", value = dense<[[4.000000e+00, 5.000000e+00, 6.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_1 = "tf.Const"() {device = "", value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.RandomStandardNormal"(%cst_1) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<2x3xf32>
    %1 = "tf.Mul"(%0, %cst) {device = ""} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = "tf.AddV2"(%1, %cst_0) {device = ""} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
//result is scalar
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}} {
  func @__inference_random_normal_86() -> tensor<f32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
    %0 = "tf.RandomStandardNormal"(%cst) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<0xi32>) -> tensor<f32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}


//no mean and stddev
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.rng_normal"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
//mean and stddev is 2D-tensor
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
    %2 = mhlo.constant dense<[[4.000000e+00, 5.000000e+00, 6.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<2x3xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = "mhlo.rng_normal"(%3, %4, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    %6 = mhlo.multiply %5, %2 : tensor<2x3xf32>
    %7 = mhlo.add %6, %1 : tensor<2x3xf32>
    return %7 : tensor<2x3xf32>
  }
}
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @__inference_random_normal_60() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
    %2 = mhlo.constant dense<[[4.000000e+00, 5.000000e+00, 6.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<2x3xf32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = "mhlo.rng_normal"(%3, %4, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    %6 = mhlo.multiply %5, %2 : tensor<2x3xf32>
    %7 = mhlo.add %6, %1 : tensor<2x3xf32>
    return %7 : tensor<2x3xf32>
  }
}
//result is scalar
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<f32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<> : tensor<0xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.rng_normal"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    return %3 : tensor<f32>
  }
}

