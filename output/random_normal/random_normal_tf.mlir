module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}} {
  func @main() -> tensor<3x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = "tf.Const"() {value = dense<[3, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.RandomStandardNormal"(%cst) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> tensor<3x4xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
}
