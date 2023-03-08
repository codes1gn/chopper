module  {
  func @main(%loc: tensor<f32>, %scale: tensor<f32>) -> tensor<3x5xf32> {
    %0 = "tosa.const"(){value = dense<[1.02256835, 0.553560376, 0.971909821, 1.55706441, 1.61216831, 0.645053744, 1.36710894, 1.4741081]> : tensor<8xf32>} : () -> tensor<8xf32>
    %1 = "tosa.const"(){value = dense<[5.582620e-01, 0.573728263, -0.747536361, 0.937180519, -0.990169346, -0.992096722, 0.514863789, -0.597038865]> : tensor<8xf32>} : () -> tensor<8xf32>
    %2 = "tosa.const"(){value = dense<[-0.829664707, -0.819045722, 0.664220869, 0.348844767, 0.139873877, 0.125475585, -8.572720e-01, -0.802212297]> : tensor<8xf32>} : () -> tensor<8xf32>
    %3 = "tosa.mul"(%0, %scale) {shift = 0 : i32} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
    %4 = "tosa.mul"(%3, %1) {shift = 0 : i32} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %5 = "tosa.add"(%4, %loc) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
    %6 = "tosa.mul"(%3, %2) {shift = 0 : i32} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %7 = "tosa.add"(%6, %loc) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
    %8 = "tosa.concat"(%5, %7) {axis = 0 : i64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
    %9 = "tosa.slice"(%8){start = [0], size = [15]}: (tensor<16xf32>) -> tensor<15xf32>
    %10 = "tosa.reshape"(%9){new_shape = [3 : i64, 5 : i64]} : (tensor<15xf32>) -> tensor<3x5xf32>
    return %10 : tensor<3x5xf32>
  }
}