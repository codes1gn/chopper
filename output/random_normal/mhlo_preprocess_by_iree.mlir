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