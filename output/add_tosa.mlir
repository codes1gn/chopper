module @forward_57e61f4343a64aed8f54e69d11d27266  {
  func @forward(%arg0: tensor<7x5xf32>, %arg1: tensor<7x5xf32>) -> tensor<7x5xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<7x5xf32>, tensor<7x5xf32>) -> tensor<7x5xf32>
    return %0 : tensor<7x5xf32>
  }
}