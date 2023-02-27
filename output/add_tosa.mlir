module @forward_3a45baf4b2f14b61b589a314a96f1823  {
  func @forward(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = "tosa.identity"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = "tosa.identity"(%arg1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1, %2, %3 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
}