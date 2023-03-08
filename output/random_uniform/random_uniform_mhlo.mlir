module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "mhlo.rng_uniform"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}

// iree-opt --iree-mhlo-to-linalg-on-tensors output/random_uniform/random_uniform_mhlo.mlir 
// LCG algri
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
    %0 = linalg.init_tensor [2, 3] : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0, %cst_1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<2x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
      %c0_i32 = arith.constant 0 : i32
      %c1103515245_i32 = arith.constant 1103515245 : i32
      %c12345_i32 = arith.constant 12345 : i32
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.addi %3, %c0_i32 : i32
      %5 = arith.muli %4, %c1103515245_i32 : i32
      %6 = arith.addi %5, %c12345_i32 : i32
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      %9 = arith.addi %8, %6 : i32
      %10 = arith.muli %9, %c1103515245_i32 : i32
      %11 = arith.addi %10, %c12345_i32 : i32
      %cst_2 = arith.constant 2.32830644E-10 : f32
      %12 = arith.subf %arg1, %arg0 : f32
      %13 = arith.mulf %12, %cst_2 : f32
      %14 = arith.uitofp %11 : i32 to f32
      %15 = arith.mulf %14, %13 : f32
      %16 = arith.addf %15, %arg0 : f32
      linalg.yield %16 : f32
    } -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
