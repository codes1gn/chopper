// Before iree preprocess
// module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1087 : i32}}  {
//   func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
//     %0 = mhlo.constant dense<[3, 4]> : tensor<2xi64>
//     %3 = "mhlo.rng_normal"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x4xf32>
//     return %3 : tensor<3x4xf32>
//   }
// }

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

// iree-opt --iree-mhlo-to-linalg-on-tensors output/random_normal/mhlo_preprocess_by_iree.mlir 
// #map0 = affine_map<(d0) -> ()>
// #map1 = affine_map<(d0) -> (d0)>
// module  {
//   func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<3x5xf32> {
//     %cst = arith.constant dense<[1.02256835, 0.553560376, 0.971909821, 1.55706441, 1.61216831, 0.645053744, 1.36710894, 1.4741081]> : tensor<8xf32>
//     %cst_0 = arith.constant dense<[5.582620e-01, 0.573728263, -0.747536361, 0.937180519, -0.990169346, -0.992096722, 0.514863789, -0.597038865]> : tensor<8xf32>
//     %cst_1 = arith.constant dense<[-0.829664707, -0.819045722, 0.664220869, 0.348844767, 0.139873877, 0.125475585, -8.572720e-01, -0.802212297]> : tensor<8xf32>
//     %0 = linalg.init_tensor [8] : tensor<8xf32>
//     %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
//       linalg.yield %arg2 : f32
//     } -> tensor<8xf32>
//     %2 = linalg.init_tensor [8] : tensor<8xf32>
//     %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%1, %cst : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//       %22 = arith.mulf %arg2, %arg3 : f32
//       linalg.yield %22 : f32
//     } -> tensor<8xf32>
//     %4 = linalg.init_tensor [8] : tensor<8xf32>
//     %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<f32>) outs(%4 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
//       linalg.yield %arg2 : f32
//     } -> tensor<8xf32>
//     %6 = linalg.init_tensor [8] : tensor<8xf32>
//     %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%3, %cst_0 : tensor<8xf32>, tensor<8xf32>) outs(%6 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//       %22 = arith.mulf %arg2, %arg3 : f32
//       linalg.yield %22 : f32
//     } -> tensor<8xf32>
//     %8 = linalg.init_tensor [8] : tensor<8xf32>
//     %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%7, %5 : tensor<8xf32>, tensor<8xf32>) outs(%8 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//       %22 = arith.addf %arg2, %arg3 : f32
//       linalg.yield %22 : f32
//     } -> tensor<8xf32>
//     %10 = linalg.init_tensor [8] : tensor<8xf32>
//     %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%3, %cst_1 : tensor<8xf32>, tensor<8xf32>) outs(%10 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//       %22 = arith.mulf %arg2, %arg3 : f32
//       linalg.yield %22 : f32
//     } -> tensor<8xf32>
//     %12 = linalg.init_tensor [8] : tensor<8xf32>
//     %13 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%11, %5 : tensor<8xf32>, tensor<8xf32>) outs(%12 : tensor<8xf32>) {
//     ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//       %22 = arith.addf %arg2, %arg3 : f32
//       linalg.yield %22 : f32
//     } -> tensor<8xf32>
//     %c0 = arith.constant 0 : index
//     %c0_2 = arith.constant 0 : index
//     %c8 = arith.constant 8 : index
//     %c1 = arith.constant 1 : index
//     %c0_3 = arith.constant 0 : index
//     %c0_4 = arith.constant 0 : index
//     %c8_5 = arith.constant 8 : index
//     %c8_6 = arith.constant 8 : index
//     %c0_7 = arith.constant 0 : index
//     %c8_8 = arith.constant 8 : index
//     %c16 = arith.constant 16 : index
//     %14 = linalg.init_tensor [16] : tensor<16xf32>
//     %cst_9 = arith.constant 0.000000e+00 : f32
//     %15 = linalg.fill(%cst_9, %14) : f32, tensor<16xf32> -> tensor<16xf32> 
//     %c0_10 = arith.constant 0 : index
//     %c0_11 = arith.constant 0 : index
//     %c8_12 = arith.constant 8 : index
//     %16 = tensor.insert_slice %9 into %15[0] [8] [1] : tensor<8xf32> into tensor<16xf32>
//     %17 = arith.addi %c0_10, %c8_12 : index
//     %c0_13 = arith.constant 0 : index
//     %c8_14 = arith.constant 8 : index
//     %18 = tensor.insert_slice %13 into %16[%17] [8] [1] : tensor<8xf32> into tensor<16xf32>
//     %19 = arith.addi %17, %c8_14 : index
//     %20 = tensor.extract_slice %18[0] [15] [1] : tensor<16xf32> to tensor<15xf32>
//     %21 = tensor.expand_shape %20 [[0, 1]] : tensor<15xf32> into tensor<3x5xf32>
//     return %21 : tensor<3x5xf32>
//   }
// }