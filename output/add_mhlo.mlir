module {
  func @main(%arg0: tensor<7x5xf32>, %arg1: tensor<7x5xf32>) -> tensor<7x5xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input0,input1", outputs = "Add"}} {
    %0 = mhlo.add %arg0, %arg1 : tensor<7x5xf32>
    return %0 : tensor<7x5xf32>
  }
}