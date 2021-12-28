// RUN: not chopper-compiler-runmlir %s \
// RUN:   -invoke invalid_broadcast \
// RUN:   -arg-value="dense<[1.0, 2.0]> : tensor<2xf32>" \
// RUN:   -arg-value="dense<[3.0, 4.0, 5.0]> : tensor<3xf32>" \
// RUN:   -shared-libs=%chopper_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: CHOPPER: aborting: invalid broadcast
func @invalid_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
