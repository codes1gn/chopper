// RUN: chopper-compiler-runmlir %s \
// RUN:   -invoke matmul \
// RUN:   -arg-value="dense<[[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]> : tensor<2x3xf32>" \
// RUN:   -arg-value="dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>" \
// RUN:   -shared-libs=%chopper_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// RUN: chopper-compiler-runmlir %s \
// RUN:   -invoke matmul \
// RUN:   -arg-value="dense<0.0> : tensor<2x3xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<3x2xf32>" \
// RUN:   -shared-libs=%chopper_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZEROS

// Basic correctness check:
// [1 0 1] * [1 2] = [6  8]
// [1 1 1]   [3 4]   [9 12]
//           [5 6]

// CHECK: output #0: dense<[
// CHECK-SAME:   [6.000000e+00, 8.000000e+00], [9.000000e+00, 1.200000e+01]
// CHECK-SAME: ]> : tensor<2x2xf32>

// Check with zeros as well. The result should be identically zeros.
// If any uninitialized data sneaks in (even very small values that would be
// rounding errors for the test case above), it will show up here.
// ZEROS: output #0: dense<0.000000e+00> : tensor<2x2xf32>
func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = atir.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

