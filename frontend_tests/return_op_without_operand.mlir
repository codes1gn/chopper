// RUN: chopper-compiler-runmlir %s \
// RUN:   -invoke basic \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%chopper_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<1xf32>
func @constant1(){
  return
}

