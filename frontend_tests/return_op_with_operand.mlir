// RUN: chopper-compiler-runmlir %s \
// RUN:   -invoke basic \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%chopper_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<1xf32>
// func @scalar_add_float(%arg0: f32) -> f32 {
//  %0 =  tcf.add %arg0, %arg0 : (f32, f32) -> f32
//  return %0 : f32
//}
func @constant1() -> f32{
  return %ret1 : f32
}

