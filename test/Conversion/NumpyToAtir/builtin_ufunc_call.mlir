// RUN: chopper-opt <%s -convert-numpy-to-atir | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @unknownBuiltinUfunc
func @unknownBuiltinUfunc(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: numpy.builtin_ufunc_call
  // CHECK-NOT: atir.add
  %0 = numpy.builtin_ufunc_call<"NON_EXISTING"> (%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @illagalTernary
func @illagalTernary(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: numpy.builtin_ufunc_call
  // CHECK-NOT: atir.add
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1, %arg0) : (tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @numpyAdd
func @numpyAdd(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK: atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
