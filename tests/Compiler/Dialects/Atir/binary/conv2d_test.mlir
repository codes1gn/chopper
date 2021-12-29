// RUN: chopper-opt <%s | FileCheck %s


// CHECK-LABEL: func @conv_2d_legal_unknown
func @conv_2d_legal_unknown(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = atir.conv_2d_cfirst %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

