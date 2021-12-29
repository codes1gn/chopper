// RUN: chopper-opt <%s -pass-pipeline=atir-refback-lowering-pipeline | FileCheck %s --dump-input=fail
// RUN: chopper-opt <%s -pass-pipeline=atir-refback-lowering-pipeline{optimize} | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @rank1
func @rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rank2
func @rank2(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = atir.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHxCK-LABEL: func @rank1and2
func @rank1and2(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = atir.add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
