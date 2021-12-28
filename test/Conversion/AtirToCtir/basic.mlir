// RUN: chopper-opt <%s -convert-atir-to-ctir | FileCheck %s

// NOTE: We are keeping this pass around, even though it currently does
// nothing, in order to avoid having to reintroduce the same
// boilerplate.

// CHECK: @f
func @f() {
  return
}
