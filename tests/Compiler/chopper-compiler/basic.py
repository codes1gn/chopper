# RUN: python %s 1 %chopper_runtime_shlib 2>&1 | FileCheck %s --check-prefix=CHECK-ONE
# RUN: python %s 2 %chopper_runtime_shlib 2>&1 | FileCheck %s --check-prefix=CHECK-TWO

# CHECK-ONE: output #0: dense<2.000000e+00> : tensor<1xf32>
# CHECK-TWO: output #0: dense<2.600000e+00> : tensor<1xf32>
#
# CHECK-ONE-NOT: output #0: dense<2.600000e+00> : tensor<1xf32>
# CHECK-ONE-NOT: output #0: dense<2.000000e+00> : tensor<1xf16>
# CHECK-TWO-NOT: output #0: dense<2.000000e+00> : tensor<1xf32>
# CHECK-TWO-NOT: output #0: dense<2.600000e+00> : tensor<1xf16>


import sys
import chopper_compiler

# TODO(albert) make this none hardcode of path
argslist = ['placeholder', 'placeholder', '-invoke', 'basic', '-arg-value=dense<[1.0]> : tensor<1xf32>', '-shared-libs=']

argslist2 = ['placeholder', 'placeholder', '-invoke', 'basic', '-arg-value=dense<[1.3]> : tensor<1xf32>', '-shared-libs=']


if __name__ == '__main__':
  assert(len(sys.argv) > 1)
  if int(sys.argv[1]) == 1:
    _args = argslist
  else:
    _args = argslist2

  _args[1] = sys.argv[0].replace('-compiler', '-compiler-runmlir')
  _args[1] = _args[1].replace('py', 'mlir')

  _args[5] += sys.argv[2]

  # execution
  chopper_compiler.load_and_execute(_args)

