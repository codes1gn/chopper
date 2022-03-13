# RUN: python %s 2>&1 | FileCheck %s --check-prefix=CHECK-CRT

import chopper.crt.rust as crt

print(crt.rust_func())
# CHECK-CRT: 14
