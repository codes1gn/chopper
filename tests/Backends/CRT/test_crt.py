# RUN: python %s 2>&1 | FileCheck %s --check-prefix=CHECK-CRT

import chopper.crt.Runtime as CRT

di = CRT.DeviceInstance()
print(CRT.load_and_invoke(di, [1.1, 2.2], [1, 2], [3.3, 4.4], [1, 2], "31415926"))
# CHECK-CRT: 31415926
