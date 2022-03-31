# RUN: python %s 2>&1 | FileCheck %s --check-prefix=CHECK-CRT

import chopper.crt.Runtime as CRT
import numpy as np

di = CRT.DeviceInstance()
print(CRT.load_and_invoke(di, [1.1, 2.2], [1, 2], [3.3, 4.4], [1, 2], "31415926"))
print(CRT.testing(np.array([[1.1, 3.3], [1.1, 3.3]], dtype=np.float32)))
# CHECK-CRT: 31415926
