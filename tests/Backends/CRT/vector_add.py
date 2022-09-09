# RUN: python %s | FileCheck %s --dump-input=fail
import chopper.crt.Runtime as CRT
import numpy as np
DINST = CRT.DeviceInstance()

_forward_callable = CRT.CallableModule("%4 = crt.add.f32! %1, %0 : f32\n", "").forward

data0 = np.array([[1.1, 2.2, 3.3]], dtype=np.float32)
data1 = np.array([[1.5, 2.3, 3.7]], dtype=np.float32)
data4 = _forward_callable(DINST, data0, data1)
print(data4)

# CHECK: [2.6 4.5 7. ]
