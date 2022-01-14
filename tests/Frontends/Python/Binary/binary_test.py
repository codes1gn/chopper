# RUN: python %s 2>&1 | FileCheck %s -dump-input=fail

""" Tests pyMLIR on examples that use the Toy dialect. """
import os
import numpy as np
from chopper.python import PythonRunner
from typing import Callable, List

Vector = List[float]


def analyse(the_func: Callable) -> None:

    # TODO wrapper this logics into functions
    pyast = PythonRunner.parse_python(the_func)
    print(PythonRunner.dump_python(pyast))
    mlast = PythonRunner.convert_python_to_mlir(pyast)
    print(PythonRunner.dump_mlir(mlast))


if __name__ == "__main__":

    # TODO: Add OP
    cunt = 0
    def add_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 + var0
        return res

    def add_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 + arg1
        return res

    def add_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 + var1
        return res

    def listAdd0(arg0: list, arg1: list) -> list:

        res = np.array(arg0) + np.array(arg1)
        return res

    def listAdd1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) + np.array(arg1)
        return res

    # analyse(add_scalar0)
    # cunt +=1
    # analyse(add_scalar1)
    # cunt +=1
    # analyse(add_scalar2)
    # cunt +=1
    # analyse(listAdd0)
    # cunt +=1
    # analyse(listAdd1)
    # cunt +=1
    # print(f'Add pass: {cunt}')

    # TODO: Sub OP
    cunt_sub = 0
    def sub_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 - var0
        return res

    def sub_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 - arg1
        return res

    def sub_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 - var1
        return res

    def listSub0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) - np.array(arg1)
        return res

    def listSub1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) - np.array(arg1)
        return res

    # analyse(sub_scalar0)
    # cunt_sub +=1
    # analyse(sub_scalar1)
    # cunt_sub +=1
    # analyse(sub_scalar2)
    # cunt_sub +=1
    # analyse(listSub0)
    # cunt_sub +=1
    # analyse(listSub1)
    # cunt_sub +=1
    # print(f'cunt_sub={cunt_sub}')

    # TODO: Mul OP
    cunt_mul= 0
    def mul_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 * var0
        return res

    def mul_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 * arg1
        return res

    def mul_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 * var1
        return res

    def listMul0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) * np.array(arg1)
        return res

    def listMul1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) * np.array(arg1)
        # res = arg0 * arg1
        return res
    # analyse(mul_scalar0)
    # cunt_mul +=1
    # analyse(mul_scalar1)
    # cunt_mul +=1
    # analyse(mul_scalar2)
    # cunt_mul +=1
    # analyse(listMul0)
    # cunt_mul +=1
    # analyse(listMul1)
    # cunt_mul +=1
    # print(f'cunt_mul={cunt_mul}')


    # TODO: Mat OP

    def listMat0(arg0: list, arg1: list) -> list:
        res = np.mat(arg0) * np.mat(arg1)
        return res

    def listMat1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.mat(arg0) * np.mat(arg1)
        return res

    # analyse(listMat0)
    # analyse(listMat1)

    # TODO: Div
    def div_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 / var0
        return res
    
    def div_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 / arg1
        return res

    def div_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 / var1
        return res
    
    def listDiv0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) / np.array(arg1)
        return res
    
    def listDiv1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) / np.array(arg1)
        return res   
    
    # analyse(div_scalar0)
    # analyse(div_scalar1)
    # analyse(div_scalar2)
    # analyse(listDiv0)
    # analyse(listDiv1)
    
    # TODO: Mod
    def mod_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 / var0
        return res
    
    def mod_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 / arg1
        return res

    def mod_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 / var1
        return res
    
    def listMod0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) / np.array(arg1)
        return res
    
    def listMod1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) / np.array(arg1)
        return res 

    # analyse(mod_scalar0)
    # analyse(mod_scalar1)
    # analyse(mod_scalar2)
    # analyse(listMod0)
    # analyse(listMod1)
    
    # TODO: Pow
    def pow_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 ** var0
        return res
    
    def pow_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 ** arg1
        return res

    def pow_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 ** var1
        return res
    
    def pow_scalar3(arg0:float, arg1:float) -> float:
        arg0 **= arg1
        return arg0
    
    def listPow0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) ** np.array(arg1)
        return res
    
    def listPow1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) ** np.array(arg1)
        return res 

    # analyse(pow_scalar0)
    # analyse(pow_scalar1)
    # analyse(pow_scalar2)
    # analyse(pow_scalar3)
    # analyse(listPow0)
    # analyse(listPow1)
    
    
    # TODO: LShift
    def lshift_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 << var0
        return res
    
    def lshift_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 << arg1
        return res

    def lshift_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 << var1
        return res
    
    def lshift_scalar3(arg0:float, arg1:float) -> float:
        arg0 <<= arg1
        return arg0
    
    def listLshift0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) << np.array(arg1)
        return res
    
    def listLshift1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) << np.array(arg1)
        return res
    
    # analyse(lshift_scalar0)
    # analyse(lshift_scalar1)
    # analyse(lshift_scalar2)
    # analyse(lshift_scalar3)
    # analyse(listLshift0)
    # analyse(listLshift1)
    
    # TODO: RShift
    def rshift_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 >> var0
        return res
    
    def rshift_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 >> arg1
        return res

    def rshift_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 >> var1
        return res

    def rshift_scalar3(arg0:float, arg1:float) -> float:
        arg0 >>= arg1
        return arg0
    
    def listRshift0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) >> np.array(arg1)
        return res
    
    def listRshift1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) >> np.array(arg1)
        return res
    
    # analyse(rshift_scalar0)
    # analyse(rshift_scalar1)
    # analyse(rshift_scalar2)
    # analyse(rshift_scalar3)
    # analyse(listRshift0)
    # analyse(listRshift1)
    
    # TODO: bitor
    def bitor_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 | var0
        return res
    
    def bitor_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 | arg1
        return res

    def bitor_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 | var1
        return res

    def bitor_scalar3(arg0:float, arg1:float) -> float:
        arg0 |= arg1
        return arg0
    
    def listbitor0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) | np.array(arg1)
        return res
    
    def listbitor1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) | np.array(arg1)
        return res
    
    # analyse(bitor_scalar0)
    # analyse(bitor_scalar1)
    # analyse(bitor_scalar2)
    # analyse(bitor_scalar3)
    # analyse(listbitor0)
    # analyse(listbitor1)
    
    # TODO: bitxor
    def bitxor_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 ^ var0
        return res
    
    def bitxor_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 ^ arg1
        return res

    def bitxor_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 ^ var1
        return res

    def bitxor_scalar3(arg0:float, arg1:float) -> float:
        arg0 ^= arg1
        return arg0
    
    def listbitxor0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) ^ np.array(arg1)
        return res
    
    def listbitxor1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) ^ np.array(arg1)
        return res
    
    # analyse(bitxor_scalar0)
    # analyse(bitxor_scalar1)
    # analyse(bitxor_scalar2)
    # analyse(bitxor_scalar3)
    # analyse(listbitxor0)
    # analyse(listbitxor1)
    
    # TODO: bitAnd
    def bitand_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 & var0
        return res
    
    def bitand_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 & arg1
        return res

    def bitand_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 & var1
        return res

    def bitand_scalar3(arg0:float, arg1:float) -> float:
        arg0 &= arg1
        return arg0
    
    def listbitand0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) & np.array(arg1)
        return res
    
    def listbitand1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) & np.array(arg1)
        return res
    
    # analyse(bitand_scalar0)
    # analyse(bitand_scalar1)
    # analyse(bitand_scalar2)
    # analyse(bitand_scalar3)
    # analyse(listbitand0)
    # analyse(listbitand1)
    
    
    # TODO: floordiv
    def floordiv_scalar0(arg0:float) -> float:
        var0 = 2.0
        res = arg0 // var0
        return res
    
    def floordiv_scalar1(arg0:float, arg1:float) -> float:
        res = arg0 // arg1
        return res

    def floordiv_scalar2() -> float:
        var0 = 1.0
        var1 = 2.0
        res = var0 // var1
        return res

    def floordiv_scalar3(arg0:float, arg1:float) -> float:
        arg0 //= arg1
        return arg0
    
    def listfloordiv0(arg0: list, arg1: list) -> list:
        res = np.array(arg0) // np.array(arg1)
        return res

    def listfloordiv1(arg0: List[float], arg1: List[float]) -> List[float]:
        res = np.array(arg0) // np.array(arg1)
        return res

    #analyse(floordiv_scalar0)
    # analyse(floordiv_scalar1)
    #analyse(floordiv_scalar2)
    #analyse(floordiv_scalar3)
    #analyse(listfloordiv0)

    # TODO(should not pass, fix customoperation by atirops then switch on this)
    analyse(listfloordiv1)
    # CHECK: func @listfloordiv1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    # CHECK-NEXT: atir.floordiv
    # CHECK-NEXT: return %res : tensor<?xf32>
