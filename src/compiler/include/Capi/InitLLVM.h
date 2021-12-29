/*===-- chopper-c/InitLLVM.h - C API for initializing LLVM  --------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

//#ifndef CHOPPER_C_INITLLVM_H
//#define CHOPPER_C_INITLLVM_H
#ifndef CAPI_INITLLVM_H
#define CAPI_INITLLVM_H

#ifdef __cplusplus
extern "C" {
#endif

/** Initializes LLVM codegen infrastructure and related MLIR bridge components.
 */
void chopperInitializeLLVMCodegen();

#ifdef __cplusplus
}
#endif

#endif // CAPI_INITLLVM_H
