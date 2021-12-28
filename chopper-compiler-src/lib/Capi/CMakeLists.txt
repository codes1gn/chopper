set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_chopper_library(CHOPPERCAPI
  InitLLVM.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIRExport
  CHOPPERInit
  CHOPPERDialectBasicpy
  CHOPPERDialectNumpy
  )
