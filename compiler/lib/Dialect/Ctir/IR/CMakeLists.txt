add_chopper_dialect_library(CHOPPERDialectCtir
  CtirDialect.cpp
  CtirOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Ctir

  DEPENDS
  MLIRCtirOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
