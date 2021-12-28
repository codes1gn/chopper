add_chopper_dialect_library(CHOPPERDialectAtir
  AtirDialect.cpp
  AtirOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Atir

  DEPENDS
  MLIRAtirOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
