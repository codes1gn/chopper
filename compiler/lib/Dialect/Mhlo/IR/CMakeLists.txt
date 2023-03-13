add_chopper_dialect_library(CHOPPERDialectMhlo
  MhloDialect.cpp
  MhloOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Mhlo

  DEPENDS
  MLIRMhloOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
