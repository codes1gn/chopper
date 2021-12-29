add_chopper_dialect_library(CHOPPERDialectRefback
  RefbackDialect.cpp
  RefbackOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Refback

  DEPENDS
  MLIRRefbackOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
