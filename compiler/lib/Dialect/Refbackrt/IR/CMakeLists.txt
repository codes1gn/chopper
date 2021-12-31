add_chopper_dialect_library(CHOPPERDialectRefbackrt
  RefbackrtDialect.cpp
  RefbackrtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Refbackrt

  DEPENDS
  MLIRRefbackrtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
