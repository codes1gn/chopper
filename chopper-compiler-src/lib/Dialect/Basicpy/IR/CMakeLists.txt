add_mlir_dialect_library(CHOPPERDialectBasicpy
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy

  DEPENDS
  MLIRBasicpyOpsIncGen

  LINK_LIBS PUBLIC
  CHOPPERTypingCPA
  MLIRIR
)
