add_chopper_dialect_library(CHOPPERDialectNumpy
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Numpy

  DEPENDS
  MLIRNumpyOpsIncGen

  LINK_LIBS PUBLIC
  CHOPPERDialectBasicpy
  MLIRIR
)
