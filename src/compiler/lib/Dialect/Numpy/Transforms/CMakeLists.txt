add_chopper_conversion_library(CHOPPERNumpyPasses
  ArrayToTensor.cpp
  Passes.cpp
  PublicFunctionToTensor.cpp
  RefinePublicReturn.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Numpy/Transforms

  DEPENDS
  CHOPPERNumpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERDialectNumpy
)
