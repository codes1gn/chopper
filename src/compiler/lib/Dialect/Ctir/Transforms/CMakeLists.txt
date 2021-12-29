add_chopper_conversion_library(CHOPPERCtirPasses
  Passes.cpp
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Ctir/Transforms

  DEPENDS
  CHOPPERCtirPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERDialectCtir
)
