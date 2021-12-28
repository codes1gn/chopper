add_chopper_conversion_library(CHOPPERAtirPasses
  Passes.cpp
  ShapeRefinement.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Atir/Transforms

  DEPENDS
  CHOPPERAtirPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERDialectAtir
)
