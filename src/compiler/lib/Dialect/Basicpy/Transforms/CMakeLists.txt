add_chopper_conversion_library(CHOPPERBasicpyPasses
  Passes.cpp
  TypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Basicpy/Transforms

  DEPENDS
  CHOPPERBasicpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERTypingCPA
)
