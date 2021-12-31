add_chopper_conversion_library(CHOPPERTypingPasses
  Passes.cpp
  CPATypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Typing/Transforms

  DEPENDS
  CHOPPERTypingTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERTypingCPASupport
  CHOPPERTypingCPA
)
