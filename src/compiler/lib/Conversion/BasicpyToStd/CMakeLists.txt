add_chopper_conversion_library(CHOPPERBasicpyToSTD
  Passes.cpp
  PrimitiveOpsConversion.cpp

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  CHOPPERDialectBasicpy
)
