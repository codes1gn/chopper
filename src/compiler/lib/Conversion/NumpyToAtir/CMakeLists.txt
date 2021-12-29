add_chopper_conversion_library(CHOPPERNumpyToAtir
  Passes.cpp

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  CHOPPERDialectBasicpy
  CHOPPERDialectNumpy
)
