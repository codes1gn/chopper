add_chopper_conversion_library(CHOPPERAtirToCtir
  AtirToCtir.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Conversion/AtirToCtir

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  CHOPPERDialectAtir
  CHOPPERDialectCtir
)
