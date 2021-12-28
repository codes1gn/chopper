add_chopper_conversion_library(CHOPPERAtirToStd
  AtirToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Conversion/AtirToStd

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRStandard
  MLIRLinalg
  CHOPPERDialectAtir
)
