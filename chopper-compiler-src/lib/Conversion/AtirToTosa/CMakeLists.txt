add_chopper_conversion_library(CHOPPERAtirToTosa
  AtirToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Conversion/AtirToTosa

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
)
