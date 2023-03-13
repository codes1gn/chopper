add_chopper_conversion_library(CHOPPERAtirToMhlo
  AtirToMhlo.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Conversion/AtirToMhlo

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
  CHOPPERDialectMhlo
)
