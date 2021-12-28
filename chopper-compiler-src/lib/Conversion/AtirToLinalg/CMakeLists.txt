add_chopper_conversion_library(CHOPPERAtirToLinalg
  AtirToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Conversion/AtirToLinalg

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRMemRef
  CHOPPERDialectAtir
)
