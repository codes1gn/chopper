add_chopper_conversion_library(CHOPPERMhloPasses
  Passes.cpp
  MhloCustomPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chopper/Dialect/Mhlo/Transforms

  DEPENDS
  CHOPPERMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CHOPPERDialectMhlo
)
