add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToAtir)
add_subdirectory(AtirToLinalg)
add_subdirectory(AtirToStd)
add_subdirectory(AtirToCtir)
add_subdirectory(AtirToTosa)
add_subdirectory(AtirToMhlo)

add_chopper_library(CHOPPERConversionPasses
  Passes.cpp

  DEPENDS
  CHOPPERConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CHOPPERBasicpyToSTD
  CHOPPERNumpyToAtir
  CHOPPERAtirToCtir
  CHOPPERAtirToTosa
  CHOPPERAtirToMhlo
)
