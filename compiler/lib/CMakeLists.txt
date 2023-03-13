add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(RefBackend)
add_subdirectory(Typing)
add_subdirectory(Capi)


# TODO fix this init process to opt not to init
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(chopper_dialect_libs GLOBAL PROPERTY CHOPPER_DIALECT_LIBS)
get_property(chopper_conversion_libs GLOBAL PROPERTY CHOPPER_CONVERSION_LIBS)

message(STATUS "CHOPPER Dialect libs: ${chopper_dialect_libs}")
message(STATUS "CHOPPER Conversion libs: ${chopper_conversion_libs}")

add_chopper_library(CHOPPERInit
    ChopperInit.cpp

  LINK_LIBS

  PUBLIC
  # Local depends
  CHOPPERDialectBasicpy
  CHOPPERDialectNumpy
  CHOPPERDialectAtir
  CHOPPERDialectCtir
  CHOPPERDialectRefback
  CHOPPERDialectRefbackrt
  CHOPPERDialectMhlo

  CHOPPERBasicpyPasses
  CHOPPERNumpyPasses
  CHOPPERAtirPasses
  CHOPPERCtirPasses
  CHOPPERMhloPasses

  CHOPPERConversionPasses
  CHOPPERRefBackend

  ${chopper_dialect_libs}
  ${chopper_conversion_libs}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
)


