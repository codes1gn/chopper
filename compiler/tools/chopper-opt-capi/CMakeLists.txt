get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        CHOPPERInit
        CHOPPERSHLIB
        MLIR
        CHOPPERCAPI
        MLIRAnalysis
        MLIREDSC
        MLIRIR
        MLIRParser
        MLIRSupport
        )

# add_chopper_executable(chopper-opt-capi chopper-opt-capi.cpp)
add_chopper_library(chopper-opt-capi
    chopper-opt-capi.cpp

    LINK_LIBS PUBLIC
    ${LIBS}
    )

add_dependencies(chopper-opt-capi
  CHOPPERCompilerRuntimeShlib
  CHOPPERSHLIB
  )

# target_link_libraries(chopper-opt-capi PRIVATE ${LIBS})

mlir_check_all_link_libraries(chopper-opt-capi)
