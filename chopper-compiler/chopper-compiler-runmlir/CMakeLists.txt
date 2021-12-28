# chopper_runner is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

# TODO fix this init process to opt not to init
get_property(chopper_dialect_libs GLOBAL PROPERTY CHOPPER_DIALECT_LIBS)
get_property(chopper_conversion_libs GLOBAL PROPERTY CHOPPER_CONVERSION_LIBS)

message(STATUS "CHOPPER Dialect libs: ${chopper_dialect_libs}")
message(STATUS "CHOPPER Conversion libs: ${chopper_conversion_libs}")

# build chopper-runner-backend as the public lib
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_chopper_library(chopper-compiler-runmlir-capi
  chopper-compiler-runmlir.cpp

  LINK_LIBS PUBLIC
  # Shared library deps first ensure we get most of what we need from libraries.
  CHOPPERSHLIB
  MLIR

  CHOPPERCAPI
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRJitRunner
  MLIRParser
  MLIRSupport
  CHOPPERInit
  CHOPPERRefBackendJITHelpers
  ${conversion_libs}
  ${dialect_libs}
)

add_dependencies(chopper-compiler-runmlir-capi
  CHOPPERCompilerRuntimeShlib
  CHOPPERSHLIB
  )

add_chopper_executable(chopper-compiler-runmlir
  chopper-compiler-runmlir.cpp
  )

llvm_update_compile_flags(chopper-compiler-runmlir)
target_link_libraries(chopper-compiler-runmlir PRIVATE
  # Shared library deps first ensure we get most of what we need from libraries.
  CHOPPERSHLIB
  MLIR

  CHOPPERCAPI
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRJitRunner
  MLIRParser
  MLIRSupport
  CHOPPERInit
  CHOPPERRefBackendJITHelpers
  ${conversion_libs}
  ${dialect_libs}
)
add_dependencies(chopper-compiler-runmlir
  CHOPPERCompilerRuntimeShlib
  CHOPPERSHLIB
  )
