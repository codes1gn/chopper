# chopper_runner is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.

if(CHOPPER_PYTHON_BINDINGS_VERSION_LOCKED)
  set(CHOPPER_PYEXT_LINK_MODE SHARED)
  set(CHOPPER_PYEXT_LIBADD ${Python3_LIBRARIES})
else()
  set(CHOPPER_PYEXT_LINK_MODE MODULE)
  set(CHOPPER_PYEXT_LIBADD)
endif()

message(STATUS "CHOPPER_PYEXT_LINK_MODE = :${CHOPPER_PYEXT_LINK_MODE}")
message(STATUS "CHOPPER_PYEXT_LIBADD = :${CHOPPER_PYEXT_LIBADD}")

# TODO fix this init process to opt not to init
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(chopper_dialect_libs GLOBAL PROPERTY CHOPPER_DIALECT_LIBS)
get_property(chopper_conversion_libs GLOBAL PROPERTY CHOPPER_CONVERSION_LIBS)

message(STATUS "CHOPPER Dialect libs: ${chopper_dialect_libs}")
message(STATUS "CHOPPER Conversion libs: ${chopper_conversion_libs}")



# TODO(albert) use chopper_compiler_module as target name for resolve python 
# slash issue
pybind11_add_module(chopper_compiler_module chopper-compiler.cpp)
# add_library(chopper_compiler_module ${CHOPPER_PYEXT_LINK_MODE} 
#     chopper-compiler.cpp
# )

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(chopper_compiler_module PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

set_target_properties(chopper_compiler_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(chopper_compiler_module PROPERTIES OUTPUT_NAME chopper_compiler_module)
set_target_properties(chopper_compiler_module PROPERTIES PREFIX
                      "${PYTHON_MODULE_PREFIX}")
set_target_properties(chopper_compiler_module PROPERTIES SUFFIX
                      "${PYTHON_MODULE_EXTENSION}")

message(STATUS "CHOPPERSHLIB is ${CHOPPERSHLIB}")
# pybind requires binding code to be compiled with -fvisibility=hidden
# Better code can be generated if the entire project compiles that way, but
# that is not enforced here.
set_target_properties(chopper_compiler_module PROPERTIES CXX_VISIBILITY_PRESET "hidden")

target_link_libraries(chopper_compiler_module
  PRIVATE
    # Transitive dep on the shared library links most things from there.
    # Shared library deps first ensure we get most of what we need from libraries.
    CHOPPERSHLIB

    chopper-compiler-runmlir-capi
    ${CHOPPER_PYEXT_LIBADD}
  )
chopper_python_target_compile_options(chopper_compiler_module)

mlir_check_all_link_libraries(chopper_compiler_module)
