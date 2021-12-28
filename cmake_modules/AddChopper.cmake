# Declare an chopper library which can be compiled in libCHOPPER.so.
# This is adapted from add_mlir_library.
function(add_chopper_library name)
  cmake_parse_arguments(ARG
    "SHARED;EXCLUDE_FROM_LIBCHOPPER"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
  ${ARGN})
  set(srcs)
  # TODO: Port the source description logic for IDEs from add_mlir_library.


  if(ARG_SHARED)
    # Rule explicitly requested a shared library.
    set(LIBTYPE SHARED)
  else()
    if(NOT ARG_EXCLUDE_FROM_LIBCHOPPER)
      set_property(GLOBAL APPEND PROPERTY CHOPPER_STATIC_LIBS ${name})
    endif()
  endif()

  # TODO: Enable chopper header export.
  # list(APPEND ARG_DEPENDS chopper-generic-headers)
  llvm_add_library(
    ${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs}
    OBJECT
    DEPENDS ${ARG_DEPENDS}
    LINK_COMPONENTS ${ARG_LINK_COMPONENTS}
    LINK_LIBS ${ARG_LINK_LIBS})

  set_target_properties(${name} PROPERTIES FOLDER "CHOPPER libraries")

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    RUNTIME DESTINATION bin COMPONENT ${name})

endfunction()

# Declare the library associated with a dialect.
function(add_chopper_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CHOPPER_DIALECT_LIBS ${name})
  # TODO: Add DEPENDS chopper-headers
  add_chopper_library(${ARGV})
endfunction()

# Declare the library associated with a conversion.
function(add_chopper_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY CHOPPER_CONVERSION_LIBS ${name})
  # TODO: Add DEPENDS chopper-headers
  add_chopper_library(${ARGV})
endfunction()

function(add_chopper_executable name)
  add_executable(${ARGV})
  llvm_update_compile_flags(${name})
  add_link_opts( ${name} )
  set_output_directory(${name}
    BINARY_DIR ${PROJECT_BINARY_DIR}/bin
    LIBRARY_DIR ${PROJECT_BINARY_DIR}/lib)
  if (LLVM_PTHREAD_LIB)
    # libpthreads overrides some standard library symbols, so main
    # executable must be linked with it in order to provide consistent
    # API for all shared libaries loaded by this executable.
    target_link_libraries(${name} PRIVATE ${LLVM_PTHREAD_LIB})
  endif()

  install(TARGETS ${name}
    RUNTIME DESTINATION ${LLVM_UTILS_INSTALL_DIR}
    COMPONENT ${name})

endfunction()

function(chopper_enable_exceptions name)
  target_compile_options(${name} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -fexceptions
  >
  $<$<CXX_COMPILER_ID:MSVC>:
  /EHsc>
  )
endfunction()
