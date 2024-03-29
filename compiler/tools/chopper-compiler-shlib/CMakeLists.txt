# Building libmlir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  message(WARNING "Not building CHOPPER dylib because PIC is disabled")
  return()
endif()

# Building libmlir-cpp.so may not work on MSVC
if (MSVC)
  message(WARNING "Not building CHOPPER dylib because not yet supported on MSVC")
  return()
endif()

if(NOT CHOPPER_BUILD_CHOPPER_DYLIB)
  message(WARNING "Not building CHOPPER dylib (not CHOPPER_BUILD_CHOPPER_DYLIB): Fully static builds not yet supported")
  return()
endif()

get_property(chopper_libs GLOBAL PROPERTY CHOPPER_STATIC_LIBS)
list(REMOVE_DUPLICATES chopper_libs)

# Populate _OBJECTS and _DEPS as necessary per platform.
foreach (lib ${chopper_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # Add transitive deps explcitly since otherwise, there would just be
  # objects.
  list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach()

# Note: Does not use add_chopper_library, which is used for things that go
# *into* the libCHOPPER.so. This is building the shared library, so use
# a higher-level rule.
llvm_add_library(
  CHOPPERSHLIB
  SHARED
  chopper-compiler-shlib.cpp
  ${_OBJECTS}
  LINK_LIBS PUBLIC
  # Public dependencies on the MLIR public API and impl shared libraries.
  MLIRPublicAPI
  MLIR
  ${_DEPS}
)
target_link_libraries(CHOPPERSHLIB PRIVATE ${LLVM_PTHREAD_LIB})
