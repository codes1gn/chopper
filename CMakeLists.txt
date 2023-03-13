cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# set backend option, use refjit
option(BUILD_CHOPPER_REFJIT "Enables the reference JIT backend." ON)
option(CHOPPER_BUILD_CHOPPER_DYLIB "Enables shared build of NPCOMP dylib (depends on LLVM/MLIR dylib support)" ON)

# build project as standalone structure, use mlir and llvm as external lib
project(chopper LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

# config mlir and llvm
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LIT_ARGS_DEFAULT "-sv")
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(CMAKE_PREFIX_PATH "/usr/bin/python")
message(STATUS "PYTHON_INSTALL_PREFIX = ${CMAKE_PREFIX_PATH}")

# set mlir and llvm to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(EnvFuncs)
include(AddChopper)
include(ChopperPython)

set(CHOPPER_BUILT_STANDALONE 1)
set(CHOPPER_TABLEGEN_ARGS "")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include)
include_directories(${PROJECT_BINARY_DIR}/compiler/include)

set(MLIR_HLO_SOURCE ${PROJECT_SOURCE_DIR}/external/mlir-hlo/include)
# set(MLIR_HLO_BUILD ${PROJECT_BINARY_DIR}/iree_build/third_party/llvm-project/llvm/tools/mlir-hlo/include)
# message(STATUS "MLIR_HLO_SOURCE = ${MLIR_HLO_SOURCE}")
# message(STATUS "MLIR_HLO_BUILD = ${MLIR_HLO_BUILD}")


# set(MLIR_HLO_SOURCE ${PROJECT_SOURCE_DIR}/external/iree/third_party/mlir-hlo/include)
# set(MLIR_HLO_BUILD ${PROJECT_BINARY_DIR}/iree_build/third_party/llvm-project/llvm/tools/mlir-hlo/include)
# message(STATUS "MLIR_HLO_SOURCE = ${MLIR_HLO_SOURCE}")
# message(STATUS "MLIR_HLO_BUILD = ${MLIR_HLO_BUILD}")
# include_directories(${MLIR_HLO_SOURCE})
# include_directories(${MLIR_HLO_BUILD})
# include_directories(${PROJECT_BINARY_DIR}/iree_build/third_party/llvm-project/llvm/tools/mlir/include)
# include_directories(${PROJECT_SOURCE_DIR}/external/iree/third_party/llvm-project/llvm/include)
# include_directories(${PROJECT_SOURCE_DIR}/external/iree/third_party/llvm-project/mlir/include)
# include_directories(${PROJECT_BINARY_DIR}/iree_build/third_party/llvm-project/llvm/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# look for python env
set(Python_INCLUDE_DIRS "/home/zp/anaconda3/envs/python36/include/")
set(Python_LIBRARIES "/home/zp/anaconda3/envs/python36/lib/")
# set(CHOPPER_PYTHON_BINDINGS_VERSION_LOCKED 1 CACHE BOOL
#     ¦  "Links to specific python libraries, resolving all symbols.")
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Found python libraries: ${Python_LIBRARIES}")



# add dependency
add_subdirectory(external/pybind11)

# add targets for chopper projects
add_subdirectory(compiler)
add_subdirectory(tests)


