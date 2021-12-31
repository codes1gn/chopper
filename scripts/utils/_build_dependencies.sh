#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/../.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
mkdir ${top_dir_realpath}/mlir_build
cd ${top_dir_realpath}/mlir_build
mkdir -p ${top_dir_realpath}/mlir_build/install_dir

# try to fix linker issue
# export LDFLAGS=-fuse-ld=$(which ld.lld)

# config mlir
# build with clang
cmake -G Ninja \
    ${top_dir_realpath}/external/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang-11 \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_INSTALL_PREFIX=${top_dir_realpath}/mlir_build/install_dir \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON
    # -DLLVM_USE_LINKER=lld
    # -DLLVM_ENABLE_LLD=ON \
# build with gcc
# cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_INSTALL_PREFIX=./install_dir

# build mlir
cmake --build . --target check-mlir
cmake --build . --target install

cd -
