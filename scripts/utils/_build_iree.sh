#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/../.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
iree_source=${top_dir_realpath}/external/iree
mkdir -p ${top_dir_realpath}/build
mkdir -p ${top_dir_realpath}/build/iree_build
cd ${top_dir_realpath}/build/iree_build
mkdir -p ${top_dir_realpath}/build/iree_build/iree_install
iree_build_dir="${top_dir_realpath}/build/iree_build/"
iree_install_dir="${top_dir_realpath}/build/iree_build/iree_install/"

# preliminaries
pip3 install absl-py

cd $iree_source
cmake -GNinja -B $iree_build_dir -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-11 \
        -DCMAKE_CXX_COMPILER=clang++-11 \
        -DCMAKE_INSTALL_PREFIX=$iree_install_dir \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DIREE_BUILD_PYTHON_BINDINGS=ON \
        -DIREE_ENABLE_LLD=ON

cmake --build $iree_build_dir --target install

