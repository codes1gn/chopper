#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/../.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
mhlo_source=${top_dir_realpath}/external/mlir-hlo
mkdir -p ${top_dir_realpath}/build
mkdir -p ${top_dir_realpath}/build/mlir_hlo_build

cd ${top_dir_realpath}/build/mlir_hlo_build
mhlo_build_dir="${top_dir_realpath}/build/mlir_hlo_build/"

mlir_dir=${top_dir_realpath}/build/mlir_build/lib/cmake/mlir
echo "mlir-dir = "$mlir_dir

cd $mhlo_source
cmake -GNinja -B $mhlo_build_dir \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=On \
        -DMLIR_DIR=$mlir_dir \
        # -DLLVM_USE_LINKER=lld-11
        # -DLLVM_ENABLE_LLD=ON 
        
        # -DCMAKE_C_COMPILER=clang-11 \
        # -DCMAKE_CXX_COMPILER=clang++-11 

cd $mhlo_build_dir
ninja check-mlir-hlo