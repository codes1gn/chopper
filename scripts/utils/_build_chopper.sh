
#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/../.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
mkdir -p ${top_dir_realpath}/build
cd ${top_dir_realpath}/build

cmake -G Ninja .. \
    -DMLIR_DIR=${top_dir_realpath}/build/mlir_build/install_dir/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=${top_dir_realpath}/build/mlir_build/bin/llvm-lit \
    -DCMAKE_C_COMPILER=clang-11 \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld

# build util-tools
cmake --build . --target chopper-opt
cmake --build . --target chopper-translate
cmake --build . --target chopper-compiler-runmlir

# build apis
cmake --build . --target chopper-compiler-runmlir-capi
cmake --build . --target chopper-opt-capi

# build compiler
cmake --build . --target chopper_compiler

# build mlir doc
cmake --build . --target mlir-doc

