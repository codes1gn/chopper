
script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# regression tests for CRT runtime
cd ../CRT
cargo test
cd -

cd ../Raptors
cargo test
cd -


cd ${top_dir_realpath}/build

# add iree backend execution env
iree_build_dir="${top_dir_realpath}/iree_build/"
export PYTHONPATH=${iree_build_dir}bindings/python:${iree_build_dir}compiler-api/python_package:$PYTHONPATH && cmake --build . --target check-chopper

# regression tests for entire repo
cmake --build . --target check-chopper

