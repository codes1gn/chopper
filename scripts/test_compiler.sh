
script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# TODO change into a inspect code, to avoid recompile before test
sh ${top_dir_realpath}/scripts/_install_python_package.sh

cd ${top_dir_realpath}/build

cmake --build . --target check-cancer
