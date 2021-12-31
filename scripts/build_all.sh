#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

sh ${top_dir_realpath}/scripts/utils/_build_dependencies.sh
sh ${top_dir_realpath}/scripts/utils/_build_iree.sh
sh ${top_dir_realpath}/scripts/utils/_build_chopper.sh

