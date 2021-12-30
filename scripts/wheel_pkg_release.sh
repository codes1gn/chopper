
#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
cd ${top_dir_realpath}

# invoke setup.py to build wheel installation pkg
cat install_cache.txt | xargs rm -rf
rm -rf dist
python setup.py bdist_wheel
