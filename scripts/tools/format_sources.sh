#!/bin/bash
# Formats all source files.

set +e
td="$(dirname $0)/../.."

function find_cc_sources() {
  local dir="$1"
  find "$dir" -name "*.h"
  find "$dir" -name "*.cpp"
}

function find_py_sources() {
  local dir="$1"
  find "$dir" -name "*.py"
}

# C/C++ sources.
set -o xtrace
clang-format -i \
  $(find_cc_sources src)

# Python sources.
yapf --recursive -i "$td/chopper"
