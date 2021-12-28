#!/bin/bash
# Formats all source files.

set +e
td="$(dirname $0)/.."

function find_cc_sources() {
  local dir="$1"
  find "$dir" -name "*.h"
  find "$dir" -name "*.cpp"
}
# C/C++ sources.
set -o xtrace
clang-format -i \
  $(find_cc_sources chopper-compiler) \
  $(find_cc_sources chopper-compiler-src)

# Python sources.
yapf --recursive -i "$td/chopper_frontend"
