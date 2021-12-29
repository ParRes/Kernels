#!/bin/sh

set -e
set -x

CI_ROOT="$1"

git clone --depth 1 https://github.com/triSYCL/triSYCL.git $CI_ROOT/triSYCL
