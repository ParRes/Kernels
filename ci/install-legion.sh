#!/bin/sh

set -e
set -x

CI_ROOT="$1"

echo "compiler versions:"
$CC --version
$CXX --version

if [ ! -d "$CI_ROOT/legion" ]; then
    cd $CI_ROOT
    git clone -b master --depth 1 https://github.com/StanfordLegion/legion.git
else
    echo "Legion present..."
    find $CI_ROOT/legion
fi
