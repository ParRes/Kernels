#!/bin/sh

set -e
set -x

CI_ROOT="$1"
os=`uname`

WEBSITE=https://github.com/01org/tbb/releases/download
VERSION=2018_U1
DIRECTORY=tbb2018_20170919oss

case "$os" in
    Darwin)
        echo "Mac"
        wget --no-check-certificate -q ${WEBSITE}/${VERSION}/${DIRECTORY}_mac.tgz
        tar -xzf ${DIRECTORY}_mac.tgz
        ;;

    Linux)
        echo "Linux"
        wget --no-check-certificate -q ${WEBSITE}/${VERSION}/${DIRECTORY}_lin.tgz
        tar -xzf ${DIRECTORY}_lin.tgz
        ;;
esac
export TBBROOT=${PWD}/${DIRECTORY}
mv ${TBBROOT} ${CI_ROOT}/tbb
find ${CI_ROOT}/tbb -name "libtbb.so"
