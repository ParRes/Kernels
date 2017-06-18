#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

WEBSITE=https://github.com/01org/tbb/releases/download
VERSION=2017_U7
DIRECTORY=tbb2017_20170604oss

case "${TRAVIS_OS_NAME}" in
    osx)
        echo "Mac"
        wget --no-check-certificate -q ${WEBSITE}/${VERSION}/${DIRECTORY}_mac.tgz
        tar -xzf ${DIRECTORY}_mac.tgz
        ;;

    linux)
        echo "Linux"
        wget --no-check-certificate -q ${WEBSITE}/${VERSION}/${DIRECTORY}_lin.tgz
        tar -xzf ${DIRECTORY}_lin.tgz
        ;;
esac
export TBBROOT=${PWD}/${DIRECTORY}
ls -l ${TBBROOT}
mv ${TBBROOT} ${TRAVIS_ROOT}/tbb
ls -l ${TRAVIS_ROOT}/tbb
find ${TRAVIS_ROOT}/tbb -name "*.h"
find ${TRAVIS_ROOT}/tbb -name "libtbb*"
