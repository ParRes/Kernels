#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"
MUSL_CC="$2"

if [ "${MUSL_CC}" = "" ] ; then
    MUSL_CC=${CC}
fi

WEBSITE=https://www.musl-libc.org
VERSION=1.1.16
DIRECTORY=releases

if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
    cd ${TRAVIS_ROOT}
    wget --no-check-certificate -q ${WEBSITE}/${DIRECTORY}/musl-${VERSION}.tar.gz
    tar -xzf musl-${VERSION}.tar.gz
    cd musl-${VERSION}
    ./configure --prefix=${TRAVIS_ROOT}/musl CC=${MUSL_CC} && make -j2 && make install
else
    echo "MUSL does not support Mac"
    exit 99
fi

