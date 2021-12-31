#!/bin/sh

set -e
set -x

CI_ROOT="$1"
MUSL_CC="$2"
os=`uname`

if [ "${MUSL_CC}" = "" ] ; then
    MUSL_CC=${CC}
fi

WEBSITE=https://www.musl-libc.org
VERSION=1.1.16
DIRECTORY=releases

if [ "$os" = "Linux" ] ; then
    cd ${CI_ROOT}
    wget --no-check-certificate -q ${WEBSITE}/${DIRECTORY}/musl-${VERSION}.tar.gz
    tar -xzf musl-${VERSION}.tar.gz
    cd musl-${VERSION}
    ./configure --prefix=${CI_ROOT}/musl CC=${MUSL_CC} && make -j2 && make install
else
    echo "MUSL does not support Mac"
    exit 99
fi

