#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

case $CXX in
    g++)
        for major in "-9" "-8" "-7" "-6" "-5" "" ; do
          if [ -f "`which ${CXX}${major}`" ]; then
              export PRK_CXX="${CXX}${major}"
              export PRK_CC="${CC}${major}"
              echo "Found C++: $PRK_CXX"
              break
          fi
        done
        if [ "x$PRK_CXX" = "x" ] ; then
            export PRK_CXX="${CXX}"
            export PRK_CC="${CC}"
        fi
        ;;
    clang++)
        for version in "-7" "-6" "-5" "-4" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
          if [ -f "`which ${CXX}${version}`" ]; then
              export PRK_CXX="${CXX}${version}"
              export PRK_CC="${CC}${version}"
              echo "Found C++: $PRK_CXX"
              break
          fi
        done
        if [ "x$PRK_CXX" = "x" ] ; then
            export PRK_CXX="${CXX}"
            export PRK_CC="${CC}"
        fi
        ;;
esac
${PRK_CXX} -v

if [ ! -d "$TRAVIS_ROOT/occa" ]; then
    BRANCH="1.0"
    git clone --recursive --depth 1 -b ${BRANCH} https://github.com/libocca/occa.git $TRAVIS_ROOT/occa
    CXX=${PRK_CXX} OCCA_CUDA_ENABLED=0 OCCA_FORTRAN_ENABLED=0 make -C $TRAVIS_ROOT/occa
else
    echo "OCCA installed..."
    find $TRAVIS_ROOT/occa -name occa.hpp
fi
