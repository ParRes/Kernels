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
        USE_OPENMP="On"
        ;;
    clang++)
        for version in "-5" "-4" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
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
        USE_OPENMP="Off"
        ;;
esac
${PRK_CXX} -v

if [ ! -d "$TRAVIS_ROOT/raja" ]; then
    BRANCH=develop
    git clone --recursive --depth 1 -b ${BRANCH} https://github.com/LLNL/RAJA.git
    cd RAJA
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=${PRK_CXX} -DCMAKE_C_COMPILER=${PRK_CC} \
             -DCMAKE_INSTALL_PREFIX=${TRAVIS_ROOT}/raja \
             -DENABLE_TBB=On -DENABLE_OPENMP=${USE_OPENMP}
    make -j2
    make install -j2
else
    echo "RAJA installed..."
    find $TRAVIS_ROOT/raja -name RAJA.hxx
fi
