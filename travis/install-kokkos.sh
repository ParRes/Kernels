#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

case ${TRAVIS_OS_NAME} in
    osx)
        brew install gnu-sed || brew upgrade gnu-sed || true
        SED="gsed"
        ;;
    linux)
        SED="sed"
        ;;
esac

# --with-openmp:                        Enable OpenMP backend.
# --with-pthread:                       Enable Pthreads backend.
# --with-serial:                        Enable Serial backend.
case "${CXX}" in
    g*)
        # All relevant versions of GCC support OpenMP...
        KOKKOS_BACKEND="--with-openmp"
        ;;
    clang*)
        # Clang support for OpenMP is recent...
        KOKKOS_BACKEND="--with-pthread"
        ;;
esac

case $CXX in
    g++)
        for major in "-9" "-8" "-7" "-6" "-5" "" ; do
          if [ -f "`which ${CXX}${major}`" ]; then
              export PRK_CXX="${CXX}${major}"
              echo "Found C++: $PRK_CXX"
              break
          fi
        done
        if [ "x$PRK_CXX" = "x" ] ; then
            export PRK_CXX="${CXX}"
        fi
        ;;
    clang++)
        for version in "-11" "-10" "-9" "-8" "-7" "-6" "-5" "" ; do
          if [ -f "`which ${CXX}${version}`" ]; then
              export PRK_CXX="${CXX}${version}"
              echo "Found C++: $PRK_CXX"
              break
          fi
        done
        if [ "x$PRK_CXX" = "x" ] ; then
            export PRK_CXX="${CXX}"
        fi
        ;;
esac
${PRK_CXX} -v

if [ ! -d "$TRAVIS_ROOT/kokkos" ]; then
    git clone -b develop --depth 1 https://github.com/kokkos/kokkos.git
    cd kokkos
    mkdir build
    cd build
    ${SED} -i "s/DKokkos_ENABLE_TESTS=ON/DKokkos_ENABLE_TESTS=OFF/g" ../generate_makefile.bash
    ../generate_makefile.bash --prefix=${TRAVIS_ROOT}/kokkos \
                              --compiler=${PRK_CXX} ${KOKKOS_BACKEND}
    make -j2
    make -j2 install
else
    echo "KOKKOS installed..."
    find $TRAVIS_ROOT/kokkos -name Kokkos_Core.hpp
fi
