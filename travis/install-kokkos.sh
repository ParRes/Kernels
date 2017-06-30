#!/bin/sh

set -e
set -x

TRAVIS_ROOT="$1"

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

if [ ! -d "$TRAVIS_ROOT/kokkos" ]; then
    git clone --depth 10 https://github.com/kokkos/kokkos.git
    cd kokkos
    mkdir build
    cd build
    # Build for SNB just to be safe...
    ../generate_makefile.bash --prefix=${TRAVIS_ROOT}/kokkos \
                              --compiler=${CXX} ${KOKKOS_BACKEND} \
                              --arch=SNB \
                              --make-j=2
    make
    make install
else
    echo "KOKKOS installed..."
    find $TRAVIS_ROOT/kokkos -name Kokkos_Core.hpp
fi
