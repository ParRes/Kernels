#!/bin/sh

set -e
set -x

CI_ROOT="$1"
os=`uname`

# Kokkos_ENABLE_OPENMP:  Enable OpenMP backend.
# Kokkos_ENABLE_THREADS: Enable C++ threads backend.
case "${CXX}" in
    g*)
        # All relevant versions of GCC support OpenMP...
        KOKKOS_BACKEND="-DKokkos_ENABLE_OPENMP=ON"
        ;;
    clang*)
        # Clang support for OpenMP is recent...
        KOKKOS_BACKEND="-DKokkos_ENABLE_THREADS=ON"
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

if [ ! -d "$CI_ROOT/kokkos" ]; then
    # Kokkos dropped the generate_makefile.bash bootstrap script; current
    # Kokkos is configured directly via CMake (>= 3.22 required). Pin a
    # specific release instead of tracking develop: Kokkos 4.4+ requires
    # C++20, but ci/build-run-prk.sh builds the whole Cxx11 matrix (this
    # included) at -std=c++17, so floating HEAD eventually breaks this
    # build regardless of CMake/compiler fixes. 4.3.01 is the last release
    # before that bump; verified to configure, build (OpenMP and Serial
    # backends), and link/run PRK's stencil-kokkos correctly.
    git clone -b 4.3.01 --depth 1 https://github.com/kokkos/kokkos.git
    cmake -S kokkos -B kokkos/build \
          -DCMAKE_INSTALL_PREFIX=${CI_ROOT}/kokkos \
          -DCMAKE_CXX_COMPILER=${PRK_CXX} \
          -DKokkos_ENABLE_TESTS=OFF \
          ${KOKKOS_BACKEND}
    cmake --build kokkos/build -j2
    cmake --build kokkos/build -j2 --target install
else
    echo "KOKKOS installed..."
    find $CI_ROOT/kokkos -name Kokkos_Core.hpp
fi
