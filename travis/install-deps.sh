#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

TRAVIS_ROOT="$1"
PRK_TARGET="$2"

case ${TRAVIS_OS_NAME} in
    osx)
        MPI_IMPL=openmpi
        ;;
    linux)
        MPI_IMPL=mpich
        ;;
esac

echo "PWD=$PWD"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        ;;
    alloctave)
        echo "Octave"
        sh ./travis/install-octave.sh $TRAVIS_ROOT
        ;;
    alljulia)
        echo "Julia"
        sh ./travis/install-julia.sh $TRAVIS_ROOT
        ;;
    allrust)
        echo "Rust"
        sh ./travis/install-rust.sh $TRAVIS_ROOT
        ;;
    allc1z)
        echo "C1z"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            sh ./travis/install-gcc.sh $TRAVIS_ROOT
        fi
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT 3.9
        fi
        #if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
        #    sh ./travis/install-musl.sh $TRAVIS_ROOT
        #fi
        ;;
    allcxx)
        echo "C++11"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            sh ./travis/install-gcc.sh $TRAVIS_ROOT
        fi
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT
        fi
        sh ./travis/install-tbb.sh $TRAVIS_ROOT
        sh ./travis/install-pstl.sh $TRAVIS_ROOT
        sh ./travis/install-ranges.sh $TRAVIS_ROOT
        # Boost is whitelisted and obtained from package manager
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            sh ./travis/install-boost.sh $TRAVIS_ROOT
        fi
        # CMake 3.10 or higher is required.
        sh ./travis/install-cmake.sh $TRAVIS_ROOT
        #sh ./travis/install-raja.sh $TRAVIS_ROOT
        sh ./travis/install-kokkos.sh $TRAVIS_ROOT
        #sh ./travis/install-occa.sh $TRAVIS_ROOT
        sh ./travis/install-sycl.sh $TRAVIS_ROOT
        ;;
    allfortran)
        echo "Fortran"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            brew update || true
            brew upgrade gcc || brew install gcc || true
            #
            # Workaround Homebrew issues in Travis CI...
            #
            #==> Installing gcc
            #==> Downloading https://homebrew.bintray.com/bottles/gcc-7.2.0.sierra.bottle.tar.gz
            #==> Pouring gcc-7.2.0.sierra.bottle.tar.gz
            #Error: The `brew link` step did not complete successfully
            #The formula built, but is not symlinked into /usr/local
            #Could not symlink include/c++
            #Target /usr/local/include/c++
            #already exists. You may want to remove it:
            #  rm '/usr/local/include/c++'
            brew link --overwrite --dry-run gcc
            brew link --overwrite gcc || true
        fi
        if [ "${CC}" = "gcc" ] ; then
            sh ./travis/install-cmake.sh $TRAVIS_ROOT
            sh ./travis/install-opencoarrays.sh $TRAVIS_ROOT
        fi
        ;;
    allopenmp)
        echo "OpenMP"
        if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT 3.9
        fi
        ;;
    allmpi)
        echo "Traditional MPI"
        # install except when Intel MPI used
        if [ ! -f ~/use-intel-compilers ] ; then
            sh ./travis/install-mpi.sh $TRAVIS_ROOT $MPI_IMPL 0
        fi
        ;;
    allshmem)
        echo "SHMEM"
        sh ./travis/install-hydra.sh $TRAVIS_ROOT
        sh ./travis/install-libfabric.sh $TRAVIS_ROOT
        sh ./travis/install-sandia-openshmem.sh $TRAVIS_ROOT
        ;;
    allupc)
        echo "UPC"
        case "$UPC_IMPL" in
            gupc)
                # GUPC is working fine
                sh ./travis/install-intrepid-upc.sh $TRAVIS_ROOT
                ;;
            bupc)
                # BUPC is new
                case $GASNET_CONDUIT in
                    ofi)
                        sh ./travis/install-hydra.sh $TRAVIS_ROOT
                        sh ./travis/install-libfabric.sh $TRAVIS_ROOT
                        ;;
                    mpi)
                        if [ ! -f ~/use-intel-compilers ] ; then
                            sh ./travis/install-mpi.sh $TRAVIS_ROOT $MPI_IMPL 0
                        fi
                        ;;
                esac
                sh ./travis/install-berkeley-upc.sh $TRAVIS_ROOT
                ;;
        esac
        ;;
    allcharm++)
        echo "Charm++"
        sh ./travis/install-charm++.sh $TRAVIS_ROOT charm++
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        sh ./travis/install-charm++.sh $TRAVIS_ROOT AMPI
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        sh ./travis/install-fgmpi.sh $TRAVIS_ROOT
        ;;
    allgrappa)
        echo "Grappa"
        sh ./travis/install-cmake.sh $TRAVIS_ROOT
        if [ ! -f ~/use-intel-compilers ] ; then
            sh ./travis/install-mpi.sh $TRAVIS_ROOT $MPI_IMPL 0
        fi
        sh ./travis/install-grappa.sh $TRAVIS_ROOT
        ;;
    allchapel)
        echo "Chapel"
        sh ./travis/install-chapel.sh $TRAVIS_ROOT
        ;;
    allhpx3)
        echo "HPX-3"
        sh ./travis/install-cmake.sh $TRAVIS_ROOT
        sh ./travis/install-hpx3.sh $TRAVIS_ROOT
        ;;
    allhpx5)
        echo "HPX-5"
        sh ./travis/install-autotools.sh $TRAVIS_ROOT
        sh ./travis/install-hpx5.sh $TRAVIS_ROOT
        ;;
    alllegion)
        echo "Legion"
        # GASNet is not needed, it seems
        #sh ./travis/install-gasnet.sh $TRAVIS_ROOT
        sh ./travis/install-legion.sh $TRAVIS_ROOT
        ;;
esac
