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

MPI_IMPL=mpich

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
    allcxx)
        echo "C++11"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            sh ./travis/install-gcc.sh $TRAVIS_ROOT
        fi
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT 3.9
        fi
        ;;
    allfortran*)
        echo "Fortran"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            set +e
            brew update
            p=gcc
            if [ "x`brew ls --versions $p`" = "x" ] ; then
                echo "$p is not installed - installing it"
                brew install $p
            else
                echo "$p is installed - upgrading it"
                brew upgrade $p
            fi
            brew list gcc
            set -e
        fi
        if [ "${PRK_TARGET}" = "allfortrancoarray" ] && [ "${CC}" = "gcc" ] ; then
            sh ./travis/install-opencoarrays.sh $TRAVIS_ROOT
        fi
        ;;
    allopenmp)
        echo "OpenMP"
        if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT omp
        fi
        ;;
    allmpi*)
        echo "Any normal MPI"
        # only install clang-omp when necessary
        if [ "${PRK_TARGET}" = "allmpiomp" ] ; then
            sh ./travis/install-clang.sh $TRAVIS_ROOT omp
        fi
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
