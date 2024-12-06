#!/bin/sh

set -e
set -x

CI_ROOT="$1"
PRK_TARGET="$2"
os=`uname`

# update package managers once at the beginning
case $os in
    Darwin)
        brew update
        ;;
    Linux)
        sudo apt-get update -y
        #sudo apt-get upgrade -y
        ;;
esac

case $os in
    Darwin)
        MPI_IMPL=openmpi
        ;;
    Linux)
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
        sh ./ci/install-octave.sh $CI_ROOT
        ;;
    alljulia)
        echo "Julia"
        sh ./ci/install-julia.sh $CI_ROOT
        ;;
    allpython)
        echo "Python"
        sh ./ci/install-python.sh $CI_ROOT
        ;;
    allrust)
        echo "Rust"
        sh ./ci/install-rust.sh $CI_ROOT
        ;;
    allc1z)
        echo "C1z"
        if [ "${CC}" = "gcc" ] ; then
            sh ./ci/install-gcc.sh $CI_ROOT
        fi
        if [ "${CC}" = "clang" ] ; then
            sh ./ci/install-clang.sh $CI_ROOT
        fi
        #if [ "$os" = "Linux" ] ; then
        #    sh ./ci/install-musl.sh $CI_ROOT
        #fi
        ;;
    allcxx)
        echo "C++11"
        if [ "${CC}" = "gcc" ] ; then
            sh ./ci/install-gcc.sh $CI_ROOT
        fi
        if [ "${CC}" = "clang" ] ; then
            sh ./ci/install-clang.sh $CI_ROOT
        fi
        sh ./ci/install-tbb.sh $CI_ROOT
        sh ./ci/install-pstl.sh $CI_ROOT
        sh ./ci/install-ranges.sh $CI_ROOT
        sh ./ci/install-boost.sh $CI_ROOT
        # CMake 3.10 or higher is required.
        sh ./ci/install-cmake.sh $CI_ROOT
        #sh ./ci/install-raja.sh $CI_ROOT
        sh ./ci/install-kokkos.sh $CI_ROOT
        #sh ./ci/install-occa.sh $CI_ROOT
        sh ./ci/install-sycl.sh $CI_ROOT
        ;;
    allfortran)
        echo "Fortran"
        if [ "${CC}" = "gcc" ] ; then
            sh ./ci/install-gcc.sh $CI_ROOT
            sh ./ci/install-opencoarrays.sh $CI_ROOT
        fi
        ;;
    allopenmp)
        echo "OpenMP"
        if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
            sh ./ci/install-clang.sh $CI_ROOT 3.9
        fi
        ;;
    allmpi)
        echo "Traditional MPI"
        # install except when Intel MPI used
        sh ./ci/install-mpi.sh $CI_ROOT $MPI_IMPL 0
        ;;
    allshmem)
        echo "SHMEM"
        sh ./ci/install-hydra.sh $CI_ROOT
        sh ./ci/install-libfabric.sh $CI_ROOT
        sh ./ci/install-sandia-openshmem.sh $CI_ROOT
        ;;
    allupc)
        echo "UPC"
        case "$UPC_IMPL" in
            gupc)
                # GUPC is working fine
                sh ./ci/install-intrepid-upc.sh $CI_ROOT
                ;;
            bupc)
                # BUPC is new
                case $GASNET_CONDUIT in
                    ofi)
                        sh ./ci/install-hydra.sh $CI_ROOT
                        sh ./ci/install-libfabric.sh $CI_ROOT
                        ;;
                    mpi)
                        sh ./ci/install-mpi.sh $CI_ROOT $MPI_IMPL 0
                        ;;
                esac
                sh ./ci/install-berkeley-upc.sh $CI_ROOT
                ;;
        esac
        ;;
    allcharm++)
        echo "Charm++"
        sh ./ci/install-charm++.sh $CI_ROOT charm++
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        sh ./ci/install-charm++.sh $CI_ROOT AMPI
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        sh ./ci/install-fgmpi.sh $CI_ROOT
        ;;
    allgrappa)
        echo "Grappa"
        sh ./ci/install-cmake.sh $CI_ROOT
        sh ./ci/install-mpi.sh $CI_ROOT $MPI_IMPL 0
        sh ./ci/install-grappa.sh $CI_ROOT
        ;;
    allchapel)
        echo "Chapel"
        sh ./ci/install-chapel.sh $CI_ROOT
        ;;
    allhpx3)
        echo "HPX-3"
        sh ./ci/install-cmake.sh $CI_ROOT
        sh ./ci/install-hpx3.sh $CI_ROOT
        ;;
    allhpx5)
        echo "HPX-5"
        sh ./ci/install-autotools.sh $CI_ROOT
        sh ./ci/install-hpx5.sh $CI_ROOT
        ;;
    alllegion)
        echo "Legion"
        # GASNet is not needed, it seems
        #sh ./ci/install-gasnet.sh $CI_ROOT
        sh ./ci/install-legion.sh $CI_ROOT
        ;;
esac
