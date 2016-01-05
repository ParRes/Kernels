#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        case "$MPI_IMPL" in
            mpich)
                brew install mpich
                ;;
            openmpi)
                brew install openmpi
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 10
                ;;
        esac
        ;;

    Linux)
        echo "Linux"
        case "$CC" in
            gcc)
                # I am sure there is a decent way to condense this...
                if [ -f "/usr/bin/gcc-5.3" ]; then
                    export PRK_CC=gcc-5.3
                    export PRK_CXX=g++-5.3
                elif [ -f "/usr/bin/gcc-5.2" ]; then
                    export PRK_CC=gcc-5.2
                    export PRK_CXX=g++-5.2
                elif [ -f "/usr/bin/gcc-5.1" ]; then
                    export PRK_CC=gcc-5.1
                    export PRK_CXX=g++-5.1
                elif [ -f "/usr/bin/gcc-5" ]; then
                    export PRK_CC=gcc-5
                    export PRK_CXX=g++-5
                elif [ -f "/usr/bin/gcc-4.9" ]; then
                    export PRK_CC=gcc-4.9
                    export PRK_CXX=g++-4.9
                elif [ -f "/usr/bin/gcc-4.8" ]; then
                    export PRK_CC=gcc-4.8
                    export PRK_CXX=g++-4.8
                elif [ -f "/usr/bin/gcc-4.7" ]; then
                    export PRK_CC=gcc-4.7
                    export PRK_CXX=g++-4.7
                else
                    export PRK_CC=gcc
                    export PRK_CXX=g++
                fi
                ;;
            clang)
                export PRK_CC=clang
                export PRK_CXX=clang++
                ;;
        esac
        case "$MPI_IMPL" in
            mpich)
                if [ ! -d "$TRAVIS_ROOT/mpich" ]; then
                    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    tar -xzf mpich-3.2.tar.gz
                    cd mpich-3.2
                    mkdir build && cd build
                    ../configure CC=$PRK_CC CXX=$PRK_CXX --disable-fortran --prefix=$TRAVIS_ROOT/mpich
                    make -j4
                    make install
                else
                    echo "MPICH installed..."
                    find $TRAVIS_ROOT/mpich -name mpiexec
                    find $TRAVIS_ROOT/mpich -name mpicc
                fi
                ;;
            openmpi)
                if [ ! -d "$TRAVIS_ROOT/openmpi" ]; then
                    wget --no-check-certificate -q http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    tar -xjf http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    cd openmpi-1.10.1
                    mkdir build && cd build
                    ../configure CC=$PRK_CC CXX=$PRK_CXX --enable-mpi-fortran=none --prefix=$TRAVIS_ROOT/openmpi
                    make -j4
                    make install
                else
                    echo "OpenMPI installed..."
                    find $TRAVIS_ROOT/openmpi -name mpiexec
                    find $TRAVIS_ROOT/openmpi -name mpicc
                fi


                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
