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
        case "$MPI_IMPL" in
            mpich)
                if [ ! -d "$TRAVIS_ROOT/mpich" ]; then
                    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    tar -xzf mpich-3.2.tar.gz
                    cd mpich-3.2
                    mkdir build && cd build
                    ../configure --disable-fortran --prefix=$TRAVIS_ROOT/mpich
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
                    ../configure --enable-mpi-fortran=none --prefix=$TRAVIS_ROOT/openmpi
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
