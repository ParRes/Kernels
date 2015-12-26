#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`

# unused right now, but will be used if building MPI from source
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
                    # yes sudo #
                    #sudo apt-get update -q
                    #sudo apt-get install -y gfortran libcr0 default-jdk
                    #wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.2b3/mpich_3.2b3-1ubuntu_amd64.deb
                    #sudo dpkg -i ./mpich_3.2b3-1ubuntu_amd64.deb
                    # no sudo #
                    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    tar xzf mpich-3.2.tar.gz
                    cd mpich-3.2
                    mkdir build && cd build
                    ../configure --disable-fortran --prefix=$TRAVIS_ROOT/mpich
                    make -j4 && make install
                else
                    echo "MPICH installed..."
                    find $TRAVIS_ROOT/mpich -name mpiexec
                    find $TRAVIS_ROOT/mpich -name mpicc
                    mpicc -show
                fi
                ;;
            openmpi)
                #sudo apt-get update -q
                #sudo apt-get install -y cmake gfortran openmpi-bin openmpi-common libopenmpi-dev
                echo "Need source build of Open-MPI since no-sudo"
                exit 15
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
