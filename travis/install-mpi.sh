#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
MPI_IMPL="$1"

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
        sudo apt-get update -q
        case "$MPI_IMPL" in
            mpich)
                sudo apt-get install -q gfortran libcr0 default-jdk
                wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.2b3/mpich_3.2b3-1ubuntu_amd64.deb
                sudo dpkg -i ./mpich_3.2b3-1ubuntu_amd64.deb
                ;;
            openmpi)
                sudo apt-get install -q cmake gfortran openmpi-bin openmpi-common libopenmpi-dev
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
