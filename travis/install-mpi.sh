#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

# 1=yes, else no
MPI_FORTRAN="$3"

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
                for gccversion in "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
                    if [ -f "`which gcc$gccversion`" ]; then
                        export PRK_CC="gcc$gccversion"
                        export PRK_CXX="g++$gccversion"
                        export PRK_FC="gfortran$gccversion"
                        echo "Found GCC: $PRK_CC"
                        break
                    fi
                done
                ;;
            clang)
                export PRK_CC=clang
                export PRK_CXX=clang++
                ;;
            cc) # HACK - should be icc
                export PRK_CC=icc
                export PRK_CXX=icpc
                export PRK_FC=ifort
        esac
        case "$MPI_IMPL" in
            mpich)
                if [ ! -f "$TRAVIS_ROOT/bin/mpichversion" ]; then
                    set +e
                    wget --no-check-certificate -q \
                         http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    set -e
                    if [ ! -f "$TRAVIS_ROOT/mpich-3.2.tar.gz" ]; then
                        echo "MPICH download from mpich.org failed - trying Github mirror"
                        wget --no-check-certificate -q \
                             https://github.com/jeffhammond/mpich/archive/v3.2.tar.gz \
                             -O mpich-3.2.tar.gz
                        tar -xzf mpich-3.2.tar.gz
                        cd mpich-3.2
                    else
                        tar -xzf mpich-3.2.tar.gz
                        cd mpich-3.2
                    fi
                    sh $TRAVIS_HOME/travis/install-autotools.sh $TRAVIS_ROOT
                    ./autogen.sh
                    mkdir build ; cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX --disable-fortran
                    else
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                    fi
                    make -j4
                    make install
                else
                    echo "MPICH installed..."
                    find $TRAVIS_ROOT -name mpiexec
                    find $TRAVIS_ROOT -name mpicc
                fi
                ;;
            openmpi)
                if [ ! -f "$TRAVIS_ROOT/bin/ompi_info" ]; then
                    wget --no-check-certificate -q http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    tar -xjf http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    cd openmpi-1.10.1
                    mkdir build && cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX --enable-mpi-fortran=none
                    else
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                    fi
                    make -j4
                    make install
                else
                    echo "OpenMPI installed..."
                    find $TRAVIS_ROOT -name mpiexec
                    find $TRAVIS_ROOT -name mpicc
                fi


                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
