#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

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
                brew upgrade mpich || brew install mpich || true
                ;;
            openmpi)
                brew upgrade gcc || brew install gcc || true
                brew link --overwrite gcc || true
                brew upgrade openmpi || brew install openmpi || true
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
                for gccversion in "-9" "-8" "-7" "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
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
                for clangversion in "-omp" "-5" "-4" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
                    find /usr/local -name clang$clangversion
                    if [ -f "`which clang$clangversion`" ]; then
                        export PRK_CC="clang$clangversion"
                        export PRK_CXX="clang++$clangversion"
                        echo "Found GCC: $PRK_CC"
                        break
                    fi
                done
                ;;
            icc)
                export PRK_CC=icc
                export PRK_CXX=icpc
                export PRK_FC=ifort
        esac
        case "$MPI_IMPL" in
            mpich)
                if [ ! -f "$TRAVIS_ROOT/bin/mpichversion" ]; then
                    MPICH_V=3.3a2
                    wget --no-check-certificate -q \
                         http://www.mpich.org/static/downloads/${MPICH_V}/mpich-${MPICH_V}.tar.gz || \
                         wget --no-check-certificate -q \
                         https://github.com/pmodels/mpich/archive/v${MPICH_V}.tar.gz
                    tar -xzf mpich-${MPICH_V}.tar.gz || tar -xzf v${MPICH_V}.tar.gz
                    cd mpich-${MPICH_V}
                    # Autotools not required with release tarballs
                    #sh $TRAVIS_HOME/travis/install-autotools.sh $TRAVIS_ROOT
                    #./autogen.sh
                    mkdir build ; cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX --disable-fortran
                    else
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                    fi
                    make -j2
                    make install
                else
                    echo "MPICH installed..."
                    find $TRAVIS_ROOT -name mpiexec
                    find $TRAVIS_ROOT -name mpicc
                fi
                ;;
            openmpi)
                if [ ! -f "$TRAVIS_ROOT/bin/ompi_info" ]; then
                    wget --no-check-certificate -q https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.bz2
                    tar -xjf openmpi-2.1.1.tar.bz2
                    cd openmpi-2.1.1
                    mkdir build && cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX --enable-mpi-fortran=none
                    else
                        ../configure --prefix=$TRAVIS_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                    fi
                    make -j2
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
