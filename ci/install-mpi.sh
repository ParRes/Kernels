#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
CI_ROOT="$1"
MPI_IMPL="$2"

# 1=yes, else no
MPI_FORTRAN="$3"

case "$os" in
    Darwin)
        echo "Mac"
        case "$MPI_IMPL" in
            mpich)
                brew upgrade mpich || brew install mpich || true
                ;;
            openmpi)
                brew upgrade gcc || brew install gcc || true
                brew link --overwrite gcc || true
                brew upgrade openmpi || brew install openmpi || true
                ;;
        esac
        ;;
    Linux)
        echo "Linux"
        if [ "x$MPI_FORTRAN" != "x1" ] ; then
            case "$MPI_IMPL" in
                mpich)
                    sudo apt-get install libmpich-dev
                    ;;
                openmpi)
                    sudo apt-get install libopenmpi-dev
                    ;;
            esac
        else
            case "$CC" in
                gcc)
                    for gccversion in "-9" "-8" "-7" "-6" "-5" "" ; do
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
            esac
            case "$MPI_IMPL" in
                mpich)
                    if [ ! -f "$CI_ROOT/bin/mpichversion" ]; then
                        MPICH_V=3.3a2
                        wget --no-check-certificate -q \
                             http://www.mpich.org/static/downloads/${MPICH_V}/mpich-${MPICH_V}.tar.gz || \
                             wget --no-check-certificate -q \
                             https://github.com/pmodels/mpich/archive/v${MPICH_V}.tar.gz
                        tar -xzf mpich-${MPICH_V}.tar.gz || tar -xzf v${MPICH_V}.tar.gz
                        cd mpich-${MPICH_V}
                        # Autotools not required with release tarballs
                        #sh $CI_HOME/ci/install-autotools.sh $CI_ROOT
                        #./autogen.sh
                        mkdir build ; cd build
                        if [ "x$MPI_FORTRAN" != "x1" ] ; then
                            ../configure --prefix=$CI_ROOT CC=$PRK_CC CXX=$PRK_CXX --disable-fortran
                        else
                            ../configure --prefix=$CI_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                        fi
                        make -j2
                        make install
                    else
                        echo "MPICH installed..."
                        find $CI_ROOT -name mpiexec
                        find $CI_ROOT -name mpicc
                    fi
                    ;;
                openmpi)
                    if [ ! -f "$CI_ROOT/bin/ompi_info" ]; then
                        wget --no-check-certificate -q https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.bz2
                        tar -xjf openmpi-2.1.1.tar.bz2
                        cd openmpi-2.1.1
                        mkdir build && cd build
                        if [ "x$MPI_FORTRAN" != "x1" ] ; then
                            ../configure --prefix=$CI_ROOT CC=$PRK_CC CXX=$PRK_CXX --enable-mpi-fortran=none
                        else
                            ../configure --prefix=$CI_ROOT CC=$PRK_CC CXX=$PRK_CXX FC=$PRK_FC
                        fi
                        make -j2
                        make install
                    else
                        echo "OpenMPI installed..."
                        find $CI_ROOT -name mpiexec
                        find $CI_ROOT -name mpicc
                    fi
                    ;;
            esac
        fi
        ;;
esac
