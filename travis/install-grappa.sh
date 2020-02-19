#!/bin/sh

set -e
set -x

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

os=`uname`
TRAVIS_ROOT="$1"

# TODO: Make compiler and MPI configurable...

if [ ! -d "$TRAVIS_ROOT/grappa" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            #brew update
            #brew install ruby boost
            # Homebrew location
            export MPI_ROOT=/usr/local
            ;;

        Linux)
            echo "Linux"
            export MPI_ROOT=$TRAVIS_ROOT
            ;;
    esac

    cd $TRAVIS_ROOT
    git clone --depth 1 https://github.com/uwsampa/grappa.git grappa-source
    cd grappa-source
    # DEBUG
    #find /usr -name gcc\* -type f
    #find $TRAVIS_ROOT
    # END
    # Invoking CMake directly
    mkdir build && cd build
    if [ -f ~/use-intel-compilers ] ; then
        cmake .. -DGRAPPA_INSTALL_PREFIX=$TRAVIS_ROOT/grappa \
                 -DCMAKE_C_COMPILER="mpiicc" \
                 -DCMAKE_CXX_COMPILER="mpiicpc" \
                 -DMPI_C_COMPILER="mpiicc" \
                 -DMPI_CXX_COMPILER="mpiicpc"
     else
        cmake .. -DGRAPPA_INSTALL_PREFIX=$TRAVIS_ROOT/grappa \
                 -DCMAKE_C_COMPILER="$MPI_ROOT/bin/mpicc" \
                 -DCMAKE_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" \
                 -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc" \
                 -DMPI_CXX_COMPILER="$MPI_ROOT/bin/mpicxx"
                 #-DMPI_C_LINK_FLAGS="-L$MPI_ROOT/lib" \
                 #-DMPI_C_LIBRARIES="-lmpi" \
                 #-DMPI_C_INCLUDE_PATH="$MPI_ROOT/include" \
                 #-DMPI_CXX_LINK_FLAGS="-L$MPI_ROOT/lib" \
                 #-DMPI_CXX_LIBRARIES="-lmpicxx -lmpi" \
                 #-DMPI_CXX_INCLUDE_PATH="$MPI_ROOT/include" \
    fi
    make -j2
    make install
else
    echo "Grappa installed..."
    find $TRAVIS_ROOT -name grappa.mk
fi
