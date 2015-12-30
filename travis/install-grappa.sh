set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

# TODO: Make compiler and MPI configurable...

if [ ! -d "$TRAVIS_ROOT/grappa" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            brew install ruby boost
            ;;

        Linux)
            echo "Linux"
            ;;
    esac

    cd $TRAVIS_ROOT
    git clone https://github.com/uwsampa/grappa.git grappa-source
    cd grappa-source
    # DEBUG
    find /usr -name gcc\* -type f
    find $TRAVIS_ROOT/$MPI_IMPL
    # END
    # Using Grappa's configure script
    #./configure --prefix=$TRAVIS_ROOT/grappa
    #cd build/Make+Release
    # Invoking CMake directly
    mkdir build && cd build
    export MPI_ROOT=$TRAVIS_ROOT/$MPI_IMPL
    cmake .. \
             -DCMAKE_C_COMPILER="$MPI_ROOT/bin/mpicc" \
             -DCMAKE_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" \
             -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc" \
             -DMPI_C_LINK_FLAGS="-L$MPI_ROOT/lib" \
             -DMPI_C_LIBRARIES="-lmpi" \
             -DMPI_C_INCLUDE_PATH="$MPI_ROOT/include" \
             -DMPI_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" \
             -DMPI_CXX_LINK_FLAGS="-L$MPI_ROOT/lib" \
             -DMPI_CXX_LIBRARIES="-lmpicxx -lmpi" \
             -DMPI_CXX_INCLUDE_PATH="$MPI_ROOT/include" \
             -DGRAPPA_INSTALL_PREFIX=$TRAVIS_ROOT/grappa
    # END
    make -j4
    make install
else
    echo "Grappa installed..."
    find $TRAVIS_ROOT -name grappa.mk
fi
