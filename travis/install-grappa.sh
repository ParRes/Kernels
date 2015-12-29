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
    which gcc
    #which gcc-4
    #which gcc-4.6
    #which gcc-4.7
    #which gcc-4.8
    find /usr -name gcc\* -type f
    find $TRAVIS_ROOT/$MPI_IMPL
    # END
    export MPI_C_COMPILER=$TRAVIS_ROOT/$MPI_IMPL/bin/mpicc
    export MPI_CXX_COMPILER=$TRAVIS_ROOT/$MPI_IMPL/bin/mpicxx
    export MPI_C_INCLUDE_PATH=$TRAVIS_ROOT/$MPI_IMPL/include
    export MPI_C_LIBRARIES=$TRAVIS_ROOT/$MPI_IMPL/lib/libmpi.so
    ./configure --prefix=$TRAVIS_ROOT/grappa
    cd build/Make+Release
    make -j4 && make install
else
    echo "Grappa installed..."
    find $TRAVIS_ROOT -name grappa.mk
fi
