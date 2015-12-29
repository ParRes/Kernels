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
            case "$CC" in
                gcc)
                    # I am sure there is a decent way to condense this...
                    if [ -f "/usr/bin/gcc-5.3" ]; then
                        export CMAKE_C_COMPILER=gcc-5.3
                        export CMAKE_CXX_COMPILER=g++-5.3
                    elif [ -f "/usr/bin/gcc-5.2" ]; then
                        export CMAKE_C_COMPILER=gcc-5.2
                        export CMAKE_CXX_COMPILER=g++-5.2
                    elif [ -f "/usr/bin/gcc-5.1" ]; then
                        export CMAKE_C_COMPILER=gcc-5.1
                        export CMAKE_CXX_COMPILER=g++-5.1
                    elif [ -f "/usr/bin/gcc-5" ]; then
                        export CMAKE_C_COMPILER=gcc-5
                        export CMAKE_CXX_COMPILER=g++-5
                    elif [ -f "/usr/bin/gcc-4.9" ]; then
                        export CMAKE_C_COMPILER=gcc-4.9
                        export CMAKE_CXX_COMPILER=g++-4.9
                    elif [ -f "/usr/bin/gcc-4.8" ]; then
                        export CMAKE_C_COMPILER=gcc-4.8
                        export CMAKE_CXX_COMPILER=g++-4.8
                    elif [ -f "/usr/bin/gcc-4.7" ]; then
                        export CMAKE_C_COMPILER=gcc-4.7
                        export CMAKE_CXX_COMPILER=g++-4.7
                    else
                        export CMAKE_C_COMPILER=gcc
                        export CMAKE_CXX_COMPILER=g++
                    fi
                    ;;
                clang)
                    export CMAKE_C_COMPILER=clang
                    export CMAKE_CXX_COMPILER=clang++
                    ;;
            esac
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
    export DGRAPPA_INSTALL_PREFIX=$TRAVIS_ROOT/grappa
    mkdir build && cd build
    export MPI_ROOT=$TRAVIS_ROOT/$MPI_IMPL
    cmake .. -DGRAPPA_INSTALL_PREFIX=$TRAVIS_ROOT/grappa \
             -DMPI_C_LIBRARIES=$MPI_ROOT/lib/libmpi.so \
             -DMPI_C_INCLUDE_PATH=$MPI_ROOT/include \
             -DCMAKE_CC_COMPILER=$CMAKE_CC_COMPILER \
             -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER \
             -DMPI_C_COMPILER="MPICH_CC=$CMAKE_CC_COMPILER $MPI_ROOT/bin/mpicc" \
             -DMPI_CXX_COMPILER="MPICH_CXX=$CMAKE_CXX_COMPILER $MPI_ROOT/bin/mpicxx"
    # END
    make -j4
    make install
else
    echo "Grappa installed..."
    find $TRAVIS_ROOT -name grappa.mk
fi
