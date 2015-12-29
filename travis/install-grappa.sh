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
                        export PRK_CC=gcc-5.3
                        export PRK_CXX=g++-5.3
                    elif [ -f "/usr/bin/gcc-5.2" ]; then
                        export PRK_CC=gcc-5.2
                        export PRK_CXX=g++-5.2
                    elif [ -f "/usr/bin/gcc-5.1" ]; then
                        export PRK_CC=gcc-5.1
                        export PRK_CXX=g++-5.1
                    elif [ -f "/usr/bin/gcc-5" ]; then
                        export PRK_CC=gcc-5
                        export PRK_CXX=g++-5
                    elif [ -f "/usr/bin/gcc-4.9" ]; then
                        export PRK_CC=gcc-4.9
                        export PRK_CXX=g++-4.9
                    elif [ -f "/usr/bin/gcc-4.8" ]; then
                        export PRK_CC=gcc-4.8
                        export PRK_CXX=g++-4.8
                    elif [ -f "/usr/bin/gcc-4.7" ]; then
                        export PRK_CC=gcc-4.7
                        export PRK_CXX=g++-4.7
                    else
                        export PRK_CC=gcc
                        export PRK_CXX=g++
                    fi
                    ;;
                clang)
                    export PRK_CC=clang
                    export PRK_CXX=clang++
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
    mkdir build && cd build
    export MPI_ROOT=$TRAVIS_ROOT/$MPI_IMPL
    cmake .. \
             -DCMAKE_C_COMPILER="$PRK_CC" \
             -DCMAKE_CXX_COMPILER="$PRK_CXX" \
             -DMPI_C_COMPILER="\"MPICH_CC=$PRK_CC $MPI_ROOT/bin/mpicc\"" \
             -DMPI_C_LINK_FLAGS="-L$MPI_ROOT/lib" \
             -DMPI_C_LIBRARIES="-lmpi" \
             -DMPI_C_INCLUDE_PATH="$MPI_ROOT/include" \
             -DMPI_CXX_COMPILER="\"MPICH_CXX=$PRK_CXX $MPI_ROOT/bin/mpicxx\"" \
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
