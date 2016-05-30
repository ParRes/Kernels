set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/opencoarrays" ]; then
    set +e
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            brew install gcc6
            brew install mpich
            brew install cmake
            ;;

        Linux)
            echo "Linux"
            ;;
    esac
    set -e

    cd $TRAVIS_ROOT
    git clone --depth 10 https://github.com/sourceryinstitute/opencoarrays.git opencoarrays-source
    cd opencoarrays-source
    mkdir build
    cd build
    which mpifort
    mpifort -show
    CC=mpicc FC=mpifort cmake .. -DCMAKE_INSTALL_PREFIX=$TRAVIS_ROOT/opencoarrays \
                                 -DMPI_C_COMPILER=mpicc -DMPI_Fortran_COMPILER=mpifort
    make
    ctest
    make install
    find $TRAVIS_ROOT -name caf
    find $TRAVIS_ROOT -name cafrun
else
    echo "OpenCoarrays installed..."
    find $TRAVIS_ROOT -name caf
    find $TRAVIS_ROOT -name cafrun
fi
