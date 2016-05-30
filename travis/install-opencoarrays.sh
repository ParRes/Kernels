set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/opencoarrays" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            set +e
            brew update
            brew install gcc6
            brew install mpich
            brew unlink cmake
            brew install cmake
            cmake --version
            set -e
            ;;

        Linux)
            echo "Linux"
            sh ./travis/install-mpi.sh $TRAVIS_ROOT mpich
            ;;
    esac

    cd $TRAVIS_ROOT
    git clone --depth 10 https://github.com/sourceryinstitute/opencoarrays.git opencoarrays-source
    cd opencoarrays-source
    mkdir build
    cd build
    which mpicc
    which mpifort
    mpicc -show
    mpifort -show
    export MPICH_CC=gcc-6
    export MPICH_FC=gfortran-6
    mpicc -show
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
