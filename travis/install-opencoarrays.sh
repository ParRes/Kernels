#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ -f ~/use-intel-compilers ] ; then

    echo "OpenCoarrays incompatible with Intel Fortran coarrays"

elif [ ! -d "$TRAVIS_ROOT/opencoarrays" ] ; then

    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            #brew install gcc   || brew update gcc
            #brew install mpich || brew update mpich
            #brew install cmake || brew update cmake
            #cmake --version
            brew install opencoarrays || brew update opencoarrays
            brew list opencoarrays
            which caf
            which cafrun
            ;;

        Linux)
            echo "Linux"
            sh ./travis/install-cmake.sh $TRAVIS_ROOT
            sh ./travis/install-mpi.sh $TRAVIS_ROOT mpich 1
            cd $TRAVIS_ROOT
            git clone --depth 1 https://github.com/sourceryinstitute/opencoarrays.git opencoarrays-source
            cd opencoarrays-source
            mkdir build
            cd build
            # mpif90 is more widely available than mpifort...
            which mpicc
            which mpif90
            mpicc -show
            mpif90 -show
            # override whatever is used in MPI scripts
            for gccversion in "-9" "-8" "-7" "-6" "-5" "-5.3" "-5.2" "-5.1" "" ; do
                if [ -f "`which gfortran$gccversion`" ]; then
                    export PRK_CC="gcc$gccversion"
                    export PRK_FC="gfortran$gccversion"
                    echo "Found GCC: $PRK_FC"
                    $PRK_FC -v
                    break
                fi
            done
            export MPICH_CC=$PRK_CC
            export MPICH_FC=$PRK_FC
            mpicc -show
            mpif90 -show
            CC=$PRK_CC FC=$PRK_FC cmake .. -DCMAKE_INSTALL_PREFIX=$TRAVIS_ROOT/opencoarrays \
                                           -DMPI_C_COMPILER=mpicc -DMPI_Fortran_COMPILER=mpif90
            make -j2
            ctest
            make install
            find $TRAVIS_ROOT -name caf
            find $TRAVIS_ROOT -name cafrun
            ;;
    esac

else

    echo "OpenCoarrays installed..."
    find $TRAVIS_ROOT -name caf
    find $TRAVIS_ROOT -name cafrun

fi
