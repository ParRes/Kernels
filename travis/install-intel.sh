#!/bin/sh

set -x

os=`uname`
TRAVIS_ROOT="$1" # unused for now
PRK_TARGET="$2"

if [ -f ~/icc ] ; then
    case "$os" in
        Darwin)
            echo "Intel tools in Mac Travis is on the TODO list..."
            exit 5
            ;;
        Linux)
            echo "Linux"
            case "$PRK_TARGET" in
                allserial)
                    ./travis/install-icc.sh --components icc
                    ;;
                allfortran*)
                    ./travis/install-icc.sh --components fortran,openmp
                    ;;
                allopenmp)
                    ./travis/install-icc.sh --components icc,openmp
                    ;;
                allmpiopenmp)
                    ./travis/install-icc.sh --components icc,mpi,openmp
                    ;;
                allmpi*)
                    ./travis/install-icc.sh --components icc,mpi
                    ;;
                *)
                    echo "Intel tools do not support PRK_TARGET=${PRK_TARGET}"
                    exit 10
                    ;;
            esac
            export CC=icc
            export CXX=icpc
            export FC=ifort
            ;;
    esac
fi
