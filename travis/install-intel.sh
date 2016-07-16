#!/bin/bash

echo "TRAVIS_REPO_SLUG=${TRAVIS_REPO_SLUG}"
if [ "${TRAVIS_REPO_SLUG}" = "jeffhammond/PRK" ] ; then
    echo "Using JEFFHAMMOND_PRK_INTEL_SERIAL_NUMBER"
    export INTEL_SERIAL_NUMBER=${JEFFHAMMOND_PRK_INTEL_SERIAL_NUMBER}
elif [ "${TRAVIS_REPO_SLUG}" = "ParRes/Kernels" ] ; then
    echo "Using PARRES_KERNELS_INTEL_SERIAL_NUMBER"
    export INTEL_SERIAL_NUMBER=${PARRES_KERNELS_INTEL_SERIAL_NUMBER}
else
    echo "Cannot install the Intel compiler"
    rm ~/use-intel-compilers
    exit 9
fi

set -x

os=`uname`
TRAVIS_ROOT="$1" # unused for now
PRK_TARGET="$2"

if [ -f ~/use-intel-compilers ] ; then
    case "$os" in
        Darwin)
            echo "Intel tools in Mac Travis is on the TODO list..."
            exit 5
            ;;
        Linux)
            echo "Linux"
            case "$PRK_TARGET" in
                allfortran*)
                    ./travis/install-icc.sh --components ifort,openmp
                    export FC=ifort
                    ;;
                allopenmp)
                    ./travis/install-icc.sh --components icc,openmp
                    ;;
                allmpiopenmp) # must come before allmpi*
                    ./travis/install-icc.sh --components icc,mpi,openmp
                    ;;
                allmpi*)
                    ./travis/install-icc.sh --components icc,mpi
                    ;;
                allupc)
                    if [ "${UPC_IMPL}" = "bupc" ] ; then
                        if [ "${GASNET_CONDUIT}" = "mpi" ] ; then
                            ./travis/install-icc.sh --components icc,mpi
                        else
                            ./travis/install-icc.sh --components icc
                        fi
                    else
                        echo "Intel compilers do not make sense with Intrepid GCC/Clang UPC"
                        exit 11
                    fi
                    ;;
                allpython)
                    echo "TODO: Install Intel Python here"
                    ;;
                alljulia)
                    echo "TODO: Install Intel Julia here (if it exists)"
                    ;;
                allgrappa)
                    ./travis/install-icc.sh --components icc,mpi
                    ;;
                *)
                    ./travis/install-icc.sh --components icc
                    ;;
            esac
            export CC=icc
            export CXX=icpc
            ;;
    esac
fi
