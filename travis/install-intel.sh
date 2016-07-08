#!/bin/bash

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
                *)
                    ./travis/install-icc.sh --components icc
                    ;;
            esac
            export CC=icc
            export CXX=icpc
            find ~/intel
            ;;
    esac
fi
