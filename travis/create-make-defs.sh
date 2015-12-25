#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
COMPILER="$1"

case "$os" in
    Darwin)
        echo "Mac"
        case "$COMPILER" in
            gcc)
                echo -e "CC=gcc-5\nOPENMPFLAG=-fopenmp\n" > common/make.defs
                ;;
            clang)
                echo "Clang probably does not support OpenMP yet..."
                echo -e "CC=clang\nOPENMPFLAG=-fopenmp\n" > common/make.defs
                ;;
            *)
                echo "Unknown compiler: $COMPILER"
                exit 30
                ;;
        esac
        ;;

    Linux)
        echo "Linux"
        case "$COMPILER" in
            gcc)
                echo -e "CC=gcc\nOPENMPFLAG=-fopenmp\n" > common/make.defs
                ;;
            clang)
                echo "Clang probably does not support OpenMP yet..."
                echo -e "CC=clang\nOPENMPFLAG=-fopenmp\n" > common/make.defs
                ;;
            *)
                echo "Unknown compiler: $COMPILER"
                exit 40
                ;;
        esac
        ;;
esac
