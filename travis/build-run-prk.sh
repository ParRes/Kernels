set -e
set -x

PRK_TARGET="$1"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        make $PRK_TARGET
        # widely supported
        ./SERIAL/Synch_p2p/p2p 10 1024 1024
        ./SERIAL/Stencil/stencil 10 1024 32
        ./SERIAL/Transpose/transpose 10 1024 32
        # less support
        ./SERIAL/Reduce/reduce 10 1024
        ./SERIAL/Random/random 64 10 16384
        ./SERIAL/Nstream/nstream 10 16777216 32
        ./SERIAL/Sparse/sparse 10 10 5
        ./SERIAL/DGEMM/dgemm 10 1024 32
        ;;

    allopenmp)
        echo "OpenMP"
        make $PRK_TARGET
        export OMP_NUM_THREADS=4
        # widely supported
        ./OPENMP/Synch_p2p/p2p $OMP_NUM_THREADS 10 1024 1024
        ./OPENMP/Stencil/stencil $OMP_NUM_THREADS 10 1024 32
        ./OPENMP/Transpose/transpose $OMP_NUM_THREADS 10 1024 32
        # less support
        ./OPENMP/Reduce/reduce $OMP_NUM_THREADS 10 1024
        ./OPENMP/Random/random $OMP_NUM_THREADS 64 10 16384
        ./OPENMP/Nstream/nstream $OMP_NUM_THREADS 10 16777216 32
        ./OPENMP/Sparse/sparse $OMP_NUM_THREADS 10 10 5
        ./OPENMP/DGEMM/dgemm $OMP_NUM_THREADS 10 1024 32
        # no serial equivalent
        ./OPENMP/Synch_global/global $OMP_NUM_THREADS 10 16384
        ./OPENMP/RefCount_private/private $OMP_NUM_THREADS 16777216
        ./OPENMP/RefCount_shared/shared $OMP_NUM_THREADS 16777216 1024
        ;;
esac
