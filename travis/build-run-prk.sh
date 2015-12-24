set -e
set -x

PRK_TARGET="$1"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        make $PRK_TARGET
        export PRK_TARGET_PATH=SERIAL
        # widely supported
        $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil 10 1024 32
        $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        # less support
        $PRK_TARGET_PATH/Reduce/reduce 10 1024
        $PRK_TARGET_PATH/Random/random 64 10 16384
        $PRK_TARGET_PATH/Nstream/nstream 10 16777216 32
        $PRK_TARGET_PATH/Sparse/sparse 10 10 5
        $PRK_TARGET_PATH/DGEMM/dgemm 10 1024 32
        ;;

    allopenmp)
        echo "OpenMP"
        make $PRK_TARGET
        export PRK_TARGET_PATH=OPENMP
        export OMP_NUM_THREADS=4
        # widely supported
        $PRK_TARGET_PATH/Synch_p2p/p2p $OMP_NUM_THREADS 10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil $OMP_NUM_THREADS 10 1024
        $PRK_TARGET_PATH/Transpose/transpose $OMP_NUM_THREADS 10 1024 32
        # less support
        $PRK_TARGET_PATH/Reduce/reduce $OMP_NUM_THREADS 10 16777216
        $PRK_TARGET_PATH/Nstream/nstream $OMP_NUM_THREADS 10 16777216 32
        $PRK_TARGET_PATH/Sparse/sparse $OMP_NUM_THREADS 10 10 5
        $PRK_TARGET_PATH/DGEMM/dgemm $OMP_NUM_THREADS 10 1024 32
        # random is broken right now it seems
        #$PRK_TARGET_PATH/Random/random $OMP_NUM_THREADS 10 16384 32
        # no serial equivalent
        $PRK_TARGET_PATH/Synch_global/global $OMP_NUM_THREADS 10 16384
        $PRK_TARGET_PATH/RefCount_private/private $OMP_NUM_THREADS 16777216
        $PRK_TARGET_PATH/RefCount_shared/shared $OMP_NUM_THREADS 16777216 1024
        ;;
    allmpi1)
        echo "MPI-1"
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPI1
        export PRK_MPI_PROCS=4
        # widely supported
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        # less support
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Reduce/reduce 10 16777216
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream 10 16777216 32
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Sparse/sparse 10 10 5
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/DGEMM/dgemm 10 1024 32 1
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Random/random 32 20
        # no serial equivalent
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_global/global 10 16384
        ;;
    allmpiomp)
        echo "MPI+OpenMP"
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIOMP
        export PRK_MPI_PROCS=4
        # widely supported
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        # less support
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream 10 16777216 32
        ;;
    allmpirma)
        echo "MPI-RMA"
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIRMA
        export PRK_MPI_PROCS=4
        # widely supported
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allmpishm)
        echo "MPI+MPI"
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPISHM
        export PRK_MPI_PROCS=4
        export PRK_MPISHM_RANKS=$(($PRK_MPI_PROCS/2))
        # widely supported
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil $PRK_MPISHM_RANKS 10 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose $PRK_MPISHM_RANKS 10 1024 32
        ;;
    allshmem)
        echo "SHMEM"
        make $PRK_TARGET
        export PRK_TARGET_PATH=SHMEM
        export PRK_SHMEM_PROCS=4
        # widely supported
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpirun -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
esac
