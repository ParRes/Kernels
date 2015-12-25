set -e
set -x

PRK_TARGET="$1"

# make this runtime configurable later
COMPILER=gcc

# Needed for Charm++ and AMPI below
os=`uname`
case "$os" in
    Darwin)
        export MY_CHARM_TOP=$HOME/charm/netlrts-darwin-x86_64-smp
        ;;
    Linux)
        export MY_CHARM_TOP=$HOME/charm/netlrts-linux-x86_64-smp
        ;;
esac

# Needed for FG-MPI below
os=`uname`
case "$os" in
    Darwin)
        export MY_FGMPI_TOP=$HOME/fgmpi
        ;;
    Linux)
        export MY_FGMPI_TOP=/usr
        ;;
esac

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        sh ./travis/create-make-defs.sh $COMPILER
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
        sh ./travis/create-make-defs.sh $COMPILER
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
        echo "MPICC=mpicc" > common/make.defs
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
        echo "MPICC=mpicc" > common/make.defs
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
        echo "MPICC=mpicc" > common/make.defs
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
        echo "MPICC=mpicc" > common/make.defs
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
        echo "SHMEMTOP=$HOME\nSHMEMCC=oshcc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=SHMEM
        export PRK_SHMEM_PROCS=4
        # widely supported
        mpirun -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpirun -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpirun -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allupc)
        echo "UPC"
        # compiler for static thread execution, so set this prior to build
        echo "UPCC=/usr/local/gupc/bin/upc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=UPC
        export PRK_UPC_PROCS=4
        # widely supported
        $PRK_TARGET_PATH/Synch_p2p/p2p -n $PRK_UPC_PROCS 10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil -n $PRK_UPC_PROCS 10 1024
        $PRK_TARGET_PATH/Transpose/transpose -n $PRK_UPC_PROCS 10 1024 32
        ;;
    allcharm++)
        echo "Charm++"
        echo "CHARMTOP=$MY_CHARM_TOP" > common/make.defs
        find $HOME -name charmc
        find $HOME -name charmrun
        make $PRK_TARGET
        export PRK_TARGET_PATH=CHARM++
        export PRK_CHARM_PROCS=4
        # widely supported
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Synch_p2p/p2p +p$PRK_CHARM_PROCS 10 1024 1024
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Stencil/stencil +p$PRK_CHARM_PROCS 10 1024
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Transpose/transpose +p$PRK_CHARM_PROCS 10 1024 32
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        echo "CHARMTOP=$MY_CHARM_TOP" > common/make.defs
        find $HOME -name ampicc
        find $HOME -name charmrun
        make $PRK_TARGET
        export PRK_TARGET_PATH=CHARM++
        export PRK_CHARM_PROCS=4
        # widely supported
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Synch_p2p/p2p +p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS 10 1024 1024
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Stencil/stencil +p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS 10 1024
        $MY_CHARM_TOP/bin/charmrun $PRK_TARGET_PATH/Transpose/transpose +p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS 10 1024 32
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        echo "FGMPITOP=$MY_FGMPI_TOP" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=FG_MPI
        export PRK_MPI_PROCS=2
        export PRK_FGMPI_THREADS=2
        # widely supported
        # widely supported
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Synch_p2p/p2p 10 1024 1024
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Stencil/stencil 10 1024
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        # less support
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Reduce/reduce 10 16777216
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Nstream/nstream 10 16777216 32
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Sparse/sparse 10 10 5
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/DGEMM/dgemm 10 1024 32 1
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Random/random 32 20
        # no serial equivalent
        mpiexec -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Synch_global/global 10 16384
        ;;
esac
