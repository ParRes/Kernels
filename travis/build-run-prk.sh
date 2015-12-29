set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
PRK_TARGET="$2"
# Travis exports this
PRK_COMPILER="$CC"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        echo "CC=$PRK_COMPILER" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=SERIAL
        $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_TARGET_PATH/Reduce/reduce       10 1024
        $PRK_TARGET_PATH/Random/random       64 10 16384
        $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32
        ;;
    allopenmp)
        echo "OpenMP"
        echo "CC=$PRK_COMPILER\nOPENMPFLAG=-fopenmp" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=OPENMP
        export OMP_NUM_THREADS=4
        $PRK_TARGET_PATH/Synch_p2p/p2p            $OMP_NUM_THREADS 10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil          $OMP_NUM_THREADS 10 1000
        $PRK_TARGET_PATH/Transpose/transpose      $OMP_NUM_THREADS 10 1024 32
        $PRK_TARGET_PATH/Reduce/reduce            $OMP_NUM_THREADS 10 16777216
        $PRK_TARGET_PATH/Nstream/nstream          $OMP_NUM_THREADS 10 16777216 32
        $PRK_TARGET_PATH/Sparse/sparse            $OMP_NUM_THREADS 10 10 5
        $PRK_TARGET_PATH/DGEMM/dgemm              $OMP_NUM_THREADS 10 1024 32
        $PRK_TARGET_PATH/Synch_global/global      $OMP_NUM_THREADS 10 16384
        $PRK_TARGET_PATH/RefCount_private/private $OMP_NUM_THREADS 16777216
        $PRK_TARGET_PATH/RefCount_shared/shared   $OMP_NUM_THREADS 16777216 1024
        # random is broken right now it seems
        #$PRK_TARGET_PATH/Random/random $OMP_NUM_THREADS 10 16384 32
        ;;
    allmpi1)
        echo "MPI-1"
        export MPI_ROOT=$TRAVIS_ROOT/mpich
        echo "MPICC=$MPI_ROOT/bin/mpicc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPI1
        export PRK_MPI_PROCS=4
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Reduce/reduce       10 16777216
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32 1
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Random/random       32 20
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_global/global 10 16384
        ;;
    allmpio*mp)
        echo "MPI+OpenMP"
        export MPI_ROOT=$TRAVIS_ROOT/mpich
        echo "MPICC=$MPI_ROOT/bin/mpicc\nOPENMPFLAG=-fopenmp" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIOMP
        export PRK_MPI_PROCS=2
        export OMP_NUM_THREADS=2
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        # less support
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        ;;
    allmpirma)
        echo "MPI-RMA"
        export MPI_ROOT=$TRAVIS_ROOT/mpich
        echo "MPICC=$MPI_ROOT/bin/mpicc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIRMA
        export PRK_MPI_PROCS=4
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allmpishm)
        echo "MPI+MPI"
        export MPI_ROOT=$TRAVIS_ROOT/mpich
        echo "MPICC=$MPI_ROOT/bin/mpicc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPISHM
        export PRK_MPI_PROCS=4
        export PRK_MPISHM_RANKS=$(($PRK_MPI_PROCS/2))
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p                         10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     $PRK_MPISHM_RANKS 10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose $PRK_MPISHM_RANKS 10 1024 32
        ;;
    allshmem)
        echo "SHMEM"
        # BEGIN TEMPORARY FIX
        # This should be fixed by rpath (https://github.com/regrant/sandia-shmem/issues/83)
        export LD_LIBRARY_PATH=$TRAVIS_ROOT/sandia-openshmem/lib:$LD_LIBRARY_PATH
        # END TEMPORARY FIX
        export SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem
        echo "SHMEMTOP=$SHMEM_ROOT\nSHMEMCC=$SHMEM_ROOT/bin/oshcc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=SHMEM
        export PRK_SHMEM_PROCS=4
        find $SHMEM_ROOT
        export OSHRUN_LAUNCHER=$SHMEM_ROOT/bin/mpirun
        export PRK_LAUNCHER=$SHMEM_ROOT/bin/oshrun
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allupc)
        echo "UPC"
        case "$CC" in
            gcc)
                export UPC_ROOT=$TRAVIS_ROOT/gupc
                ;;
            clang)
                export UPC_ROOT=$TRAVIS_ROOT/clupc
                ;;
        esac
        echo "UPCC=$UPC_ROOT/bin/upc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=UPC
        export PRK_UPC_PROCS=4
        $PRK_TARGET_PATH/Synch_p2p/p2p -n $PRK_UPC_PROCS       10 1024 1024
        $PRK_TARGET_PATH/Stencil/stencil -n $PRK_UPC_PROCS     10 1024
        $PRK_TARGET_PATH/Transpose/transpose -n $PRK_UPC_PROCS 10 1024 32
        ;;
    allcharm++)
        echo "Charm++"
        os=`uname`
        case "$os" in
            Darwin)
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64-smp
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=CHARM++
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        # For Charm++, the last argument is the overdecomposition factor -->               \|/
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       +p$PRK_CHARM_PROCS 10 1024 1024  1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     +p$PRK_CHARM_PROCS 10 1000       1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose +p$PRK_CHARM_PROCS 10 1024 32    1
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        os=`uname`
        case "$os" in
            Darwin)
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64-smp
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=AMPI
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        export PRK_LAUNCHER_ARGS=+p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS +isomalloc_sync
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       $PRK_LAUNCHER_ARGS 10 1024 1024
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     $PRK_LAUNCHER_ARGS 10 1000
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose $PRK_LAUNCHER_ARGS 10 1024 32
        # FIXME Fails with timeout - bug in AMPI?
        #$PRK_LAUNCHER $PRK_TARGET_PATH/Reduce/reduce       $PRK_LAUNCHER_ARGS 10 16777216
        $PRK_LAUNCHER $PRK_TARGET_PATH/Nstream/nstream     $PRK_LAUNCHER_ARGS 10 16777216 32
        $PRK_LAUNCHER $PRK_TARGET_PATH/Sparse/sparse       $PRK_LAUNCHER_ARGS 10 10 5
        $PRK_LAUNCHER $PRK_TARGET_PATH/DGEMM/dgemm         $PRK_LAUNCHER_ARGS 10 1024 32 1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Random/random       $PRK_LAUNCHER_ARGS 32 20
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_global/global $PRK_LAUNCHER_ARGS 10 16384
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        export FGMPI_ROOT=$TRAVIS_ROOT/fgmpi
        echo "FGMPITOP=$FGMPI_ROOT\nFGMPICC=$FGMPI_ROOT/bin/mpicc" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=FG_MPI
        export PRK_MPI_PROCS=2
        export PRK_FGMPI_THREADS=2
        export PRK_LAUNCHER=$FGMPI_ROOT/bin/mpiexec
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        # FIXME Fails with:
        # ERROR: rank 2 has work tile smaller then stencil radius
        #$PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Reduce/reduce       10 16777216
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32 1
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Random/random       32 20
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Synch_global/global 10 16384
        ;;
    allgrappa)
        echo "Grappa"
        # compiler for static thread execution, so set this prior to build
        echo "GRAPPATOP=$TRAVIS_ROOT/grappa" > common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=GRAPPA
        export PRK_MPI_PROCS=4
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
esac
