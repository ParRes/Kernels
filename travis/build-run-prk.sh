set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
PRK_TARGET="$2"

if [ -f ~/use-intel-compilers ] ; then
    export CC=icc
    export CXX=icpc
    export FC=ifort
fi

case "$os" in
    Darwin)
        # Homebrew should put MPI here...
        export MPI_ROOT=/usr/local
        ;;
    Linux)
        export MPI_ROOT=$TRAVIS_ROOT
        ;;
esac
# default to mpirun but override later when necessary
if [ -f ~/use-intel-compilers ] ; then
    # use Intel MPI
    export PRK_MPICC="mpiicc -std=c99"
    export PRK_LAUNCHER=mpirun
else
    export PRK_MPICC="$MPI_ROOT/bin/mpicc -std=c99"
    export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
fi

echo "PRKVERSION=\"'2.16'\"" > common/make.defs

case "$PRK_TARGET" in
    allpython)
        echo "Python"
        which python
        python --version
        export PRK_TARGET_PATH=PYTHON
        python $PRK_TARGET_PATH/p2p.py             10 1024 1024
        python $PRK_TARGET_PATH/p2p-numpy.py       10 1024 1024
        python $PRK_TARGET_PATH/stencil.py         10 1000
        python $PRK_TARGET_PATH/stencil-numpy.py   10 1000
        python $PRK_TARGET_PATH/transpose.py       10 1024
        python $PRK_TARGET_PATH/transpose-numpy.py 10 1024
        ;;
    alljulia)
        echo "Julia"
        case "$os" in
            Darwin)
                export JULIA_PATH=/usr/local/bin/
                ;;
            Linux)
                export JULIA_PATH=$TRAVIS_ROOT/julia/bin/
                ;;
        esac
        ${JULIA_PATH}julia --version
        export PRK_TARGET_PATH=JULIA
        ${JULIA_PATH}julia $PRK_TARGET_PATH/p2p.jl             10 1024 1024
        ${JULIA_PATH}julia $PRK_TARGET_PATH/stencil.jl         10 1000
        ${JULIA_PATH}julia $PRK_TARGET_PATH/transpose.jl       10 1024
        ;;
    allserial)
        echo "Serial"
        echo "CC=$CC -std=c99" >> common/make.defs
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
        $PRK_TARGET_PATH/PIC/pic             10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_TARGET_PATH/PIC/pic             10 1000 1000000 0 1 SINUSOIDAL
        $PRK_TARGET_PATH/PIC/pic             10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_TARGET_PATH/PIC/pic             10 1000 1000000 1 0 PATCH 0 200 100 200
        $PRK_TARGET_PATH/AMR/amr             10 1000 10 3 2 1 5
        ;;
    allfortran*)
        # allfortranserial allfortranopenmp allfortrancoarray allfortranpretty
        echo "Fortran"
        case "$CC" in
            icc)
                echo "FC=ifort" >> common/make.defs
                ;;
            gcc)
                for gccversion in "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
                    if [ -f "`which gfortran$gccversion`" ]; then
                        export PRK_FC="gfortran$gccversion"
                        echo "Found GCC Fortran: $PRK_FC"
                        break
                    fi
                done
                if [ "x$PRK_FC" = "x" ] ; then
                    echo "No Fortran compiler found!"
                    exit 9
                fi
                ;;
            clang)
                echo "LLVM Fortran is not supported."
                exit 9
                echo "FC=flang" >> common/make.defs
                ;;
        esac
        case "$PRK_TARGET" in
            allfortrancoarray)
                if [ "${CC}" = "gcc" ] ; then
                    #echo "FC=$PRK_FC\nCOARRAYFLAG=-fcoarray=single" >> common/make.defs
                    export PRK_CAFC=$TRAVIS_ROOT/opencoarrays/bin/caf
                    echo "FC=$PRK_CAFC\nCOARRAYFLAG=-cpp -std=f2008 -fcoarray=lib" >> common/make.defs
                elif [ "${CC}" = "icc" ] ; then
                    export PRK_CAFC="ifort"
                    echo "FC=$PRK_CAFC\nCOARRAYFLAG=-fpp -std08 -traceback -coarray" >> common/make.defs
                fi
                ;;
            *)
                if [ "${CC}" = "gcc" ] ; then
                    export PRK_FC="$PRK_FC -std=f2008 -cpp"
                    echo "FC=$PRK_FC\nOPENMPFLAG=-fopenmp" >> common/make.defs
                elif [ "${CC}" = "icc" ] ; then
                    # -heap-arrays prevents SEGV in transpose-pretty (?)
                    export PRK_FC="ifort -fpp -std08 -traceback -heap-arrays"
                    echo "FC=$PRK_FC\nOPENMPFLAG=-qopenmp" >> common/make.defs
                fi
                ;;
        esac
        make $PRK_TARGET
        export PRK_TARGET_PATH=FORTRAN
        case "$PRK_TARGET" in
            allfortranserial)
                $PRK_TARGET_PATH/Synch_p2p/p2p               10 1024 1024
                $PRK_TARGET_PATH/Stencil/stencil             10 1000
                $PRK_TARGET_PATH/Transpose/transpose         10 1024 1
                $PRK_TARGET_PATH/Transpose/transpose         10 1024 32
                ;;
            allfortranpretty)
                #$PRK_TARGET_PATH/Synch_p2p/p2p-pretty        10 1024 1024
                # pretty versions do not support tiling...
                $PRK_TARGET_PATH/Stencil/stencil-pretty      10 1000
                $PRK_TARGET_PATH/Transpose/transpose-pretty  10 1024
                ;;
            allfortranopenmp)
                export OMP_NUM_THREADS=2
                $PRK_TARGET_PATH/Synch_p2p/p2p-omp           10 1024 1024 # not threaded yet
                $PRK_TARGET_PATH/Stencil/stencil-omp         10 1000
                $PRK_TARGET_PATH/Transpose/transpose-omp     10 1024 1
                $PRK_TARGET_PATH/Transpose/transpose-omp     10 1024 32
                ;;
            allfortrancoarray)
                export PRK_MPI_PROCS=4
                if [ "${CC}" = "gcc" ] ; then
                    export PRK_LAUNCHER=$TRAVIS_ROOT/opencoarrays/bin/cafrun
                    $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p-coarray       10 1024 1024
                    $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil-coarray     10 1000
                    $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 1
                    $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 32
                elif [ "${CC}" = "icc" ] ; then
                    export FOR_COARRAY_NUM_IMAGES=$PRK_MPI_PROCS
                    $PRK_TARGET_PATH/Synch_p2p/p2p-coarray       10 1024 1024
                    $PRK_TARGET_PATH/Stencil/stencil-coarray     10 1000
                    $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 1
                    $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 32
                fi
                ;;
            esac
        ;;
    allopenmp)
        echo "OpenMP"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            which clang-omp
            echo "CC=clang-omp -std=c99\nOPENMPFLAG=-fopenmp" >> common/make.defs
        else
            echo "CC=$CC -std=c99\nOPENMPFLAG=-fopenmp" >> common/make.defs
        fi
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
        $PRK_TARGET_PATH/Refcount/refcount        $OMP_NUM_THREADS 16777216 1024
        # random is broken right now it seems
        #$PRK_TARGET_PATH/Random/random $OMP_NUM_THREADS 10 16384 32
        ;;
    allmpi1)
        echo "MPI-1"
        echo "MPICC=$PRK_MPICC" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPI1
        export PRK_MPI_PROCS=4
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Reduce/reduce       10 16777216
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32 1
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Random/random       32 20
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_global/global 10 16384
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 0 1 SINUSOIDAL
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 PATCH 0 200 100 200 
        ;;
    allmpio*mp)
        echo "MPI+OpenMP"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            # Mac Clang does not support OpenMP but we should have installed clang-omp via Brew.
            export PRK_MPICC="${PRK_MPICC} -cc=clang-omp"
        fi
        echo "MPICC=$PRK_MPICC\nOPENMPFLAG=-fopenmp" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIOPENMP
        export PRK_MPI_PROCS=2
        export OMP_NUM_THREADS=2
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       $OMP_NUM_THREADS 10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     $OMP_NUM_THREADS 10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose $OMP_NUM_THREADS 10 1024 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     $OMP_NUM_THREADS 10 16777216 32
        ;;
    allmpirma)
        echo "MPI-RMA"
        echo "MPICC=$PRK_MPICC" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPIRMA
        export PRK_MPI_PROCS=4
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allmpishm)
        echo "MPI+MPI"
        echo "MPICC=$PRK_MPICC" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=MPISHM
        export PRK_MPI_PROCS=4
        export PRK_MPISHM_RANKS=$(($PRK_MPI_PROCS/2))
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p                         10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     $PRK_MPISHM_RANKS 10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose $PRK_MPISHM_RANKS 10 1024 32
        ;;
    allshmem)
        echo "SHMEM"
        # This should be fixed by rpath (https://github.com/regrant/sandia-shmem/issues/83)
        export LD_LIBRARY_PATH=$TRAVIS_ROOT/sandia-openshmem/lib:$TRAVIS_ROOT/libfabric/lib:$LD_LIBRARY_PATH
        export SHMEM_ROOT=$TRAVIS_ROOT/sandia-openshmem
        echo "SHMEMTOP=$SHMEM_ROOT\nSHMEMCC=$SHMEM_ROOT/bin/oshcc" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=SHMEM
        export PRK_SHMEM_PROCS=4
        export OSHRUN_LAUNCHER=$TRAVIS_ROOT/hydra/bin/mpirun
        export PRK_LAUNCHER=$SHMEM_ROOT/bin/oshrun
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_SHMEM_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        ;;
    allupc)
        echo "UPC"
        export PRK_UPC_PROCS=4
        case "$UPC_IMPL" in
            gupc)
                case "$CC" in
                    gcc)
                        # If building from source (impossible)
                        #export UPC_ROOT=$TRAVIS_ROOT/gupc
                        # If installing deb file
                        export UPC_ROOT=$TRAVIS_ROOT/gupc/usr/local/gupc
                        ;;
                    clang)
                        echo "Clang UPC is not supported."
                        exit 9
                        export UPC_ROOT=$TRAVIS_ROOT/clupc
                        ;;
                esac
                echo "UPCC=$UPC_ROOT/bin/upc" >> common/make.defs
                export PRK_LAUNCHER=""
                export PRK_LAUNCHER_ARGS="-n $PRK_UPC_PROCS"
                make $PRK_TARGET
                ;;
            bupc)
                export UPC_ROOT=$TRAVIS_ROOT/bupc-$CC
                echo "UPCC=$UPC_ROOT/bin/upcc" >> common/make.defs
                # -N $nodes -n UPC threads -c $cores_per_node
                # -localhost is only for UDP
                case "$GASNET_CONDUIT" in
                    udp)
                        export PRK_LAUNCHER="$UPC_ROOT/bin/upcrun -N 1 -n $PRK_UPC_PROCS -c $PRK_UPC_PROCS -localhost"
                        ;;
                    ofi)
                        export GASNET_SSH_SERVERS="localhost"
                        export LD_LIBRARY_PATH="$TRAVIS_ROOT/libfabric/lib:$LD_LIBRARY_PATH"
                        export PRK_LAUNCHER="$UPC_ROOT/bin/upcrun -v -N 1 -n $PRK_UPC_PROCS -c $PRK_UPC_PROCS"
                        ;;
                    mpi)
                        # see if this is causing Mac tests to hang
                        export MPICH_ASYNC_PROGRESS=1
                        # so that upcrun can find mpirun - why it doesn't cache this from build is beyond me
                        export PATH="$MPI_ROOT/bin:$PATH"
                        export PRK_LAUNCHER="$UPC_ROOT/bin/upcrun -N 1 -n $PRK_UPC_PROCS -c $PRK_UPC_PROCS"
                        ;;
                    *)
                        export PRK_LAUNCHER="$UPC_ROOT/bin/upcrun -N 1 -n $PRK_UPC_PROCS -c $PRK_UPC_PROCS"
                        ;;
                esac
                make $PRK_TARGET default_opt_flags="-Wc,-O3"
                ;;
            *)
                echo "Invalid value of UPC_IMPL ($UPC_IMPL)"
                exit 7
                ;;
        esac
        export PRK_TARGET_PATH=UPC
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       $PRK_LAUNCHER_ARGS 10 1024 1024
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     $PRK_LAUNCHER_ARGS 10 1024
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose $PRK_LAUNCHER_ARGS 10 1024 32
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
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=CHARM++
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        export PRK_LAUNCHER_ARGS="+p$PRK_CHARM_PROCS ++local"
        # For Charm++, the last argument is the overdecomposition factor -->               \|/
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       $PRK_LAUNCHER_ARGS 10 1024 1024  1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     $PRK_LAUNCHER_ARGS 10 1000       1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose $PRK_LAUNCHER_ARGS 10 1024 32    1
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
                export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=$TRAVIS_ROOT/charm-6.7.0/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=AMPI
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        export PRK_LAUNCHER_ARGS="+p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS +isomalloc_sync ++local"
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       $PRK_LAUNCHER_ARGS 10 1024 1024
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     $PRK_LAUNCHER_ARGS 10 1000
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose $PRK_LAUNCHER_ARGS 10 1024 32
        # FIXME Fails with timeout - bug in AMPI?
        #$PRK_LAUNCHER $PRK_TARGET_PATH/Reduce/reduce       $PRK_LAUNCHER_ARGS 10 16777216
        $PRK_LAUNCHER $PRK_TARGET_PATH/Nstream/nstream     $PRK_LAUNCHER_ARGS 10 16777216 32
        $PRK_LAUNCHER $PRK_TARGET_PATH/Sparse/sparse       $PRK_LAUNCHER_ARGS 10 10 5
        $PRK_LAUNCHER $PRK_TARGET_PATH/DGEMM/dgemm         $PRK_LAUNCHER_ARGS 10 1024 32 1
        # FIXME This one hangs - bug in AMPI?
        #$PRK_LAUNCHER $PRK_TARGET_PATH/Random/random       $PRK_LAUNCHER_ARGS 32 20
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_global/global $PRK_LAUNCHER_ARGS 10 16384
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        export FGMPI_ROOT=$TRAVIS_ROOT/fgmpi
        echo "FGMPITOP=$FGMPI_ROOT\nFGMPICC=$FGMPI_ROOT/bin/mpicc -std=c99" >> common/make.defs
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
        ########################
        #. $TRAVIS_ROOT/grappa/bin/settings.sh
        export GRAPPA_PREFIX=$TRAVIS_ROOT/grappa
        export SCRIPT_PATH=$TRAVIS_ROOT/grappa/bin
        ########################
        echo "GRAPPATOP=$TRAVIS_ROOT/grappa" >> common/make.defs
        make $PRK_TARGET
        export PRK_TARGET_PATH=GRAPPA
        export PRK_MPI_PROCS=2
        export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Random/random       32 20
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_global/global 10 16384
        ;;
    allchapel)
        echo "Nothing to do yet"
        ;;
    allhpx3)
        echo "Nothing to do yet"
        ;;
    allhpx5)
        echo "Nothing to do yet"
        ;;
    alllegion)
        echo "Legion"
        echo "LEGIONTOP=$TRAVIS_ROOT/legion" > common/make.defs
        make $PRK_TARGET -k
        ;;
esac
