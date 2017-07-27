#!/usr/bin/env bash
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

echo "PRKVERSION=\"'2.16'\"" > common/make.defs

case "$PRK_TARGET" in
    allpython)
        echo "Python"
        which python
        python --version
        export PRK_TARGET_PATH=PYTHON
        python $PRK_TARGET_PATH/p2p.py             10 100 100
        python $PRK_TARGET_PATH/p2p-numpy.py       10 1024 1024
        python $PRK_TARGET_PATH/stencil.py         10 100
        python $PRK_TARGET_PATH/stencil-numpy.py   10 1000
        python $PRK_TARGET_PATH/transpose.py       10 100
        python $PRK_TARGET_PATH/transpose-numpy.py 10 1024
        ;;
    alloctave)
        echo "Octave"
        which octave
        octave --version
        export PRK_TARGET_PATH=OCTAVE
        ./$PRK_TARGET_PATH/p2p.m               10 100 100
        ./$PRK_TARGET_PATH/stencil.m           10 100
        ./$PRK_TARGET_PATH/stencil-pretty.m    10 1000
        ./$PRK_TARGET_PATH/transpose.m         10 100
        ./$PRK_TARGET_PATH/transpose-pretty.m  10 1024
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
        $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5
        ;;
    allrust)
        echo "Rust"
        which rustc
        rustc --version
        make $PRK_TARGET
        export PRK_TARGET_PATH=RUST
        ./$PRK_TARGET_PATH/p2p               10 100 100
        ./$PRK_TARGET_PATH/stencil           10 100
        ./$PRK_TARGET_PATH/transpose         10 100
        ;;
    allc1z)
        echo "C1z"
        export PRK_TARGET_PATH=C1z
        case $CC in
            g*)
                for major in "-9" "-8" "-7" "-6" "-5" "" ; do
                  if [ -f "`which ${CC}${major}`" ]; then
                      export PRK_CC="${CC}${major}"
                      echo "Found C: $PRK_CC"
                      break
                  fi
                done
                if [ "x$PRK_CC" = "x" ] ; then
                    export PRK_CC="${CC}"
                fi
                ;;
            clang*)
                for version in "-5" "-4" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
                  if [ -f "`which ${CC}${version}`" ]; then
                      export PRK_CC="${CC}${version}"
                      echo "Found C: $PRK_CC"
                      break
                  fi
                done
                if [ "x$PRK_CC" = "x" ] ; then
                    export PRK_CC="${CC}"
                fi
                ;;
        esac
        ${PRK_CC} -v
        # Need to increment this for CPLEX (some day)
        echo "CC=${PRK_CC} -std=c11 -DPRK_USE_GETTIMEOFDAY" >> common/make.defs
        echo "EXTRA_CLIBS=-lm -lpthread" >> common/make.defs

        # C11 without external parallelism
        make -C $PRK_TARGET_PATH p2p stencil transpose p2p-innerloop
        $PRK_TARGET_PATH/p2p             10 1024 1024
        $PRK_TARGET_PATH/p2p             10 1024 1024 100 100
        $PRK_TARGET_PATH/p2p-innerloop   10 1024
        $PRK_TARGET_PATH/stencil         10 1000
        $PRK_TARGET_PATH/transpose       10 1024 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil 10 200 $s $r
            done
        done

        # C11 with POSIX or C11 thread parallelism
        # not testing C11 threads for now
        make -C $PRK_TARGET_PATH transpose-thread
        $PRK_TARGET_PATH/transpose-thread   10 1024 512

        # C11 with OpenMP
        export OMP_NUM_THREADS=2
        case "$CC" in
            g*)
                # Host
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                make -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp         10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-innerloop-openmp     10 1024
                $PRK_TARGET_PATH/stencil-openmp           10 1000
                $PRK_TARGET_PATH/transpose-openmp         10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 $s $r
                    done
                done
                # Offload
                echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                make -C $PRK_TARGET_PATH target
                $PRK_TARGET_PATH/stencil-target     10 1000
                $PRK_TARGET_PATH/transpose-target   10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-target 10 200 $s $r
                    done
                done
                ;;
            clang*)
                # Host
                echo "Skipping Clang since OpenMP support probably missing"
                #echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                #make -C $PRK_TARGET_PATH openmp
                #$PRK_TARGET_PATH/p2p-tasks-openmp         10 1024 1024 100 100
                #$PRK_TARGET_PATH/stencil-openmp           10 1000
                #$PRK_TARGET_PATH/transpose-penmp          10 1024 32
                #echo "Test stencil code generator"
                #for s in star grid ; do
                #    for r in 1 2 3 4 5 6 7 8 9 ; do
                #        $PRK_TARGET_PATH/stencil-penmp 10 200 $s $r
                #    done
                #done
                ;;
            ic*)
                # Host
                echo "OPENMPFLAG=-qopenmp" >> common/make.defs
                make -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp         10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-innerloop-openmp     10 1024 1024
                $PRK_TARGET_PATH/stencil-openmp           10 1000
                $PRK_TARGET_PATH/transpose-openmp         10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 $s $r
                    done
                done
                # Offload - not supported on MacOS
                if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    echo "OFFLOADFLAG=-qopenmp -qopenmp-offload=host" >> common/make.defs
                    make -C $PRK_TARGET_PATH target
                    $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                    $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                    #echo "Test stencil code generator"
                    for s in star grid ; do
                        for r in 1 2 3 4 5 6 7 8 9 ; do
                            $PRK_TARGET_PATH/stencil-openmp-target 10 200 $s $r
                        done
                    done
                fi
                ;;
            *)
                echo "Figure out your OpenMP flags..."
                ;;
        esac

        # C11 with Cilk
        if [ "${CC}" = "gcc" ] ; then
            echo "CILKFLAG=-fcilkplus" >> common/make.defs
            make -C $PRK_TARGET_PATH stencil-cilk transpose-cilk
            $PRK_TARGET_PATH/stencil-cilk     10 1000
            $PRK_TARGET_PATH/transpose-cilk   10 1024 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 6 7 8 9 ; do
                    $PRK_TARGET_PATH/stencil-cilk 10 200 $s $r
                done
            done
        fi

        ;;
    allcxx)
        echo "C++11"
        export PRK_TARGET_PATH=Cxx11
        case $CXX in
            g++)
                for major in "-9" "-8" "-7" "-6" "-5" "" ; do
                  if [ -f "`which ${CXX}${major}`" ]; then
                      export PRK_CXX="${CXX}${major}"
                      echo "Found C++: $PRK_CXX"
                      break
                  fi
                done
                if [ "x$PRK_CXX" = "x" ] ; then
                    export PRK_CXX="${CXX}"
                fi
                ;;
            clang++)
                for version in "-5" "-4" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
                  if [ -f "`which ${CXX}${version}`" ]; then
                      export PRK_CXX="${CXX}${version}"
                      echo "Found C++: $PRK_CXX"
                      break
                  fi
                done
                if [ "x$PRK_CXX" = "x" ] ; then
                    export PRK_CXX="${CXX}"
                fi
                ;;
        esac
        ${PRK_CXX} -v
        # Need to increment this for PSTL
        echo "CXX=${PRK_CXX} -std=c++11" >> common/make.defs

        # C++11 without external parallelism
        make -C $PRK_TARGET_PATH transpose-valarray
        $PRK_TARGET_PATH/transpose-valarray 10 1024 32

        # C++11 without external parallelism
        make -C $PRK_TARGET_PATH p2p-vector p2p-innerloop-vector stencil-vector transpose-vector
        $PRK_TARGET_PATH/p2p-vector              10 1024 1024
        $PRK_TARGET_PATH/p2p-vector              10 1024 1024 100 100
        $PRK_TARGET_PATH/p2p-innerloop-vector    10 1024
        $PRK_TARGET_PATH/stencil-vector          10 1000
        $PRK_TARGET_PATH/transpose-vector        10 1024 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil-vector 10 200 $s $r
            done
        done

        # C++11 with rangefor
        echo "BOOSTFLAG=-DUSE_BOOST" >> common/make.defs
        make -C $PRK_TARGET_PATH rangefor
        $PRK_TARGET_PATH/stencil-vector-rangefor     10 1000
        $PRK_TARGET_PATH/transpose-vector-rangefor   10 1024 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil-vector-rangefor 10 200 $s $r
            done
        done

        # C++11 with STL (C++17 PSTL disabled)
        echo "PSTLFLAG=" >> common/make.defs
        make -C $PRK_TARGET_PATH pstl
        $PRK_TARGET_PATH/stencil-vector-pstl     10 1000
        $PRK_TARGET_PATH/transpose-vector-pstl   10 1024 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil-vector-pstl 10 200 $s $r
            done
        done

        # C++11 with OpenMP
        export OMP_NUM_THREADS=2
        case "$CC" in
            gcc)
                # Host
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                make -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-innerloop-vector-openmp stencil-vector-openmp transpose-vector-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-innerloop-vector-openmp      10 1024
                $PRK_TARGET_PATH/stencil-vector-openmp            10 1000
                $PRK_TARGET_PATH/transpose-vector-openmp          10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-vector-openmp 10 200 $s $r
                    done
                done
                # Offload
                echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                make -C $PRK_TARGET_PATH target
                $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-vector-openmp 10 200 $s $r
                    done
                done
                ;;
            clang)
                # Host
                echo "Skipping Clang since OpenMP support probably missing"
                #echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                #make -C $PRK_TARGET_PATH openmp
                #$PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                #$PRK_TARGET_PATH/stencil-vector-openmp            10 1000
                #$PRK_TARGET_PATH/transpose-vector-openmp          10 1024 32
                #echo "Test stencil code generator"
                #for s in star grid ; do
                #    for r in 1 2 3 4 5 6 7 8 9 ; do
                #        $PRK_TARGET_PATH/stencil-vector-openmp 10 200 $s $r
                #    done
                #done
                ;;
            icc)
                # Host
                echo "OPENMPFLAG=-qopenmp" >> common/make.defs
                make -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-innerloop-openmp stencil-vector-openmp transpose-vector-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-innerloop-openmp             10 1024 1024
                $PRK_TARGET_PATH/stencil-vector-openmp            10 1000
                $PRK_TARGET_PATH/transpose-vector-openmp          10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 6 7 8 9 ; do
                        $PRK_TARGET_PATH/stencil-vector-openmp 10 200 $s $r
                    done
                done
                # Offload - not supported on MacOS
                if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    echo "OFFLOADFLAG=-qopenmp -qopenmp-offload=host" >> common/make.defs
                    make -C $PRK_TARGET_PATH target
                    $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                    $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                    #echo "Test stencil code generator"
                    for s in star grid ; do
                        for r in 1 2 3 4 5 6 7 8 9 ; do
                            $PRK_TARGET_PATH/stencil-openmp-target 10 200 $s $r
                        done
                    done
                fi
                ;;
            *)
                echo "Figure out your OpenMP flags..."
                ;;
        esac

        # C++11 with TBB
        # Skip Clang because older Clang from Linux chokes on max_align_t (https://travis-ci.org/jeffhammond/PRK/jobs/243395307)
        if [ "${CC}" = "gcc" ] || [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            TBBROOT=${TRAVIS_ROOT}/tbb
            case "$os" in
                Linux)
                    ${CC} --version
                    echo "TBBFLAG=-I${TBBROOT}/include -L${TBBROOT}/lib/intel64/gcc4.7 -ltbb" >> common/make.defs
                    export LD_LIBRARY_PATH=${TBBROOT}/lib/intel64/gcc4.7:${LD_LIBRARY_PATH}
                    ;;
                Darwin)
                    echo "TBBFLAG=-I${TBBROOT}/include -L${TBBROOT}/lib -ltbb" >> common/make.defs
                    export LD_LIBRARY_PATH=${TBBROOT}/lib:${LD_LIBRARY_PATH}
                    ;;
            esac
            # Only build transpose because stencil is wrong in at least one way (https://travis-ci.org/jeffhammond/PRK/jobs/243395309)
            make -C $PRK_TARGET_PATH stencil-vector-tbb transpose-vector-tbb
            #$PRK_TARGET_PATH/p2p-vector-tbb     10 1024 1024 64 64
            $PRK_TARGET_PATH/stencil-vector-tbb     10 1000
            $PRK_TARGET_PATH/transpose-vector-tbb   10 1024 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 6 7 8 9 ; do
                    $PRK_TARGET_PATH/stencil-vector-tbb 10 200 32 $s $r
                done
            done
        fi

        # C++11 with OpenCL
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            echo "OPENCLFLAG=-framework OpenCL" >> common/make.defs
            make -C $PRK_TARGET_PATH opencl
            # must run programs in same directory as OpenCL source files...
            cd $PRK_TARGET_PATH
            ./stencil-opencl     10 1000
            ./transpose-opencl   10 1024 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 6 7 8 9 ; do
                    ./stencil-opencl 10 200 $s $r
                done
            done
            cd ..
        fi

        # C++11 with Cilk
        if [ "${CC}" = "gcc" ] ; then
            echo "CILKFLAG=-fcilkplus" >> common/make.defs
            make -C $PRK_TARGET_PATH stencil-vector-cilk transpose-vector-cilk
            $PRK_TARGET_PATH/stencil-vector-cilk     10 1000
            $PRK_TARGET_PATH/transpose-vector-cilk   10 1024 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 6 7 8 9 ; do
                    $PRK_TARGET_PATH/stencil-vector-cilk 10 200 $s $r
                done
            done
        fi

        # C++11 with Kokkos, RAJA
        case "$CC" in
            gcc)
                # Kokkos and Raja are built with OpenMP support with GCC
                export EXTRAFLAG="-fopenmp -ldl"
                ;;
            clang)
                # Kokkos is built with Pthread support with Clang
                export EXTRAFLAG="-lpthread -ldl"
                ;;
        esac
        # RAJA
        echo "RAJAFLAG=-DUSE_RAJA -I${TRAVIS_ROOT}/raja/include -L${TRAVIS_ROOT}/raja/lib -lRAJA ${EXTRAFLAG}" >> common/make.defs
        make -C $PRK_TARGET_PATH stencil-vector-raja transpose-vector-raja
        $PRK_TARGET_PATH/stencil-vector-raja     10 1000
        # RAJA variant 11 should be the best
        $PRK_TARGET_PATH/transpose-vector-raja   10 1024 11
        # test all the RAJA variants with a smaller problem
        for v in 1 2 3 4 5 6 7 10 11 12 13 14 15 ; do
            $PRK_TARGET_PATH/transpose-vector-raja   10 200 $v
        done
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil-vector-raja 10 200 $s $r
            done
        done
        # Kokkos
        echo "KOKKOSFLAG=-DUSE_KOKKOS -I${TRAVIS_ROOT}/kokkos/include -L${TRAVIS_ROOT}/kokkos/lib -lkokkos ${EXTRAFLAG}" >> common/make.defs
        make -C $PRK_TARGET_PATH stencil-kokkos transpose-kokkos
        $PRK_TARGET_PATH/stencil-kokkos     10 1000
        $PRK_TARGET_PATH/transpose-kokkos   10 1024 32
        for s in star grid ; do
            for r in 1 2 3 4 5 6 7 8 9 ; do
                $PRK_TARGET_PATH/stencil-kokkos 10 200 $s $r
            done
        done
        ;;
    allfortran)
        echo "Fortran"
        export PRK_TARGET_PATH=FORTRAN
        case "$CC" in
            gcc)
                for major in "-9" "-8" "-7" "-6" "-5" "-4" "-3" "-2" "-1" "" ; do
                    if [ -f "`which gfortran$major`" ]; then
                        export PRK_FC="gfortran$major"
                        echo "Found GCC Fortran: $PRK_FC"
                        break
                    fi
                done
                if [ "x$PRK_FC" = "x" ] ; then
                    echo "No Fortran compiler found!"
                    exit 9
                fi
                export PRK_FC="$PRK_FC -std=f2008 -cpp"
                echo "FC=$PRK_FC" >> common/make.defs
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
                    # Homebrew installs a symlink in /usr/local/bin
                    export PRK_CAFC=caf
                elif [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    export PRK_CAFC=$TRAVIS_ROOT/opencoarrays/bin/caf
                fi
                echo "CAFC=$PRK_CAFC -std=f2008 -cpp" >> common/make.defs
                echo "COARRAYFLAG=-fcoarray=single" >> common/make.defs
                ;;
            clang)
                echo "LLVM Fortran is not supported."
                exit 9
                echo "FC=flang" >> common/make.defs
                ;;
            icc)
                # -heap-arrays prevents SEGV in transpose-pretty (?)
                export PRK_FC="ifort -fpp -std08 -heap-arrays"
                echo "FC=$PRK_FC" >> common/make.defs
                echo "OPENMPFLAG=-qopenmp" >> common/make.defs
                echo "OFFLOADFLAG=-qopenmp-offload=host" >> common/make.defs
                echo "COARRAYFLAG=-coarray" >> common/make.defs
                ;;
        esac

        # Serial
        make -C ${PRK_TARGET_PATH} p2p p2p-innerloop stencil transpose
        $PRK_TARGET_PATH/p2p               10 1024 1024
        $PRK_TARGET_PATH/p2p-innerloop     10 1024
        $PRK_TARGET_PATH/stencil           10 1000
        $PRK_TARGET_PATH/transpose         10 1024 1
        $PRK_TARGET_PATH/transpose         10 1024 32

        # Pretty
        make -C ${PRK_TARGET_PATH} stencil-pretty transpose-pretty
        #$PRK_TARGET_PATH/p2p-pretty          10 1024 1024
        # pretty versions do not support tiling...
        $PRK_TARGET_PATH/stencil-pretty      10 1000
        $PRK_TARGET_PATH/transpose-pretty    10 1024

        # OpenMP host
        make -C ${PRK_TARGET_PATH} p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp
        export OMP_NUM_THREADS=2
        $PRK_TARGET_PATH/p2p-tasks-openmp     10 1024 1024
        $PRK_TARGET_PATH/p2p-innerloop-openmp 10 1024
        #$PRK_TARGET_PATH/p2p-openmp-doacross  10 1024 1024 # most compilers do not support doacross yet
        $PRK_TARGET_PATH/stencil-openmp       10 1000
        $PRK_TARGET_PATH/transpose-openmp     10 1024 1
        $PRK_TARGET_PATH/transpose-openmp     10 1024 32

        # Intel Mac does not support OpenMP target or coarrays
        if [ "${CC}" = "gcc" ] || [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
            # OpenMP target
            make -C ${PRK_TARGET_PATH} stencil-openmp-target transpose-openmp-target
            export OMP_NUM_THREADS=2
            #$PRK_TARGET_PATH/p2p-openmp-target           10 1024 1024 # most compilers do not support doacross yet
            $PRK_TARGET_PATH/stencil-openmp-target       10 1000
            $PRK_TARGET_PATH/transpose-openmp-target     10 1024 1
            $PRK_TARGET_PATH/transpose-openmp-target     10 1024 32

            # Fortran coarrays
            make -C ${PRK_TARGET_PATH} coarray
            export PRK_MPI_PROCS=4
            if [ "${CC}" = "gcc" ] ; then
                if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
                    # Homebrew installs a symlink in /usr/local/bin
                    export PRK_LAUNCHER=cafrun
                    # OpenCoarrays uses Open-MPI on Mac thanks to Homebrew
                    # see https://github.com/open-mpi/ompi/issues/2956
                    export TMPDIR=/tmp
                elif [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    export PRK_LAUNCHER=$TRAVIS_ROOT/opencoarrays/bin/cafrun
                fi
                $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/p2p-coarray       10 1024 1024
                $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/stencil-coarray   10 1000
                $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/transpose-coarray 10 1024 1
                $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/transpose-coarray 10 1024 32
            elif [ "${CC}" = "icc" ] ; then
                export FOR_COARRAY_NUM_IMAGES=$PRK_MPI_PROCS
                $PRK_TARGET_PATH/Synch_p2p/p2p-coarray       10 1024 1024
                $PRK_TARGET_PATH/Stencil/stencil-coarray     10 1000
                $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 1
                $PRK_TARGET_PATH/Transpose/transpose-coarray 10 1024 32
            fi
        fi
        ;;
    allopenmp)
        echo "OpenMP"
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
            CLANG_VERSION=3.9
            brew install llvm@$CLANG_VERSION || brew upgrade llvm@$CLANG_VERSION
            echo "CC=/usr/local/opt/llvm@${CLANG_VERSION}/bin/clang-${CLANG_VERSION} -std=c99" >> common/make.defs
            echo "OPENMPFLAG=-fopenmp" \
                            " -L/usr/local/opt/llvm@$CLANG_VERSION/lib -lomp" \
                            " /usr/local/opt/llvm@$CLANG_VERSION/lib/libomp.dylib" \
                            " -Wl,-rpath -Wl,/usr/local/opt/llvm@$CLANG_VERSION/lib" >> common/make.defs
            export LD_RUN_PATH=/usr/local/opt/llvm@$CLANG_VERSION/lib:$LD_RUN_PATH
            export LD_LIBRARY_PATH=/usr/local/opt/llvm@$CLANG_VERSION/lib:$LD_LIBRARY_PATH
            export DYLD_LIBRARY_PATH=/usr/local/opt/llvm@$CLANG_VERSION/lib:$DYLD_LIBRARY_PATH
        else
            echo "CC=$CC -std=c99" >> common/make.defs
            echo "OPENMPFLAG=-fopenmp" >> common/make.defs
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
        $PRK_TARGET_PATH/PIC/pic                  $OMP_NUM_THREADS 10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_TARGET_PATH/PIC/pic                  $OMP_NUM_THREADS 10 1000 1000000 0 1 SINUSOIDAL
        $PRK_TARGET_PATH/PIC/pic                  $OMP_NUM_THREADS 10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_TARGET_PATH/PIC/pic                  $OMP_NUM_THREADS 10 1000 1000000 1 0 PATCH 0 200 100 200
        # random is broken right now it seems
        #$PRK_TARGET_PATH/Random/random $OMP_NUM_THREADS 10 16384 32
        ;;
    allmpi)
        echo "All MPI"
        if [ -f ~/use-intel-compilers ] ; then
            # use Intel MPI
            export PRK_MPICC="mpiicc -std=c99"
            export PRK_LAUNCHER=mpirun
        else # Clang or GCC
            export PRK_MPICC="$MPI_ROOT/bin/mpicc -std=c99"
            export PRK_LAUNCHER=$MPI_ROOT/bin/mpirun
        fi
        # Inline the Homebrew OpenMP stuff here so versions do not diverge.
        # Note that -cc= likely only works with MPICH.
        if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
            GCC_VERSION=6
            brew install gcc@$GCC_VERSION || brew upgrade gcc@$GCC_VERSION
            export PRK_MPICC="${PRK_MPICC} -cc=/usr/local/opt/gcc@${GCC_VERSION}/bin/gcc-${GCC_VERSION}"
        fi
        #if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "clang" ] ; then
        #    CLANG_VERSION=3.9
        #    brew install llvm@$CLANG_VERSION || brew upgrade llvm@$CLANG_VERSION
        #    export PRK_MPICC="${PRK_MPICC} -cc=/usr/local/opt/llvm@${CLANG_VERSION}/bin/clang-${CLANG_VERSION}"
        #fi
        #if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "${CC}" = "clang" ] ; then
        #    # According to http://openmp.llvm.org/, we need version 3.8 or later to get OpenMP.
        #    for version in "-5" "-4" "-3.9" "-3.8" "" ; do
        #      if [ -f "`which ${CC}${version}`" ]; then
        #          export PRK_CC="${CC}${version}"
        #          echo "Found C: $PRK_CC"
        #          break
        #      fi
        #    done
        #    if [ "x$PRK_CC" = "x" ] ; then
        #        export PRK_CC="${CC}"
        #    fi
        #    ${PRK_CC} -v
        #    export PRK_MPICC="${PRK_MPICC} -cc=${PRK_CC}"
        #fi
        echo "MPICC=$PRK_MPICC" >> common/make.defs
        echo "OPENMPFLAG=-fopenmp" >> common/make.defs


        echo "MPI-1"
        make allmpi1
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
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 FINE_GRAIN 2
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 HIGH_WATER
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 NO_TALK

        # MPI+OpenMP is just too much of a pain with Clang right now.
        if [ "${CC}" = "gcc" ] ; then
            echo "MPI+OpenMP"
            make allmpiomp
            export PRK_TARGET_PATH=MPIOPENMP
            export PRK_MPI_PROCS=2
            export OMP_NUM_THREADS=2
            $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       $OMP_NUM_THREADS 10 1024 1024
            $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     $OMP_NUM_THREADS 10 1000
            $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose $OMP_NUM_THREADS 10 1024 32
            $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Nstream/nstream     $OMP_NUM_THREADS 10 16777216 32
        fi

        echo "MPI-RMA"
        make allmpirma
        export PRK_TARGET_PATH=MPIRMA
        export PRK_MPI_PROCS=4
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -n $PRK_MPI_PROCS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32

        echo "MPI+MPI"
        make allmpishm
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
                make $PRK_TARGET PRK_FLAGS="-Wc,-O3"
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
                export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-linux-x86_64
                export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=$TRAVIS_ROOT/charm/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        make $PRK_TARGET PRK_FLAGS=-O3
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
                export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-linux-x86_64
                export CHARM_ROOT=$TRAVIS_ROOT/charm/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=$TRAVIS_ROOT/charm/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        make $PRK_TARGET PRK_FLAGS="-O3 -std=gnu99"
        export PRK_TARGET_PATH=AMPI
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        export PRK_LAUNCHER_ARGS="+p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS +isomalloc_sync ++local"
        export PRK_LOAD_BALANCER_ARGS="+balancer RefineLB"
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_p2p/p2p       $PRK_LAUNCHER_ARGS 10 1024 1024
        $PRK_LAUNCHER $PRK_TARGET_PATH/Stencil/stencil     $PRK_LAUNCHER_ARGS 10 1000
        $PRK_LAUNCHER $PRK_TARGET_PATH/Transpose/transpose $PRK_LAUNCHER_ARGS 10 1024 32
        $PRK_LAUNCHER $PRK_TARGET_PATH/Reduce/reduce       $PRK_LAUNCHER_ARGS 10 16777216
        $PRK_LAUNCHER $PRK_TARGET_PATH/Nstream/nstream     $PRK_LAUNCHER_ARGS 10 16777216 32
        $PRK_LAUNCHER $PRK_TARGET_PATH/Sparse/sparse       $PRK_LAUNCHER_ARGS 10 10 5
        $PRK_LAUNCHER $PRK_TARGET_PATH/DGEMM/dgemm         $PRK_LAUNCHER_ARGS 10 1024 32 1
        $PRK_LAUNCHER $PRK_TARGET_PATH/Random/random       $PRK_LAUNCHER_ARGS 32 20
        $PRK_LAUNCHER $PRK_TARGET_PATH/Synch_global/global $PRK_LAUNCHER_ARGS 10 16384
        $PRK_LAUNCHER $PRK_TARGET_PATH/PIC/pic             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_LAUNCHER $PRK_TARGET_PATH/PIC/pic             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 1000000 0 1 SINUSOIDAL
        $PRK_LAUNCHER $PRK_TARGET_PATH/PIC/pic             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_LAUNCHER $PRK_TARGET_PATH/PIC/pic             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 1000000 1 0 PATCH 0 200 100 200
        $PRK_LAUNCHER $PRK_TARGET_PATH/AMR/amr             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 100 2 2 1 5 FINE_GRAIN
        $PRK_LAUNCHER $PRK_TARGET_PATH/AMR/amr             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 100 2 2 1 5 HIGH_WATER
        $PRK_LAUNCHER $PRK_TARGET_PATH/AMR/amr             $PRK_LAUNCHER_ARGS $PRK_LOAD_BALANCER_ARGS 10 1000 100 2 2 1 5 NO_TALK
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
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Reduce/reduce       10 16777216
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32 1
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Random/random       32 20
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/Synch_global/global 10 16384
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 0 1 SINUSOIDAL
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_LAUNCHER -np $PRK_MPI_PROCS -nfg $PRK_FGMPI_THREADS $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 PATCH 0 200 100 200
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
