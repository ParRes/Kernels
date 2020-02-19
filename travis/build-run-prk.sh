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
    FreeBSD)
        MAKE=gmake
        ;;
    *)
        MAKE=make
        ;;
esac

case "$os" in
    Darwin)
        # Homebrew should put MPI here...
        export MPI_ROOT=/usr/local
        ;;
    Linux)
        export MPI_ROOT=${TRAVIS_ROOT}
        ;;
esac

echo "PRKVERSION=\"'2.16'\"" > common/make.defs

case "$PRK_TARGET" in
    allpython)
        echo "Python"
        # workaround for trusty since cannot find numpy when using /opt/python/2.7.13/bin/python
        if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
            export PATH=/usr/bin:$PATH
        fi
        which python3 || which python || true
        python3 --version || python --version || true
        export PRK_TARGET_PATH=PYTHON
        export PRK_PYTHON=python
        # Native
        $PRK_PYTHON $PRK_TARGET_PATH/p2p.py             10 100 100
        $PRK_PYTHON $PRK_TARGET_PATH/stencil.py         10 100
        $PRK_PYTHON $PRK_TARGET_PATH/transpose.py       10 100
        $PRK_PYTHON $PRK_TARGET_PATH/nstream.py         10 100000
        $PRK_PYTHON $PRK_TARGET_PATH/sparse.py          10 10 5
        $PRK_PYTHON $PRK_TARGET_PATH/dgemm.py           10 100
        # Numpy
        $PRK_PYTHON $PRK_TARGET_PATH/p2p-numpy.py       10 1024 1024
        $PRK_PYTHON $PRK_TARGET_PATH/stencil-numpy.py   10 1000
        $PRK_PYTHON $PRK_TARGET_PATH/transpose-numpy.py 10 1024
        $PRK_PYTHON $PRK_TARGET_PATH/nstream-numpy.py   10 16777216
        $PRK_PYTHON $PRK_TARGET_PATH/sparse-numpy.py    10 10 5
        $PRK_PYTHON $PRK_TARGET_PATH/dgemm-numpy.py     10 400
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
                export JULIA_PATH=${TRAVIS_ROOT}/julia/bin/
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
        ${MAKE} $PRK_TARGET
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
        export PRK_TARGET_PATH=RUST
        cd $TRAVIS_HOME/$PRK_TARGET_PATH/p2p       && cargo run 10 100 100
        cd $TRAVIS_HOME/$PRK_TARGET_PATH/stencil   && cargo run 10 100
        cd $TRAVIS_HOME/$PRK_TARGET_PATH/transpose && cargo run 10 100
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
        echo "CC=${PRK_CC} -std=c11 -DPRK_USE_GETTIMEOFDAY" >> common/make.defs
        echo "EXTRA_CLIBS=-lm -lpthread" >> common/make.defs

        # C11 without external parallelism
        ${MAKE} -C $PRK_TARGET_PATH nstream p2p stencil transpose p2p-hyperplane
        $PRK_TARGET_PATH/nstream         10 16777216 32
        $PRK_TARGET_PATH/p2p             10 1024 1024
        $PRK_TARGET_PATH/p2p             10 1024 1024 100 100
        $PRK_TARGET_PATH/p2p-hyperplane  10 1024
        $PRK_TARGET_PATH/p2p-hyperplane  10 1024 32
        $PRK_TARGET_PATH/stencil         10 1000
        $PRK_TARGET_PATH/transpose       10 1024 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil 10 200 $s $r
            done
        done

        # C11 with POSIX or C11 thread parallelism - test POSIX here, C11 at the end.
        ${MAKE} -C $PRK_TARGET_PATH transpose-thread
        $PRK_TARGET_PATH/transpose-thread   10 1024 512

        # C11 with OpenMP
        export OMP_NUM_THREADS=2
        case "$CC" in
            clang*)
                echo "Skipping Clang since OpenMP support probably missing"
                ;;
            g*)
                # Host
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH nstream-openmp p2p-tasks-openmp p2p-hyperplane-openmp stencil-openmp transpose-openmp
                $PRK_TARGET_PATH/nstream-openmp           10 16777216 32
                $PRK_TARGET_PATH/p2p-tasks-openmp         10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-hyperplane-openmp    10 1024
                $PRK_TARGET_PATH/p2p-hyperplane-openmp    10 1024 32
                $PRK_TARGET_PATH/stencil-openmp           10 1000
                $PRK_TARGET_PATH/transpose-openmp         10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 $s $r
                    done
                done
                # Offload
                echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH target
                $PRK_TARGET_PATH/stencil-target     10 1000
                $PRK_TARGET_PATH/transpose-target   10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-target 10 200 $s $r
                    done
                done
                ;;
            ic*)
                # Host
                echo "OPENMPFLAG=-qopenmp" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH nstream-openmp p2p-tasks-openmp p2p-hyperplane-openmp stencil-openmp transpose-openmp
                $PRK_TARGET_PATH/nstream-openmp           10 16777216 32
                $PRK_TARGET_PATH/p2p-tasks-openmp         10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-hyperplane-openmp    10 1024
                $PRK_TARGET_PATH/p2p-hyperplane-openmp    10 1024 32
                $PRK_TARGET_PATH/stencil-openmp           10 1000
                $PRK_TARGET_PATH/transpose-openmp         10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 $s $r
                    done
                done
                # Offload - not supported on MacOS
                if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    echo "OFFLOADFLAG=-qopenmp -qopenmp-offload=host" >> common/make.defs
                    ${MAKE} -C $PRK_TARGET_PATH target
                    $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                    $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                    #echo "Test stencil code generator"
                    for s in star grid ; do
                        for r in 1 2 3 4 5 ; do
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
        #if [ "${CC}" = "gcc" ] ; then
        #    echo "CILKFLAG=-fcilkplus" >> common/make.defs
        #    ${MAKE} -C $PRK_TARGET_PATH stencil-cilk transpose-cilk
        #    $PRK_TARGET_PATH/stencil-cilk     10 1000
        #    $PRK_TARGET_PATH/transpose-cilk   10 1024 32
        #    #echo "Test stencil code generator"
        #    for s in star grid ; do
        #        for r in 1 2 3 4 5 ; do
        #            $PRK_TARGET_PATH/stencil-cilk 10 200 $s $r
        #        done
        #    done
        #fi

        # Use MUSL for GCC+Linux only
        if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "$CC" = "gcc" ] ; then
            ${MAKE} -C $PRK_TARGET_PATH clean
            ./travis/install-musl.sh ${TRAVIS_ROOT} ${PRK_CC}
            echo "PRKVERSION=\"'2.16'\"" > common/make.defs
            echo "CC=${TRAVIS_ROOT}/musl/bin/musl-gcc -static -std=c11 -DUSE_C11_THREADS" >> common/make.defs
            echo "EXTRA_CLIBS=-lm -lpthread" >> common/make.defs
            ${MAKE} -C $PRK_TARGET_PATH transpose-thread
            $PRK_TARGET_PATH/transpose-thread   10 1024 512
        fi

        ;;
    allcxx)
        echo "C++11"
        export PRK_TARGET_PATH=Cxx11
        case $CXX in
            g++)
                if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "x$PRK_CXX" = "x" ] ; then
                  for version in "9" "8" "7" "6" "5" "" ; do
                    if [ -f "`which /usr/local/opt/gcc@${version}/bin/g++-${version}`" ]; then
                        export PRK_CXX="`which /usr/local/opt/gcc@${version}/bin/g++-${version}`"
                        echo "Found C++: $PRK_CXX"
                        break
                    fi
                  done
                fi
                if [ "x$PRK_CXX" = "x" ] ; then
                  for major in "-9" "-8" "-7" "-6" "-5" "" ; do
                    if [ -f "`which ${CXX}${major}`" ]; then
                        export PRK_CXX="${CXX}${major}"
                        echo "Found C++: $PRK_CXX"
                        break
                    fi
                  done
                fi
                if [ "x$PRK_CXX" = "x" ] ; then
                    export PRK_CXX="${CXX}"
                fi
                ;;
            clang++)
                # Homebrew does not always place the best/latest Clang/LLVM in the default path
                if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "x$PRK_CXX" = "x" ] ; then
                  for version in "" "@6" "@5" "@4" ; do
                    if [ -f "`which /usr/local/opt/llvm${version}/bin/clang++`" ]; then
                        export PRK_CXX="`which /usr/local/opt/llvm${version}/bin/clang++`"
                        echo "Found C++: $PRK_CXX"
                        break
                    fi
                  done
                fi
                if [ "x$PRK_CXX" = "x" ] ; then
                  for version in "-6" "-5" "-4.1" "-4" "-4.0" "-3.9" "-3.8" "-3.7" "-3.6" "" ; do
                    if [ -f "`which ${CXX}${version}`" ]; then
                        export PRK_CXX="${CXX}${version}"
                        echo "Found C++: $PRK_CXX"
                        break
                    fi
                  done
                fi
                if [ "x$PRK_CXX" = "x" ] ; then
                    export PRK_CXX="${CXX}"
                fi
                ;;
        esac
        ${PRK_CXX} -v
        # Need to increment this for PSTL
        # The pthread flag is supported by GCC and Clang at least
        echo "CXX=${PRK_CXX} -std=c++14 -pthread" >> common/make.defs

        # C++11 without external parallelism
        ${MAKE} -C $PRK_TARGET_PATH transpose-valarray nstream-valarray
        $PRK_TARGET_PATH/transpose-valarray 10 1024 32
        $PRK_TARGET_PATH/nstream-valarray   10 16777216 32

        # C++11 without external parallelism
        ${MAKE} -C $PRK_TARGET_PATH p2p-vector p2p-hyperplane-vector stencil-vector transpose-vector nstream-vector \
                                 dgemm-vector sparse-vector
        $PRK_TARGET_PATH/p2p-vector              10 1024 1024
        $PRK_TARGET_PATH/p2p-vector              10 1024 1024 100 100
        $PRK_TARGET_PATH/p2p-hyperplane-vector   10 1024
        $PRK_TARGET_PATH/p2p-hyperplane-vector   10 1024 64
        $PRK_TARGET_PATH/stencil-vector          10 1000
        $PRK_TARGET_PATH/transpose-vector        10 1024 32
        $PRK_TARGET_PATH/nstream-vector          10 16777216 32
        $PRK_TARGET_PATH/dgemm-vector            10 400 400 # untiled
        $PRK_TARGET_PATH/dgemm-vector            10 400 32
        $PRK_TARGET_PATH/sparse-vector           10 10 5
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil-vector 10 200 20 $s $r
            done
        done

        # C++11 with CBLAS
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            echo "CBLASFLAG=-DACCELERATE -framework Accelerate" >> common/make.defs
            ${MAKE} -C $PRK_TARGET_PATH transpose-cblas dgemm-cblas
            $PRK_TARGET_PATH/transpose-cblas    10 1024
            $PRK_TARGET_PATH/dgemm-cblas        10 400
        fi

        # C++11 native parallelism
        ${MAKE} -C $PRK_TARGET_PATH transpose-vector-thread transpose-vector-async
        $PRK_TARGET_PATH/transpose-vector-thread 10 1024 512 32
        $PRK_TARGET_PATH/transpose-vector-async  10 1024 512 32

        # C++11 with OpenMP
        export OMP_NUM_THREADS=2
        case "$CC" in
            gcc)
                # Host
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-hyperplane-openmp stencil-openmp \
                                         transpose-openmp nstream-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-hyperplane-openmp     10 1024
                $PRK_TARGET_PATH/p2p-hyperplane-openmp     10 1024 64
                $PRK_TARGET_PATH/stencil-openmp            10 1000
                $PRK_TARGET_PATH/transpose-openmp          10 1024 32
                $PRK_TARGET_PATH/nstream-openmp            10 16777216 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 20 $s $r
                    done
                done
                # Offload
                echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH target
                $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 20 $s $r
                    done
                done
                # ORNL-ACC
                echo "ORNLACCFLAG=-fopenacc" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH p2p-hyperplane-vector-ornlacc
                $PRK_TARGET_PATH/p2p-hyperplane-vector-ornlacc     10 1024
                $PRK_TARGET_PATH/p2p-hyperplane-vector-ornlacc     10 1024 64
                ;;
            clang)
                if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
                    # Host
                    echo "OPENMPFLAG=-fopenmp" >> common/make.defs
                    ${MAKE} -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-hyperplane-openmp stencil-openmp \
                                             transpose-openmp nstream-openmp
                    $PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                    $PRK_TARGET_PATH/p2p-hyperplane-openmp     10 1024
                    $PRK_TARGET_PATH/p2p-hyperplane-openmp     10 1024 64
                    $PRK_TARGET_PATH/stencil-openmp            10 1000
                    $PRK_TARGET_PATH/transpose-openmp          10 1024 32
                    $PRK_TARGET_PATH/nstream-openmp            10 16777216 32
                    #echo "Test stencil code generator"
                    for s in star grid ; do
                        for r in 1 2 3 4 5 ; do
                            $PRK_TARGET_PATH/stencil-openmp 10 200 20 $s $r
                        done
                    done
                    # Offload
                    #echo "OFFLOADFLAG=-foffload=\"-O3 -v\"" >> common/make.defs
                    #${MAKE} -C $PRK_TARGET_PATH target
                    #$PRK_TARGET_PATH/stencil-openmp-target     10 1000
                    #$PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                    ##echo "Test stencil code generator"
                    #for s in star grid ; do
                    #    for r in 1 2 3 4 5 ; do
                    #        $PRK_TARGET_PATH/stencil-openmp 10 200 20 $s $r
                    #    done
                    #done
                else
                    echo "Skipping Clang since OpenMP support probably missing"
                fi
                ;;
            icc)
                # Host
                echo "OPENMPFLAG=-qopenmp" >> common/make.defs
                ${MAKE} -C $PRK_TARGET_PATH p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp \
                                         transpose-openmp nstream-openmp
                $PRK_TARGET_PATH/p2p-tasks-openmp                 10 1024 1024 100 100
                $PRK_TARGET_PATH/p2p-innerloop-openmp             10 1024 1024
                $PRK_TARGET_PATH/stencil-openmp            10 1000
                $PRK_TARGET_PATH/transpose-openmp          10 1024 32
                $PRK_TARGET_PATH/nstream-openmp            10 16777216 32
                #echo "Test stencil code generator"
                for s in star grid ; do
                    for r in 1 2 3 4 5 ; do
                        $PRK_TARGET_PATH/stencil-openmp 10 200 20 $s $r
                    done
                done
                # Offload - not supported on MacOS
                if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    echo "OFFLOADFLAG=-qopenmp -qopenmp-offload=host" >> common/make.defs
                    ${MAKE} -C $PRK_TARGET_PATH target
                    $PRK_TARGET_PATH/stencil-openmp-target     10 1000
                    $PRK_TARGET_PATH/transpose-openmp-target   10 1024 32
                    #echo "Test stencil code generator"
                    for s in star grid ; do
                        for r in 1 2 3 4 5 ; do
                            $PRK_TARGET_PATH/stencil-openmp-target 10 200 20 $s $r
                        done
                    done
                fi
                ;;
            *)
                echo "Figure out your OpenMP flags..."
                ;;
        esac

        # Boost.Compute runs after OpenCL, and only available in Travis with MacOS.
        case "$os" in
            FreeBSD)
                echo "BOOSTFLAG=-DUSE_BOOST -I/usr/local/include" >> common/make.defs
                echo "RANGEFLAG=-DUSE_BOOST_IRANGE -I/usr/local/include" >> common/make.defs
                ;;
            *)
                echo "BOOSTFLAG=-DUSE_BOOST" >> common/make.defs
                echo "RANGEFLAG=-DUSE_RANGES_TS -I${TRAVIS_ROOT}/range-v3/include" >> common/make.defs
                ;;
        esac

        # C++11 with rangefor and Boost.Ranges
        ${MAKE} -C $PRK_TARGET_PATH rangefor
        $PRK_TARGET_PATH/stencil-vector-rangefor     10 1000
        $PRK_TARGET_PATH/transpose-vector-rangefor   10 1024 32
        $PRK_TARGET_PATH/nstream-vector-rangefor     10 16777216 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil-vector-rangefor 10 200 20 $s $r
            done
        done

        # C++11 with TBB
        # Skip Clang because older Clang from Linux chokes on max_align_t (https://travis-ci.org/jeffhammond/PRK/jobs/243395307)
        if [ "${CC}" = "gcc" ] || [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            TBBROOT=${TRAVIS_ROOT}/tbb
            case "$os" in
                Linux)
                    ${CC} --version
                    export TBBFLAG="-I${TBBROOT}/include -L${TBBROOT}/lib/intel64/gcc4.7 -ltbb"
                    echo "TBBFLAG=-DUSE_TBB ${TBBFLAG}" >> common/make.defs
                    export LD_LIBRARY_PATH=${TBBROOT}/lib/intel64/gcc4.7:${LD_LIBRARY_PATH}
                    ;;
                Darwin)
                    export TBBFLAG="-I${TBBROOT}/include -L${TBBROOT}/lib -ltbb"
                    echo "TBBFLAG=-DUSE_TBB ${TBBFLAG}" >> common/make.defs
                    export LD_LIBRARY_PATH=${TBBROOT}/lib:${LD_LIBRARY_PATH}
                    ;;
            esac
            ${MAKE} -C $PRK_TARGET_PATH p2p-innerloop-vector-tbb p2p-hyperplane-vector-tbb p2p-tasks-tbb stencil-vector-tbb transpose-vector-tbb nstream-vector-tbb
            $PRK_TARGET_PATH/p2p-innerloop-vector-tbb     10 1024
            $PRK_TARGET_PATH/p2p-hyperplane-vector-tbb    10 1024 1
            $PRK_TARGET_PATH/p2p-hyperplane-vector-tbb    10 1024 32
            $PRK_TARGET_PATH/p2p-tasks-tbb                10 1024 1024 32 32
            $PRK_TARGET_PATH/stencil-vector-tbb           10 1000
            $PRK_TARGET_PATH/transpose-vector-tbb         10 1024 32
            $PRK_TARGET_PATH/nstream-vector-tbb           10 16777216 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 ; do
                    $PRK_TARGET_PATH/stencil-vector-tbb 10 200 20 $s $r
                done
            done
        fi

        # C++11 with STL
        ${MAKE} -C $PRK_TARGET_PATH p2p-hyperplane-vector-stl stencil-vector-stl transpose-vector-stl nstream-vector-stl
        $PRK_TARGET_PATH/p2p-hyperplane-vector-stl    10 1024 1
        $PRK_TARGET_PATH/p2p-hyperplane-vector-stl    10 1024 32
        $PRK_TARGET_PATH/stencil-vector-stl           10 1000
        $PRK_TARGET_PATH/transpose-vector-stl         10 1024 32
        $PRK_TARGET_PATH/nstream-vector-stl           10 16777216 32
        #echo "Test stencil code generator"
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil-vector-stl 10 200 20 $s $r
            done
        done

        # C++17 Parallel STL
        # Skip Clang+Linux because we skip TBB there - see above.
        if [ "${CC}" = "gcc" ] || [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            if [ "${CC}" = "clang" ] ; then
                # omp.h not found with clang-3.9 - just work around instead of fixing.
                echo "PSTLFLAG=-DUSE_PSTL ${TBBFLAG} -DUSE_INTEL_PSTL -I${TRAVIS_ROOT}/pstl/include ${RANGEFLAG}" >> common/make.defs
            else
                echo "PSTLFLAG=-DUSE_PSTL -fopenmp ${TBBFLAG} -DUSE_INTEL_PSTL -I${TRAVIS_ROOT}/pstl/include ${RANGEFLAG}" >> common/make.defs
            fi
            ${MAKE} -C $PRK_TARGET_PATH p2p-hyperplane-vector-pstl stencil-vector-pstl transpose-vector-pstl nstream-vector-pstl
            $PRK_TARGET_PATH/p2p-hyperplane-vector-pstl    10 1024 1
            $PRK_TARGET_PATH/p2p-hyperplane-vector-pstl    10 1024 32
            $PRK_TARGET_PATH/stencil-vector-pstl           10 1000
            $PRK_TARGET_PATH/transpose-vector-pstl         10 1024 32
            $PRK_TARGET_PATH/nstream-vector-pstl           10 16777216 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 ; do
                    $PRK_TARGET_PATH/stencil-vector-pstl 10 200 20 $s $r
                done
            done
        fi

        # C++11 with OpenCL
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            echo "OPENCLFLAG=-framework OpenCL" >> common/make.defs
            ${MAKE} -C $PRK_TARGET_PATH opencl
            # must run programs in same directory as OpenCL source files...
            cd $PRK_TARGET_PATH
            ./stencil-opencl     10 1000
            ./transpose-opencl   10 1024 32
            ./nstream-opencl     10 16777216 32
            #echo "Test stencil code generator"
            for s in star grid ; do
                for r in 1 2 3 4 5 ; do
                    ./stencil-opencl 10 200 20 $s $r
                done
            done
            cd ..
        fi

        # Boost.Compute moved after OpenCL to reuse those flags...

        # C++11 with Boost.Compute
        # Only test Mac because:
        # (1) We only test OpenCL on MacOS in Travis.
        # (2) Boost.Compute is not available from APT.
        # If we ever address 1, we need to enable the Boost.Compute install for Linux.
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            ${MAKE} -C $PRK_TARGET_PATH nstream-vector-boost-compute
            $PRK_TARGET_PATH/nstream-vector-boost-compute     10 16777216 32
        fi

        # C++11 with Kokkos, RAJA
        case "$CC" in
            gcc)
                # Kokkos and Raja are built with OpenMP support with GCC
                echo "RAJAFLAG=-DUSE_RAJA -I${TRAVIS_ROOT}/raja/include -L${TRAVIS_ROOT}/raja/lib -lRAJA ${TBBFLAG} -fopenmp" >> common/make.defs
                echo "KOKKOSFLAG=-DUSE_KOKKOS -I${TRAVIS_ROOT}/kokkos/include -L${TRAVIS_ROOT}/kokkos/lib -lkokkos -DPRK_KOKKOS_BACKEND=OpenMP -fopenmp -ldl" >> common/make.defs
                ;;
            clang)
                # RAJA can use TBB with Clang
                echo "RAJAFLAG=-DUSE_RAJA -I${TRAVIS_ROOT}/raja/include -L${TRAVIS_ROOT}/raja/lib -lRAJA ${TBBFLAG}" >> common/make.defs
                # Kokkos is built with Pthread support with Clang
                echo "KOKKOSFLAG=-DUSE_KOKKOS -I${TRAVIS_ROOT}/kokkos/include -L${TRAVIS_ROOT}/kokkos/lib -lkokkos -DPRK_KOKKOS_BACKEND=Threads -lpthread -ldl" >> common/make.defs
                ;;
        esac
        # RAJA
        if [ 0 = 1 ] ; then
        ${MAKE} -C $PRK_TARGET_PATH p2p-vector-raja stencil-vector-raja transpose-vector-raja nstream-vector-raja \
                                 p2p-raja stencil-raja transpose-raja nstream-raja
        # New (Views)
        $PRK_TARGET_PATH/p2p-raja                10 1024 1024
        $PRK_TARGET_PATH/stencil-raja            10 1000
        $PRK_TARGET_PATH/transpose-raja          10 1024
        $PRK_TARGET_PATH/nstream-raja            10 16777216 32
        # Old (STL)
        $PRK_TARGET_PATH/p2p-vector-raja         10 1024 1024
        $PRK_TARGET_PATH/stencil-vector-raja     10 1000
        $PRK_TARGET_PATH/transpose-vector-raja   10 1024
        for f in seq omp tbb ; do
         for s in y n ; do
          for t in y n ; do
           for n in y n ; do
            for p in no ij ji ; do
             $PRK_TARGET_PATH/transpose-vector-raja 4 200 nested=$n for=$f simd=$s tiled=$t permute=$p
            done
           done
          done
         done
        done
        $PRK_TARGET_PATH/nstream-vector-raja     10 16777216 32
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil-vector-raja 10 200 20 $s $r
                $PRK_TARGET_PATH/stencil-raja        10 200 20 $s $r
            done
        done
        fi
        # Kokkos
        ${MAKE} -C $PRK_TARGET_PATH stencil-kokkos transpose-kokkos nstream-kokkos
        $PRK_TARGET_PATH/stencil-kokkos     10 1000
        $PRK_TARGET_PATH/transpose-kokkos   10 1024 32
        $PRK_TARGET_PATH/nstream-kokkos     10 16777216 32
        for s in star grid ; do
            for r in 1 2 3 4 5 ; do
                $PRK_TARGET_PATH/stencil-kokkos 10 200 20 $s $r
            done
        done

        # C++ w/ OCCA
        # OCCA sets  -Wl,-rpath=${OCCA_LIB}, which chokes Mac's ld.
        #if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
        #    echo "OCCADIR=${TRAVIS_ROOT}/occa" >> common/make.defs
        #    export OCCA_CXX=${PRK_CXX}
        #    ${MAKE} -C $PRK_TARGET_PATH transpose-occa nstream-occa
        #    $PRK_TARGET_PATH/transpose-occa   10 1024 32
        #    $PRK_TARGET_PATH/nstream-occa     10 16777216 32
        #fi

        # C++ w/ SYCL
        # triSYCL requires Boost.  We are having Boost issues with Travis Linux builds.
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            SYCLDIR=${TRAVIS_ROOT}/triSYCL
            if [ "${CC}" = "clang" ] ; then
                # SYCL will compile without OpenMP
                echo "SYCLCXX=${PRK_CXX} -pthread -std=c++1z" >> common/make.defs
            else
                echo "SYCLCXX=${PRK_CXX} -fopenmp -std=c++1z" >> common/make.defs
            fi
            echo "SYCLFLAG=-DUSE_SYCL -I${SYCLDIR}/include" >> common/make.defs
            ${MAKE} -C $PRK_TARGET_PATH p2p-hyperplane-sycl stencil-sycl transpose-sycl nstream-sycl
            #$PRK_TARGET_PATH/p2p-hyperplane-sycl 10 50 1 # 100 takes too long :-o
            $PRK_TARGET_PATH/stencil-sycl        10 1000
            $PRK_TARGET_PATH/transpose-sycl      10 1024 32
            $PRK_TARGET_PATH/nstream-sycl        10 16777216 32
            #echo "Test stencil code generator"
            for s in star ; do # grid ; do # grid not supported yet
                for r in 1 2 3 4 5 ; do
                    $PRK_TARGET_PATH/stencil-sycl 10 200 20 $s $r
                done
            done
        fi

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
                    export PRK_CAFC=/usr/local/bin/caf
                elif [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    export PRK_CAFC=${TRAVIS_ROOT}/opencoarrays/bin/caf
                fi
                echo "CAFC=$PRK_CAFC -std=f2008 -cpp" >> common/make.defs
                echo "COARRAYFLAG=-fcoarray=single" >> common/make.defs
                ;;
            clang)
                case "$os" in
                    FreeBSD)
                        echo "FC=flang -Mpreprocess -Mfreeform -I/usr/local/flang/include -lexecinfo" >> common/make.defs
                        ;;
                    *)
                        # untested
                        echo "FC=flang -Mpreprocess -Mfreeform" >> common/make.defs
                        ;;
                esac
                echo "OPENMPFLAG=-fopenmp" >> common/make.defs
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
        ${MAKE} -C ${PRK_TARGET_PATH} p2p p2p-innerloop stencil transpose nstream dgemm
        $PRK_TARGET_PATH/p2p               10 1024 1024
        $PRK_TARGET_PATH/p2p-innerloop     10 1024
        $PRK_TARGET_PATH/stencil           10 1000
        $PRK_TARGET_PATH/transpose         10 1024 1
        $PRK_TARGET_PATH/transpose         10 1024 32
        $PRK_TARGET_PATH/nstream           10 16777216
        $PRK_TARGET_PATH/dgemm             10 400 400 # untiled
        $PRK_TARGET_PATH/dgemm             10 400 32

        # Pretty
        ${MAKE} -C ${PRK_TARGET_PATH} stencil-pretty transpose-pretty nstream-pretty dgemm-pretty
        #$PRK_TARGET_PATH/p2p-pretty          10 1024 1024
        # pretty versions do not support tiling...
        $PRK_TARGET_PATH/stencil-pretty      10 1000
        $PRK_TARGET_PATH/transpose-pretty    10 1024
        $PRK_TARGET_PATH/nstream-pretty      10 16777216
        $PRK_TARGET_PATH/dgemm-pretty        10 400

        # OpenMP host
        ${MAKE} -C ${PRK_TARGET_PATH} p2p-tasks-openmp p2p-innerloop-openmp stencil-openmp transpose-openmp \
                                   nstream-openmp dgemm-openmp
        export OMP_NUM_THREADS=2
        $PRK_TARGET_PATH/p2p-tasks-openmp     10 1024 1024
        $PRK_TARGET_PATH/p2p-innerloop-openmp 10 1024
        #$PRK_TARGET_PATH/p2p-openmp-doacross  10 1024 1024 # most compilers do not support doacross yet
        $PRK_TARGET_PATH/stencil-openmp       10 1000
        $PRK_TARGET_PATH/transpose-openmp     10 1024 1
        $PRK_TARGET_PATH/transpose-openmp     10 1024 32
        $PRK_TARGET_PATH/nstream-openmp       10 16777216
        $PRK_TARGET_PATH/dgemm-openmp         10 400 400 # untiled
        $PRK_TARGET_PATH/dgemm-openmp         10 400 32

        # Intel Mac does not support OpenMP target or coarrays
        if [ "${CC}" = "gcc" ] || [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
            # OpenMP target
            ${MAKE} -C ${PRK_TARGET_PATH} stencil-openmp-target transpose-openmp-target nstream-openmp-target
            export OMP_NUM_THREADS=2
            #$PRK_TARGET_PATH/p2p-openmp-target           10 1024 1024 # most compilers do not support doacross yet
            $PRK_TARGET_PATH/stencil-openmp-target       10 1000
            $PRK_TARGET_PATH/transpose-openmp-target     10 1024 1
            $PRK_TARGET_PATH/transpose-openmp-target     10 1024 32
            $PRK_TARGET_PATH/nstream-openmp-target       10 16777216

            # Fortran coarrays
            ${MAKE} -C ${PRK_TARGET_PATH} coarray
            export PRK_MPI_PROCS=4
            if [ "${CC}" = "gcc" ] ; then
                if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
                    # Homebrew installs a symlink in /usr/local/bin
                    ls -l /usr/local/bin/cafrun || true
                    which cafrun || true
                    export PRK_LAUNCHER="/usr/local/bin/cafrun"
                    # OpenCoarrays uses Open-MPI on Mac thanks to Homebrew
                    # see https://github.com/open-mpi/ompi/issues/2956
                    export PRK_OVERSUBSCRIBE="--oversubscribe"
                    export TMPDIR=/tmp
                elif [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
                    export PRK_LAUNCHER=${TRAVIS_ROOT}/opencoarrays/bin/cafrun
                fi
                $PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-} $PRK_TARGET_PATH/p2p-coarray       10 1024 1024
                $PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-} $PRK_TARGET_PATH/stencil-coarray   10 1000
                $PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-} $PRK_TARGET_PATH/transpose-coarray 10 1024 1
                $PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-} $PRK_TARGET_PATH/transpose-coarray 10 1024 32
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
        ${MAKE} $PRK_TARGET
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
        # We use Open-MPI on Mac now...
        if [ "${TRAVIS_OS_NAME}" = "osx" ] ; then
            # see https://github.com/open-mpi/ompi/issues/2956
            export PRK_OVERSUBSCRIBE="--oversubscribe"
            export TMPDIR=/tmp
        fi

        # Inline the Homebrew OpenMP stuff here so versions do not diverge.
        # Note that -cc= likely only works with MPICH.
        #if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "${CC}" = "gcc" ] ; then
        #    GCC_VERSION=6
        #    brew upgrade gcc@$GCC_VERSION || brew install gcc@$GCC_VERSION
        #    export PRK_MPICC="${PRK_MPICC} -cc=/usr/local/opt/gcc@${GCC_VERSION}/bin/gcc-${GCC_VERSION}"
        #fi
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
        ${MAKE} allmpi1
        export PRK_TARGET_PATH=MPI1
        export PRK_MPI_PROCS=4
        export PRK_RUN="$PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-}"
        $PRK_RUN $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_RUN $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_RUN $PRK_TARGET_PATH/Transpose/transpose 10 1024 32
        $PRK_RUN $PRK_TARGET_PATH/Reduce/reduce       10 16777216
        $PRK_RUN $PRK_TARGET_PATH/Nstream/nstream     10 16777216 32
        $PRK_RUN $PRK_TARGET_PATH/Sparse/sparse       10 10 5
        $PRK_RUN $PRK_TARGET_PATH/DGEMM/dgemm         10 1024 32 1
        $PRK_RUN $PRK_TARGET_PATH/Random/random       32 20
        $PRK_RUN $PRK_TARGET_PATH/Synch_global/global 10 16384
        $PRK_RUN $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 2 GEOMETRIC 0.99
        $PRK_RUN $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 0 1 SINUSOIDAL
        $PRK_RUN $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 LINEAR 1.0 3.0
        $PRK_RUN $PRK_TARGET_PATH/PIC-static/pic      10 1000 1000000 1 0 PATCH 0 200 100 200
        $PRK_RUN $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 FINE_GRAIN 2
        $PRK_RUN $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 HIGH_WATER
        $PRK_RUN $PRK_TARGET_PATH/AMR/amr             10 1000 100 2 2 1 5 NO_TALK

        # MPI+OpenMP is just too much of a pain with Clang right now.
        if [ "${CC}" = "gcc" ] ; then
            echo "MPI+OpenMP"
            ${MAKE} allmpiomp
            export PRK_TARGET_PATH=MPIOPENMP
            export PRK_MPI_PROCS=2
            export OMP_NUM_THREADS=2
            export PRK_RUN="$PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-}"
            $PRK_RUN $PRK_TARGET_PATH/Synch_p2p/p2p       $OMP_NUM_THREADS 10 1024 1024
            $PRK_RUN $PRK_TARGET_PATH/Stencil/stencil     $OMP_NUM_THREADS 10 1000
            $PRK_RUN $PRK_TARGET_PATH/Transpose/transpose $OMP_NUM_THREADS 10 1024 32
            $PRK_RUN $PRK_TARGET_PATH/Nstream/nstream     $OMP_NUM_THREADS 10 16777216 32
        fi

        echo "MPI-RMA"
        ${MAKE} allmpirma
        export PRK_TARGET_PATH=MPIRMA
        export PRK_MPI_PROCS=4
        export PRK_RUN="$PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-}"
        $PRK_RUN $PRK_TARGET_PATH/Synch_p2p/p2p       10 1024 1024
        $PRK_RUN $PRK_TARGET_PATH/Stencil/stencil     10 1000
        $PRK_RUN $PRK_TARGET_PATH/Transpose/transpose 10 1024 32

        echo "MPI+MPI"
        ${MAKE} allmpishm
        export PRK_TARGET_PATH=MPISHM
        export PRK_MPI_PROCS=4
        export PRK_RUN="$PRK_LAUNCHER -n $PRK_MPI_PROCS ${PRK_OVERSUBSCRIBE:-}"
        export PRK_MPISHM_RANKS=$(($PRK_MPI_PROCS/2))
        $PRK_RUN $PRK_TARGET_PATH/Synch_p2p/p2p                         10 1024 1024
        $PRK_RUN $PRK_TARGET_PATH/Stencil/stencil     $PRK_MPISHM_RANKS 10 1000
        $PRK_RUN $PRK_TARGET_PATH/Transpose/transpose $PRK_MPISHM_RANKS 10 1024 32
        ;;
    allshmem)
        echo "SHMEM"
        # This should be fixed by rpath (https://github.com/regrant/sandia-shmem/issues/83)
        export LD_LIBRARY_PATH=${TRAVIS_ROOT}/sandia-openshmem/lib:${TRAVIS_ROOT}/libfabric/lib:$LD_LIBRARY_PATH
        export SHMEM_ROOT=${TRAVIS_ROOT}/sandia-openshmem
        echo "SHMEMTOP=$SHMEM_ROOT\nSHMEMCC=$SHMEM_ROOT/bin/oshcc" >> common/make.defs
        ${MAKE} $PRK_TARGET
        export PRK_TARGET_PATH=SHMEM
        export PRK_SHMEM_PROCS=4
        export OSHRUN_LAUNCHER=${TRAVIS_ROOT}/hydra/bin/mpirun
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
                        #export UPC_ROOT=${TRAVIS_ROOT}/gupc
                        # If installing deb file
                        export UPC_ROOT=${TRAVIS_ROOT}/gupc/usr/local/gupc
                        ;;
                    clang)
                        echo "Clang UPC is not supported."
                        exit 9
                        export UPC_ROOT=${TRAVIS_ROOT}/clupc
                        ;;
                esac
                echo "UPCC=$UPC_ROOT/bin/upc" >> common/make.defs
                export PRK_LAUNCHER=""
                export PRK_LAUNCHER_ARGS="-n $PRK_UPC_PROCS"
                ${MAKE} $PRK_TARGET
                ;;
            bupc)
                export UPC_ROOT=${TRAVIS_ROOT}/bupc-$CC
                echo "UPCC=$UPC_ROOT/bin/upcc" >> common/make.defs
                # -N $nodes -n UPC threads -c $cores_per_node
                # -localhost is only for UDP
                case "$GASNET_CONDUIT" in
                    udp)
                        export PRK_LAUNCHER="$UPC_ROOT/bin/upcrun -N 1 -n $PRK_UPC_PROCS -c $PRK_UPC_PROCS -localhost"
                        ;;
                    ofi)
                        export GASNET_SSH_SERVERS="localhost"
                        export LD_LIBRARY_PATH="${TRAVIS_ROOT}/libfabric/lib:$LD_LIBRARY_PATH"
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
                ${MAKE} $PRK_TARGET PRK_FLAGS="-Wc,-O3"
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
                export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-linux-x86_64
                export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=${TRAVIS_ROOT}/charm/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        ${MAKE} $PRK_TARGET PRK_FLAGS=-O3
        export PRK_TARGET_PATH=CHARM++
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
            export PRK_LAUNCHER_ARGS="+autoProvision +isomalloc_sync"
        else
            export PRK_LAUNCHER_ARGS="+p$PRK_CHARM_PROCS ++local"
        fi
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
                export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-darwin-x86_64-smp
                ;;
            Linux)
                #export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-linux-x86_64
                export CHARM_ROOT=${TRAVIS_ROOT}/charm/netlrts-linux-x86_64-smp
                #export CHARM_ROOT=${TRAVIS_ROOT}/charm/multicore-linux64
                ;;
        esac
        echo "CHARMTOP=$CHARM_ROOT" >> common/make.defs
        ${MAKE} $PRK_TARGET PRK_FLAGS="-O3 -std=gnu99"
        export PRK_TARGET_PATH=AMPI
        export PRK_CHARM_PROCS=4
        export PRK_LAUNCHER=$CHARM_ROOT/bin/charmrun
        if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
            export PRK_LAUNCHER_ARGS="+autoProvision +isomalloc_sync"
        else
            export PRK_LAUNCHER_ARGS="+p$PRK_CHARM_PROCS +vp$PRK_CHARM_PROCS +isomalloc_sync ++local"
        fi
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
        export FGMPI_ROOT=${TRAVIS_ROOT}/fgmpi
        echo "FGMPITOP=$FGMPI_ROOT\nFGMPICC=$FGMPI_ROOT/bin/mpicc -std=c99" >> common/make.defs
        ${MAKE} $PRK_TARGET
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
        #. ${TRAVIS_ROOT}/grappa/bin/settings.sh
        export GRAPPA_PREFIX=${TRAVIS_ROOT}/grappa
        export SCRIPT_PATH=${TRAVIS_ROOT}/grappa/bin
        ########################
        echo "GRAPPATOP=${TRAVIS_ROOT}/grappa" >> common/make.defs
        ${MAKE} $PRK_TARGET
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
        echo "LEGIONTOP=${TRAVIS_ROOT}/legion" > common/make.defs
        ${MAKE} $PRK_TARGET -k
        ;;
esac
