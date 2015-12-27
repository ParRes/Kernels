set -e
set -x

TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/fgmpi" ]; then
    wget -q http://www.cs.ubc.ca/~humaira/code/fgmpi-2.0.tar.gz
    tar -C $TRAVIS_ROOT -xzf fgmpi-2.0.tar.gz
    cd $TRAVIS_ROOT/fgmpi-2.0
    mkdir build && cd build
    # Clang defaults to C99, which chokes on "Set_PROC_NULL"
    ../configure --disable-fortran CFLAGS="-std=gnu89" --prefix=$TRAVIS_ROOT/fgmpi
    make -j4 && make install
else
    echo "FG-MPI installed..."
    find $TRAVIS_ROOT/fgmpi -name mpiexec
    find $TRAVIS_ROOT/fgmpi -name mpicc
    mpicc -show
fi
