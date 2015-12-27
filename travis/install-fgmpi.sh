set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/fgmpi" ]; then
    case "$os" in
        Darwin)
            #echo "Mac"
            ;;

        Linux)
            #echo "Linux"
            # cannot do this without sudo...
            #sudo apt-get update -q
            #sudo apt-get install -y gfortran libcr0 default-jdk
            #wget -q http://www.cs.ubc.ca/~humaira/code/fgmpi_2.0-1_amd64.deb
            #sudo dpkg -i ./fgmpi_2.0-1_amd64.deb
            # for container builds
            ;;
    esac
    wget -q http://www.cs.ubc.ca/~humaira/code/fgmpi-2.0.tar.gz
    tar -C $TRAVIS_ROOT -xzf fgmpi-2.0.tar.gz
    cd $TRAVIS_ROOT/fgmpi-2.0
    mkdir build && cd build
    ../configure --disable-fortran --prefix=$TRAVIS_ROOT/fgmpi
    make -j4 && make install
else
    echo "FG-MPI installed..."
    find $TRAVIS_ROOT/fgmpi -name mpiexec
    find $TRAVIS_ROOT/fgmpi -name mpicc
    mpicc -show
fi
