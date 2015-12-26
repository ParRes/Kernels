set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        wget -q http://www.cs.ubc.ca/~humaira/code/fgmpi-2.0.tar.gz
        tar -C $TRAVIS_ROOT -xzvf fgmpi-2.0.tar.gz
        cd $TRAVIS_ROOT/fgmpi-2.0
        mkdir build && cd build
        ../configure --prefix=$TRAVIS_ROOT/fgmpi
        ;;

    Linux)
        echo "Linux"
        sudo apt-get update -q
        sudo apt-get install -y gfortran libcr0 default-jdk
        wget -q http://www.cs.ubc.ca/~humaira/code/fgmpi_2.0-1_amd64.deb
        sudo dpkg -i ./fgmpi_2.0-1_amd64.deb
        ;;
esac
