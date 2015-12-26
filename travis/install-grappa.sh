set -e
set -x

os=`uname`

# TODO: Make compiler and MPI configurable...

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install cmake ruby boost mpich
        ;;

    Linux)
        echo "Linux"
        sudo apt-get update -q
        sudo apt-get install -y cmake ruby clang
        # Grappa requires MPI-3, for which we can depend on MPICH
        sudo apt-get install -y gfortran libcr0 default-jdk
        wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.2b3/mpich_3.2b3-1ubuntu_amd64.deb
        sudo dpkg -i ./mpich_3.2b3-1ubuntu_amd64.deb
        # I hate CMake so so much...
        wget -q https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz
        tar -C $HOME -xzf cmake-3.4.1.tar.gz
        cd ~/cmake-3.4.1
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME
        make -j4 && make install
        ;;
esac

cd ~
git clone https://github.com/uwsampa/grappa.git
cd grappa
./configure --prefix=$HOME/grappa
cd build/Make+Release
make -j4 && make install
