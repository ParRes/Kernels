set -e
set -x

os=`uname`

# TODO: Make compiler and MPI configurable...

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install ruby boost mpich
        ;;

    Linux)
        echo "Linux"
        sudo apt-get update -q
        sudo apt-get install -y ruby
        # Grappa requires MPI-3, for which we can depend on MPICH
        sudo apt-get install -y gfortran libcr0 default-jdk
        wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.2b3/mpich_3.2b3-1ubuntu_amd64.deb
        sudo dpkg -i ./mpich_3.2b3-1ubuntu_amd64.deb
        ;;
esac

# debug
gcc --version

git clone https://github.com/uwsampa/grappa.git grappa-source
cd grappa-source
./configure --prefix=$HOME/grappa
cd build/Make+Release
make -j4 && make install
