set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

# TODO: Make compiler and MPI configurable...

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install ruby boost
        ;;

    Linux)
        echo "Linux"
        sudo apt-get update -q
        sudo apt-get install -y ruby
        ;;
esac

git clone https://github.com/uwsampa/grappa.git grappa-source
cd grappa-source
./configure --prefix=$TRAVIS_ROOT/grappa
cd build/Make+Release
make -j4 && make install
