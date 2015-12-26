set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

# TODO: Make compiler and MPI configurable...

if [ ! -d "$TRAVIS_ROOT/grappa" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            brew install ruby boost
            ;;

        Linux)
            echo "Linux"
            ;;
    esac

    cd $TRAVIS_ROOT
    git clone https://github.com/uwsampa/grappa.git grappa-source
    cd grappa-source
    ./configure --prefix=$TRAVIS_ROOT/grappa
    cd build/Make+Release
    make -j4 && make install
else
    echo "Grappa installed..."
    find $TRAVIS_ROOT -name grappa.mk
fi
