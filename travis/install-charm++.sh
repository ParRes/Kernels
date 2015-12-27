set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
# charm++ or AMPI
RUNTIME="$2"

if [ ! -d "$TRAVIS_ROOT/charm" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            cd $TRAVIS_ROOT
            wget --no-check-certificate -q https://charm.cs.illinois.edu/distrib/charm-6.7.0.tar.gz
            tar -xzf charm-6.7.0.tar.gz
            cd charm-6.7.0
            #./build $RUNTIME netlrts-darwin-x86_64 --with-production -j4
            ./build $RUNTIME netlrts-darwin-x86_64 smp --with-production -j4
            ;;

        Linux)
            echo "Linux"
            cd $TRAVIS_ROOT
            wget --no-check-certificate -q https://charm.cs.illinois.edu/distrib/charm-6.7.0.tar.gz
            tar -xzf charm-6.7.0.tar.gz
            cd charm-6.7.0
            # This fails with: The authenticity of host 'localhost (127.0.0.1)' can't be established.
            #./build $RUNTIME netlrts-linux-x86_64 --with-production -j4
            ./build $RUNTIME netlrts-linux-x86_64 smp --with-production
            #./build $RUNTIME multicore-linux64 --with-production
            ;;
    esac
else
    echo "Charm++ or AMPI already installed..."
    case "$RUNTIME" in
        AMPI)
            find $TRAVIS_ROOT/charm -name charmrun
            find $TRAVIS_ROOT/charm -name ampicc
            ;;
        charm++)
            find $TRAVIS_ROOT/charm -name charmrun
            find $TRAVIS_ROOT/charm -name charmc
            ;;
    esac
fi
