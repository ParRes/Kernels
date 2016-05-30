set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

if [ ! -d "$TRAVIS_ROOT/opencoarrays" ]; then
    case "$os" in
        Darwin)
            echo "Mac"
            brew update
            brew install gcc6
            brew install mpich
            brew install cmake
            ;;

        Linux)
            echo "Linux"
            ;;
    esac

    cd $TRAVIS_ROOT
    git clone --depth 10 https://github.com/sourceryinstitute/opencoarrays.git opencoarrays-source
    cd opencoarrays-source
    which yes
    yes | ./install.sh -j 2 -i $TRAVIS_ROOT/opencoarrays
else
    echo "OpenCoarrays installed..."
    #find $TRAVIS_ROOT -name opencoarrays.mk
fi
