set -e
set -x

os=`uname`

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install cmake
        ;;

    Linux)
        echo "Linux"
        # from source
        #wget -q https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz
        #tar -C $TRAVIS_ROOT -xzf cmake-3.4.1.tar.gz
        #cd ~/cmake-3.4.1
        #mkdir build && cd build
        #cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$TRAVIS_ROOT/cmake
        #make -j4 && make install
        # from binary
        cd $TRAVIS_ROOT
        wget -q https://cmake.org/files/v3.4/cmake-3.4.1-Linux-x86_64.sh
        chmod +x cmake-3.4.1-Linux-x86_64.sh # just in case
        ./cmake-3.4.1-Linux-x86_64.sh --prefix=$TRAVIS_ROOT/cmake --skip-license --exclude-subdir
        ;;
esac
