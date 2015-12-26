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
        sudo apt-get update -q
        # from source
        #wget -q https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz
        #tar -C $HOME -xzf cmake-3.4.1.tar.gz
        #cd ~/cmake-3.4.1
        #mkdir build && cd build
        #cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME
        #make -j4 && make install
        # from binary
        mkdir $HOME/cmake && cd $HOME/cmake
        wget -q https://cmake.org/files/v3.4/cmake-3.4.1-Linux-x86_64.sh
        chmod +x cmake-3.4.1-Linux-x86_64.sh # just in case
        ./cmake-3.4.1-Linux-x86_64.sh --prefix=$HOME/cmake --skip-license --exclude-subdir
        ;;
esac
