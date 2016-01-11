set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
CLANG_VERSION="$2"

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        case "$CLANG_VERSION" in
            omp)
                brew install clang-omp
                brew test clang-omp
                ;;
            37)
                brew install llvm$CLANG_VERSION --with-clang --with-compiler-rt --with-libcxx --with-lld --without-assertions
                brew test llvm$CLANG_VERSION
                ;;
            *)
                echo "Unsupported version of Clang"
                echo "Travis will continue and use the system default"
                ;;
        esac
        ;;
    Linux)
        echo "Install Clang using APT on Linux"
        exit 85
    ;;
esac
