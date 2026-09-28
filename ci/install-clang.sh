#!/bin/sh

set -e
set -x

CI_ROOT="$1"

if [ "${CC}" = "clang" ] || [ "${CXX}" = "clang++" ] ; then
    os=`uname`
    case "$os" in
        Darwin)
            echo "Mac"
            brew install llvm || brew upgrade llvm || true
            #brew install libomp || brew upgrade libomp || true
            ;;
        Linux)
            echo "Linux"
            # Install whatever clang/libomp-dev versions the distro's apt
            # repos currently default to, rather than looping over a fixed
            # list of old numbered packages (clang-11 down to clang-7):
            # those aged out of Ubuntu's repos as the CI runner image was
            # upgraded over time, so the loop always failed silently and
            # left no clang/libomp installed at all. Unversioned package
            # names track whatever Ubuntu ships by default, so this keeps
            # working across future runner image upgrades too.
            sudo apt-get install -y clang libomp-dev
            clang -v
        ;;
    esac
fi
