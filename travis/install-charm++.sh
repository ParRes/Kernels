set -e
set -x

os=`uname`

# charm++ or AMPI
RUNTIME="$1"

case "$os" in
    Darwin)
        echo "Mac"
        cd ~
        wget -q http://charm.cs.illinois.edu/distrib/charm-6.6.1.tar.gz
        tar -xzf charm-6.6.1.tar.gz
        cd charm
        ./build $RUNTIME netlrts-darwin-x86_64 --with-production -j4
        ;;

    Linux)
        echo "Linux"
        cd ~
        wget -q http://charm.cs.illinois.edu/distrib/charm-6.6.1.tar.gz
        tar -xzf charm-6.6.1.tar.gz
        cd charm
        #./build charm++ netlrts-linux-x86_64 smp --with-production
        ./build $RUNTIME netlrts-linux-x86_64 --with-production -j4
        ;;
esac
