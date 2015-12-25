set -e
set -x

os=`uname`

# charm++ or AMPI
RUNTIME="$1"

case "$os" in
    Darwin)
        echo "Mac"
        wget -q http://charm.cs.illinois.edu/distrib/charm-6.6.1.tar.gz
        tar -xzf charm-6.6.1.tar.gz
        cd charm-6.6.1
        ./build $RUNTIME netlrts-darwin-x86_64 --with-production
        ;;

    Linux)
        echo "Linux"
        wget -q http://charm.cs.illinois.edu/distrib/charm-6.6.1.tar.gz
        tar -xzvf charm-6.6.1.tar.gz
        cd charm-6.6.1
        #./build charm++ netlrts-linux-x86_64 smp --with-production
        ./build $RUNTIME netlrts-linux-x86_64 --with-production
        # create soft link so build+run of PRK does not need to know version
        cd ~ && ln -s charm-6.6.1 charm
        ;;
esac
