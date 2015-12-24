set -e
set -x

PRK_TARGET="$1"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        make $PRK_TARGET
        ;;

    allopenmp)
        echo "OpenMP"
        make $PRK_TARGET
        ;;
esac
