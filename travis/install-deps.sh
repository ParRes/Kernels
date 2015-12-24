set -e
set -x

PRK_TARGET="$1"

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        ;;

    allopenmp)
        echo "OpenMP"
        ;;
esac
