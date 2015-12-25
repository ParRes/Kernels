set -e
set -x

PRK_TARGET="$1"

# eventually make this runtime configurable
MPI_LIBRARY=mpich

case "$PRK_TARGET" in
    allserial)
        echo "Serial"
        ;;

    allopenmp)
        echo "OpenMP"
        ;;
    allmpi*)
        echo "Any MPI"
        sh ./travis/install-mpi.sh $MPI_LIBRARY
        ;;
    allshmem)
        echo "SHMEM"
        sh ./travis/install-sandia-openshmem.sh
        ;;
    allupc)
        echo "UPC"
        sh ./travis/install-intrepid-upc.sh
        ;;
esac
