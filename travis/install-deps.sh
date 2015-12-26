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
        echo "Any normal MPI"
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
    allcharm++)
        echo "Charm++"
        sh ./travis/install-charm++.sh charm++
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        sh ./travis/install-charm++.sh AMPI
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        sh ./travis/install-fgmpi.sh
        ;;
    allgrappa)
        echo "Grappa"
        sh ./travis/install-gcc.sh
        sh ./travis/install-grappa.sh
        ;;
esac
