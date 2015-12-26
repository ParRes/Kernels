set -e
set -x

TRAVIS_ROOT="$1"
PRK_TARGET="$2"

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
        sh ./travis/install-mpi.sh $TRAVIS_ROOT $MPI_LIBRARY
        ;;
    allshmem)
        echo "SHMEM"
        sh ./travis/install-sandia-openshmem.sh $TRAVIS_ROOT
        ;;
    allupc)
        echo "UPC"
        sh ./travis/install-intrepid-upc.sh $TRAVIS_ROOT
        ;;
    allcharm++)
        echo "Charm++"
        sh ./travis/install-charm++.sh $TRAVIS_ROOT charm++
        ;;
    allampi)
        echo "Adaptive MPI (AMPI)"
        sh ./travis/install-charm++.sh $TRAVIS_ROOT AMPI
        ;;
    allfgmpi)
        echo "Fine-Grain MPI (FG-MPI)"
        sh ./travis/install-fgmpi.sh $TRAVIS_ROOT
        ;;
    allgrappa)
        echo "Grappa"
        sh ./travis/install-grappa.sh $TRAVIS_ROOT
        ;;
esac
