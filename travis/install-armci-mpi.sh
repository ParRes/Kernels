#!/usr/bin/env bash

TRAVIS_ROOT=$1
ARMCI_MPI_DIR=$TRAVIS_ROOT/armci-mpi
ARMCI_MPI_TARBALL=$2

if [ ! -z "${MPICC}" ] ; then
    echo "Found MPICC=${MPICC} in your environment.  Using that."
    ARMCIMPICC=${MPICC}
elif [ -d /opt/cray ] ; then
    echo "You appear to be on a Cray machine."
    echo "If you don't want to use Cray MPI wrappers, "
    echo "run again with MPICC=\$YOUR_MPI ./install-armci-mpi"
    ARMCIMPICC=cc
elif [ ! -z "${MPI_DIR}" ] ; then
    echo "Found MPI_DIR=${MPI_DIR} in your environment."
    echo "Using MPICC=${MPI_DIR}/bin/mpicc."
    ARMCIMPICC=${MPI_DIR}/bin/mpicc
else
    echo "Defaulting to the mpicc in your path."
    ARMCIMPICC=mpicc
fi

if [ -d ${ARMCI_MPI_DIR} ] ; then
  # Assume that it is the right one, created previously by this script.
  echo "Existing ARMCI-MPI source directory found - updating it"
  cd ${ARMCI_MPI_DIR} && git checkout master && git pull >& git.log
  echo "Step 1 of 4 succeeded."
elif [ -x "$(command -v git)" ] ; then
  #git clone -b mpi3rma --depth 10 http://git.mpich.org/armci-mpi.git ${ARMCI_MPI_DIR} >& git.log
  git clone -b master --depth 10 https://github.com/jeffhammond/armci-mpi.git ${ARMCI_MPI_DIR} >& git.log
  stat=$?
  if [ $stat -ne 0 ] ; then
    echo "git clone failed"
    exit 1
  else
    echo "Step 1 of 4 succeeded."
  fi
else
  if test "x${ARMCI_MPI_TARBALL}" = x ; then
    echo "You have not provided a download location for the ARMCI-MPI tarball!"
    exit 1
  else
    wget ${ARMCI_MPI_TARBALL}
    if [ $stat -ne 0 ] ; then
      echo "wget failed"
      exit 1
    else
      echo "Step 1 of 4 succeeded."
    fi
  fi
fi

cd ${ARMCI_MPI_DIR}

if ! [ -f ${ARMCI_MPI_DIR}/configure ] ; then
  ./autogen.sh >& autogen.log
  stat=$?
  if [ $stat -ne 0 ] ; then
    echo "Autotools failed, likely because the versions your system has are too old."
    echo "See https://github.com/jeffhammond/HPCInfo/wiki/Autotools."
    exit 1
  else
    echo "Step 2 of 4 succeeded."
  fi
else
  echo "Step 2 of 4 succeeded."
fi

if ! [ -d ${ARMCI_MPI_DIR}/build ] ; then
  mkdir ${ARMCI_MPI_DIR}/build
fi
cd ${ARMCI_MPI_DIR}/build

${ARMCI_MPI_DIR}/configure CC=$ARMCIMPICC --prefix=${ARMCI_MPI_DIR} --enable-g >& c.log
stat=$?
if [ $stat -ne 0 ] ; then
  echo "configure failed."
  echo "Please email config.log and system details to armci-discuss@lists.mpich.org."
  exit 1
else
  echo "Step 3 of 4 succeeded."
fi

make install >& m.log
stat=$?
if [ $stat -ne 0 ] ; then
  echo "configure failed."
  echo "Please email config.log and system details to armci-discuss@lists.mpich.org."
  exit 1
else
  echo "Step 4 of 4 succeeded."
fi

echo "Congratulations! ARMCI-MPI was built successfully."
