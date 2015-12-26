set -e
set -x

# install OFI/libfabric
cd ~
git clone --depth 10 https://github.com/ofiwg/libfabric.git libfabric
cd libfabric
./autogen.sh
./configure --prefix=$HOME
make && make install
export FI_LOG_LEVEL=error

# install Hydra
cd ~
wget http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
tar xvzf hydra-3.2.tar.gz
cd hydra-3.2
./configure --prefix=$HOME
make && make install

# install Sandia OpenSHMEM
cd ~
git clone --depth 10 https://github.com/regrant/sandia-shmem.git
cd sandia-shmem
./autogen.sh
./configure --with-ofi=$HOME --with-ofi-libdir=$HOME/lib --disable-fortran --enable-error-checking --enable-remote-virtual-addressing --enable-pmi-simple --prefix=$HOME
make && make install
