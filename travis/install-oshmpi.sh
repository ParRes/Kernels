set -e
set -x

# install OSHMPI
git clone https://github.com/jeffhammond/oshmpi.git
cd oshmpi
./autogen.sh
./configure
make && sudo make install
