set -e
set -x

TRAVIS_ROOT="$1"

# install OSHMPI
git clone https://github.com/jeffhammond/oshmpi.git
cd oshmpi
./autogen.sh
./configure --prefix=$TRAVIS_ROOT
make && sudo make install
