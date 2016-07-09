#!/bin/bash

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

# R = required version
# P = provided version
# X.Y.Z version tuples
check_version()
{
    set +x
    if [ ! -z "$1" ] ; then
        RX=`echo $1 | cut -d '.' -f 1`
        RY=`echo $1 | cut -d '.' -f 2`
        RZ=`echo $1 | cut -d '.' -f 3`
    else
        echo 1
    fi
    if [ ! -z "$2" ] ; then
        PX=`echo $2 | cut -d '.' -f 1`
        PY=`echo $2 | cut -d '.' -f 2`
        PZ=`echo $2 | cut -d '.' -f 3`
    else
        echo 2
    fi
    if [ -z "$RX" ] ; then
        RX=0
    fi
    if [ -z "$RY" ] ; then
        RY=0
    fi
    if [ -z "$RZ" ] ; then
        RZ=0
    fi
    if [ -z "$PX" ] ; then
        PX=0
    fi
    if [ -z "$PY" ] ; then
        PY=0
    fi
    if [ -z "$PZ" ] ; then
        PZ=0
    fi

    RI=$((1000*1000*${RX}+1000*${RY}+${RZ}))
    PI=$((1000*1000*${PX}+1000*${PY}+${PZ}))

    if [ ${PI} -ge ${RI} ] ; then
        echo 0
    else
        echo 3
    fi
    set -x
}

case "$os" in
    Darwin)
        set +e # do not fail on error
        brew update
        brew info autoconf automake libtool
        brew install autoconf automake libtool
        brew upgrade autoconf automake libtool
        which glibtool
        which glibtoolize
        glibtool --version
        ln -s `which glibtool` ${TRAVIS_ROOT}/bin/libtool
        ln -s `which glibtoolize` ${TRAVIS_ROOT}/bin/libtoolize
        set -e # restore fail on error
        ;;
    Linux)
        MAKE_JNUM=2
        M4_VERSION=1.4.17
        LIBTOOL_VERSION=2.4.6
        AUTOCONF_VERSION=2.69
        AUTOMAKE_VERSION=1.15

        LOCAL_VERSION=$(m4 --version | awk 'NR==1{print $4}') # Mac is $3, Linux is $4 >:-(
        if [ 0 -ne $(check_version ${M4_VERSION} ${LOCAL_VERSION}) ]; then
          cd ${TRAVIS_ROOT}
          TOOL=m4
          TDIR=${TOOL}-${M4_VERSION}
          FILE=${TDIR}.tar.gz
          BIN=${TRAVIS_ROOT}/bin/${TOOL}
          if [ -f ${FILE} ] ; then
            echo ${FILE} already exists! Using existing copy.
          else
            wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
          fi
          if [ -d ${TDIR} ] ; then
            echo ${TDIR} already exists! Using existing copy.
          else
            echo Unpacking ${FILE}
            tar -xzf ${FILE}
          fi
          if [ -f ${BIN} ] ; then
            echo ${BIN} already exists! Skipping build.
          else
            cd ${TRAVIS_ROOT}/${TDIR}
            ./configure --prefix=${TRAVIS_ROOT} && make -j ${MAKE_JNUM} && make install
            if [ "x$?" != "x0" ] ; then
              echo FAILURE 1
              exit
            fi
          fi
        fi

        LOCAL_VERSION=$(libtool --version | awk 'NR==1{print $4}')
        if [ 0 -ne $(check_version ${LIBTOOL_VERSION} ${LOCAL_VERSION}) ]; then
          cd ${TRAVIS_ROOT}
          TOOL=libtool
          TDIR=${TOOL}-${LIBTOOL_VERSION}
          FILE=${TDIR}.tar.gz
          BIN=${TRAVIS_ROOT}/bin/${TOOL}
          if [ ! -f ${FILE} ] ; then
            wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
          else
            echo ${FILE} already exists! Using existing copy.
          fi
          if [ ! -d ${TDIR} ] ; then
            echo Unpacking ${FILE}
            tar -xzf ${FILE}
          else
            echo ${TDIR} already exists! Using existing copy.
          fi
          if [ -f ${BIN} ] ; then
            echo ${BIN} already exists! Skipping build.
          else
            cd ${TRAVIS_ROOT}/${TDIR}
            ./configure --prefix=${TRAVIS_ROOT} && make -j ${MAKE_JNUM} && make install
            if [ "x$?" != "x0" ] ; then
              echo FAILURE 2
              exit
            fi
          fi
        fi

        LOCAL_VERSION=$(autoconf --version | awk 'NR==1{print $4}')
        if [ 0 -ne $(check_version ${AUTOCONF_VERSION} ${LOCAL_VERSION}) ]; then
          cd ${TRAVIS_ROOT}
          TOOL=autoconf
          TDIR=${TOOL}-${AUTOCONF_VERSION}
          FILE=${TDIR}.tar.gz
          BIN=${TRAVIS_ROOT}/bin/${TOOL}
          if [ ! -f ${FILE} ] ; then
            wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
          else
            echo ${FILE} already exists! Using existing copy.
          fi
          if [ ! -d ${TDIR} ] ; then
            echo Unpacking ${FILE}
            tar -xzf ${FILE}
          else
            echo ${TDIR} already exists! Using existing copy.
          fi
          if [ -f ${BIN} ] ; then
            echo ${BIN} already exists! Skipping build.
          else
            cd ${TRAVIS_ROOT}/${TDIR}
            ./configure --prefix=${TRAVIS_ROOT} && make -j ${MAKE_JNUM} && make install
            if [ "x$?" != "x0" ] ; then
              echo FAILURE 3
              exit
            fi
          fi
        fi

        LOCAL_VERSION=$(automake --version | awk 'NR==1{print $4}')
        if [ 0 -ne $(check_version ${AUTOMAKE_VERSION} ${LOCAL_VERSION}) ]; then
          cd ${TRAVIS_ROOT}
          TOOL=automake
          TDIR=${TOOL}-${AUTOMAKE_VERSION}
          FILE=${TDIR}.tar.gz
          BIN=${TRAVIS_ROOT}/bin/${TOOL}
          if [ ! -f ${FILE} ] ; then
            wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
          else
            echo ${FILE} already exists! Using existing copy.
          fi
          if [ ! -d ${TDIR} ] ; then
            echo Unpacking ${FILE}
            tar -xzf ${FILE}
          else
            echo ${TDIR} already exists! Using existing copy.
          fi
          if [ -f ${BIN} ] ; then
            echo ${BIN} already exists! Skipping build.
          else
            cd ${TRAVIS_ROOT}/${TDIR}
            ./configure --prefix=${TRAVIS_ROOT} && make -j ${MAKE_JNUM} && make install
            if [ "x$?" != "x0" ] ; then
              echo FAILURE 4
              exit
            fi
          fi
        fi
        ;;
esac

