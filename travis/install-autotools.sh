#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Darwin)
        brew update
        brew info autoconf automake libtool || true
        brew install autoconf || brew upgrade autoconf || true
        brew install automake || brew upgrade automake || true
        brew install libtool  || brew upgrade libtool  || true
        #brew list autoconf automake libtool || true
        which glibtool
        which glibtoolize
        #glibtool --version
        ln -s `which glibtool` ${TRAVIS_ROOT}/bin/libtool
        ln -s `which glibtoolize` ${TRAVIS_ROOT}/bin/libtoolize
        ;;
    Linux)
        MAKE_JNUM=2
        M4_VERSION=1.4.17
        LIBTOOL_VERSION=2.4.6
        AUTOCONF_VERSION=2.69
        AUTOMAKE_VERSION=1.15

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
          ./configure CC=cc --prefix=${TRAVIS_ROOT} && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 1
            exit
          fi
        fi

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
          ./configure CC=cc --prefix=${TRAVIS_ROOT} M4=${TRAVIS_ROOT}/bin/m4 && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 2
            exit
          fi
        fi

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
          ./configure CC=cc --prefix=${TRAVIS_ROOT} M4=${TRAVIS_ROOT}/bin/m4 && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 3
            exit
          fi
        fi

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
          ./configure CC=cc --prefix=${TRAVIS_ROOT} M4=${TRAVIS_ROOT}/bin/m4 && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 4
            exit
          fi
        fi
        ;;
esac

