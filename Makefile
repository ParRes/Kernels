#Copyright (c) 2013, Intel Corporation
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#
# ******************************************************************

ifndef number_of_functions
  number_of_functions=40
endif

ifndef matrix_rank
  matrix_rank=5
endif

ifndef PRK_FLAGS
  PRK_FLAGS=-O3
endif

default: allserial allcxx allc1z allfortran allopenmp allmpi

help:
	@echo "Usage: \"make all\"          (re-)builds all targets"
	@echo "       \"make allserial\"    (re-)builds all serial targets"
	@echo "       \"make allcxx\"       (re-)builds all C++ targets"
	@echo "       \"make allc1z\"       (re-)builds all C1z targets"
	@echo "       \"make allrust\"      (re-)builds all Rust targets"
	@echo "       \"make allopenmp\"    (re-)builds all OpenMP targets"
	@echo "       \"make allmpi1\"      (re-)builds all conventional MPI targets"
	@echo "       \"make allfenix\"     (re-)builds all conventional MPI targets with Fenix fault tolerance"
	@echo "       \"make allfgmpi\"     (re-)builds all Fine-Grain MPI targets"
	@echo "       \"make allmpiopenmp\" (re-)builds all MPI + OpenMP targets"
	@echo "       \"make allmpiomp\"    (re-)builds all MPI + OpenMP targets"
	@echo "       \"make allmpishm\"    (re-)builds all MPI+MPI aka MPI+Shm targets"
	@echo "       \"make allmpirma\"    (re-)builds all MPI-3 RMA targets"
	@echo "       \"make allmpi\"       (re-)builds all MPI targets"
	@echo "       \"make allshmem\"     (re-)builds all SHMEM targets"
	@echo "       \"make allmpishm\"    (re-)builds all MPI-3 shared memory segments targets"
	@echo "       \"make allupc\"       (re-)builds all UPC targets"
	@echo "       \"make allpgas\"      (re-)builds all PGAS (UPC, SHMEM, MPI-3 RMA) targets"
	@echo "       \"make alldarwin\"    (re-)builds all of the above targets"
	@echo "       \"make allcharm++\"   (re-)builds all Charm++ targets"
	@echo "       \"make allampi\"      (re-)builds all Adaptive MPI targets"
	@echo "       \"make allgrappa\"    (re-)builds all Grappa targets"
	@echo "       \"make allfortran\"   (re-)builds all Fortran targets"
	@echo "       \"make alllegion\"    (re-)builds all Legion targets"
	@echo "       \"make allfreaks\"    (re-)builds the above four targets"
	@echo "       optionally, specify   \"matrix_rank=<n> number_of_functions=<m>\""
	@echo "       optionally, specify   \"default_opt_flags=<list of optimization flags>\""
	@echo "       \"make allshared\"    (re-)builds the shared-memory targets (C89, C1z, C++11, Fortran)"
	@echo "       \"make clean\"        removes all objects and executables"
	@echo "       \"make veryclean\"    removes some generated source files as well"

all: alldarwin allfreaks allshared
alldarwin: allserial allopenmp allmpi1 allmpiopenmp allmpirma allshmem allmpishm allupc allfortran
allweirdmpi: allfenix allfgmpi allampi
allfreaks: allcharm++ allgrappa alllegion
allshared: allserial allopenmp allfortran allcxx allc1z
allnew: allfortran allcxx allc1z

allmpi1:
	cd MPI1/Synch_global;        $(MAKE) global    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Synch_p2p;           $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Sparse;              $(MAKE) sparse    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Transpose;           $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Stencil;             $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/DGEMM;               $(MAKE) dgemm     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Nstream;             $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Reduce;              $(MAKE) reduce    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Random;              $(MAKE) random    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/Branch;              $(MAKE) branch    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"  \
                                                       "MATRIX_RANK         = $(matrix_rank)"        \
                                                       "NUMBER_OF_FUNCTIONS = $(number_of_functions)"
	cd MPI1/PIC-static;          $(MAKE) pic       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPI1/AMR;                 $(MAKE) amr       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allfenix:
	cd scripts/small;            $(MAKE) -f  Makefile_FENIX runfenix
	cd FENIX/Stencil;            $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FENIX/AMR;                $(MAKE) amr       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FENIX/Synch_p2p;          $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FENIX/Transpose;          $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allampi:
	cd AMPI/Synch_global;        $(MAKE) global    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Synch_p2p;           $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Sparse;              $(MAKE) sparse    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Transpose;           $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Stencil;             $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/DGEMM;               $(MAKE) dgemm     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Nstream;             $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Reduce;              $(MAKE) reduce    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Random;              $(MAKE) random    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/Branch;              $(MAKE) branch    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"  \
                                                       "MATRIX_RANK         = $(matrix_rank)"        \
                                                       "NUMBER_OF_FUNCTIONS = $(number_of_functions)"
	cd AMPI/PIC;                 $(MAKE) pic       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd AMPI/AMR;                 $(MAKE) amr       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allfgmpi:
	cd scripts/small;              $(MAKE) -f  Makefile_FG_MPI runfgmpi
	cd scripts/wide;               $(MAKE) -f  Makefile_FG_MPI runfgmpi
	cd FG_MPI/Synch_global;        $(MAKE) global    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Synch_p2p;           $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Sparse;              $(MAKE) sparse    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Transpose;           $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Stencil;             $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/DGEMM;               $(MAKE) dgemm     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Nstream;             $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Reduce;              $(MAKE) reduce    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Random;              $(MAKE) random    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd FG_MPI/Branch;              $(MAKE) branch    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"  \
                                                         "MATRIX_RANK         = $(matrix_rank)"        \
                                                         "NUMBER_OF_FUNCTIONS = $(number_of_functions)"
	cd FG_MPI/PIC-static;          $(MAKE) pic       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allmpiopenmp:
	cd MPIOPENMP/Nstream;       $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPIOPENMP/Synch_p2p;     $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPIOPENMP/Stencil;       $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPIOPENMP/Transpose;     $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allmpiomp: allmpiopenmp

allmpirma:
	cd MPIRMA/Synch_p2p;        $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPIRMA/Stencil;          $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPIRMA/Transpose;        $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allshmem:
	cd SHMEM/Synch_p2p;         $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SHMEM/Stencil;           $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SHMEM/Transpose;         $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allmpishm:
	cd MPISHM/Synch_p2p;        $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPISHM/Stencil;          $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd MPISHM/Transpose;        $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allmpi: allmpi1 allmpiomp allmpirma allmpishm

allupc:
	cd UPC/Synch_p2p;           $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd UPC/Stencil;             $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd UPC/Transpose;           $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allpgas: allshmem allupc allmpirma

allopenmp:
	cd OPENMP/DGEMM;            $(MAKE) dgemm     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Nstream;          $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Reduce;           $(MAKE) reduce    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Refcount;         $(MAKE) refcount  "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Stencil;          $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Transpose;        $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Random;           $(MAKE) random    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Sparse;           $(MAKE) sparse    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Synch_global;     $(MAKE) global    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Synch_p2p;        $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd OPENMP/Branch;           $(MAKE) branch    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"     \
                                                      "MATRIX_RANK         = $(matrix_rank)"   \
                                                      "NUMBER_OF_FUNCTIONS = $(number_of_functions)"
	cd OPENMP/PIC;              $(MAKE) pic       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allcharm++:
	cd CHARM++/Synch_p2p;       $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd CHARM++/Stencil;         $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd CHARM++/Transpose;       $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allgrappa:
	cd GRAPPA/Nstream;         $(MAKE) nstream    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd GRAPPA/Random;          $(MAKE) random     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd GRAPPA/Synch_global;    $(MAKE) global     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd GRAPPA/Synch_p2p;       $(MAKE) p2p        "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd GRAPPA/Stencil;         $(MAKE) stencil    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd GRAPPA/Transpose;       $(MAKE) transpose  "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

alllegion:
	cd LEGION/Stencil;         $(MAKE) stencil    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd LEGION/Transpose;       $(MAKE) transpose  "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allserial:
	cd SERIAL/DGEMM;            $(MAKE) dgemm     "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Nstream;          $(MAKE) nstream   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Reduce;           $(MAKE) reduce    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Stencil;          $(MAKE) stencil   "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Transpose;        $(MAKE) transpose "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Random;           $(MAKE) random    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Sparse;           $(MAKE) sparse    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Synch_p2p;        $(MAKE) p2p       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/Branch;           $(MAKE) branch    "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"     \
                                                      "MATRIX_RANK         = $(matrix_rank)"   \
                                                      "NUMBER_OF_FUNCTIONS = $(number_of_functions)"
	cd SERIAL/PIC;              $(MAKE) pic       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"
	cd SERIAL/AMR;              $(MAKE) amr       "DEFAULT_OPT_FLAGS   = $(PRK_FLAGS)"

allfortran:
	$(MAKE) -C FORTRAN

allfortranserial:
	$(MAKE) -C FORTRAN serial

allfortranpretty:
	$(MAKE) -C FORTRAN pretty

allfortranopenmp:
	$(MAKE) -C FORTRAN openmp

allfortrantarget:
	$(MAKE) -C FORTRAN target

allfortrancoarray:
	$(MAKE) -C FORTRAN coarray

allcxx:
	$(MAKE) -C Cxx11

allc1z:
	$(MAKE) -C C1z

allrust:
	$(MAKE) -C RUST

clean:
	cd MPI1/DGEMM;              $(MAKE) clean
	cd MPI1/Nstream;            $(MAKE) clean
	cd MPI1/Reduce;             $(MAKE) clean
	cd MPI1/Stencil;            $(MAKE) clean
	cd MPI1/Transpose;          $(MAKE) clean
	cd MPI1/Random;             $(MAKE) clean
	cd MPI1/Sparse;             $(MAKE) clean
	cd MPI1/Synch_global;       $(MAKE) clean
	cd MPI1/Synch_p2p;          $(MAKE) clean
	cd MPI1/Branch;             $(MAKE) clean
	cd MPI1/PIC-static;         $(MAKE) clean
	cd MPI1/AMR;                $(MAKE) clean
	cd FENIX/Stencil;           $(MAKE) clean
	cd FENIX/Transpose;         $(MAKE) clean
	cd FENIX/Sparse;            $(MAKE) clean
	cd FENIX/Synch_p2p;         $(MAKE) clean
	cd FENIX/AMR;               $(MAKE) clean
	cd FG_MPI/DGEMM;            $(MAKE) clean
	cd FG_MPI/Nstream;          $(MAKE) clean
	cd FG_MPI/Reduce;           $(MAKE) clean
	cd FG_MPI/Stencil;          $(MAKE) clean
	cd FG_MPI/Transpose;        $(MAKE) clean
	cd FG_MPI/Random;           $(MAKE) clean
	cd FG_MPI/Sparse;           $(MAKE) clean
	cd FG_MPI/Synch_global;     $(MAKE) clean
	cd FG_MPI/Synch_p2p;        $(MAKE) clean
	cd FG_MPI/Branch;           $(MAKE) clean
	cd FG_MPI/PIC-static;       $(MAKE) clean
	cd AMPI/DGEMM;              $(MAKE) clean
	cd AMPI/Nstream;            $(MAKE) clean
	cd AMPI/Reduce;             $(MAKE) clean
	cd AMPI/Stencil;            $(MAKE) clean
	cd AMPI/Transpose;          $(MAKE) clean
	cd AMPI/Random;             $(MAKE) clean
	cd AMPI/Sparse;             $(MAKE) clean
	cd AMPI/Synch_global;       $(MAKE) clean
	cd AMPI/Synch_p2p;          $(MAKE) clean
	cd AMPI/Branch;             $(MAKE) clean
	cd AMPI/PIC;                $(MAKE) clean
	cd AMPI/AMR;                $(MAKE) clean
	cd MPIRMA/Stencil;          $(MAKE) clean
	cd MPIRMA/Synch_p2p;        $(MAKE) clean
	cd MPIRMA/Transpose;        $(MAKE) clean
	cd UPC/Stencil;             $(MAKE) clean
	cd UPC/Transpose;           $(MAKE) clean
	cd UPC/Synch_p2p;           $(MAKE) clean
	cd MPISHM/Stencil;          $(MAKE) clean
	cd MPISHM/Synch_p2p;        $(MAKE) clean
	cd MPISHM/Transpose;        $(MAKE) clean
	cd SHMEM/Transpose;         $(MAKE) clean
	cd SHMEM/Stencil;           $(MAKE) clean
	cd SHMEM/Synch_p2p;         $(MAKE) clean
	cd CHARM++/Stencil;         $(MAKE) clean
	cd CHARM++/Synch_p2p;       $(MAKE) clean
	cd CHARM++/Transpose;       $(MAKE) clean
	cd GRAPPA/Synch_p2p;        $(MAKE) clean
	cd GRAPPA/Stencil;          $(MAKE) clean
	cd GRAPPA/Transpose;        $(MAKE) clean
	cd LEGION/Stencil;          $(MAKE) clean
	cd LEGION/Transpose;        $(MAKE) clean
	cd MPIOPENMP/Nstream;       $(MAKE) clean
	cd MPIOPENMP/Stencil;       $(MAKE) clean
	cd MPIOPENMP/Transpose;     $(MAKE) clean
	cd MPIOPENMP/Synch_p2p;     $(MAKE) clean
	cd OPENMP/DGEMM;            $(MAKE) clean
	cd OPENMP/Nstream;          $(MAKE) clean
	cd OPENMP/Reduce;           $(MAKE) clean
	cd OPENMP/Refcount;         $(MAKE) clean
	cd OPENMP/Stencil;          $(MAKE) clean
	cd OPENMP/Transpose;        $(MAKE) clean
	cd OPENMP/Random;           $(MAKE) clean
	cd OPENMP/Sparse;           $(MAKE) clean
	cd OPENMP/Synch_global;     $(MAKE) clean
	cd OPENMP/Synch_p2p;        $(MAKE) clean
	cd OPENMP/Branch;           $(MAKE) clean
	cd OPENMP/PIC;              $(MAKE) clean
	cd SERIAL/DGEMM;            $(MAKE) clean
	cd SERIAL/Nstream;          $(MAKE) clean
	cd SERIAL/Reduce;           $(MAKE) clean
	cd SERIAL/Stencil;          $(MAKE) clean
	cd SERIAL/Transpose;        $(MAKE) clean
	cd SERIAL/Random;           $(MAKE) clean
	cd SERIAL/Sparse;           $(MAKE) clean
	cd SERIAL/Synch_p2p;        $(MAKE) clean
	cd SERIAL/Branch;           $(MAKE) clean
	cd SERIAL/PIC;              $(MAKE) clean
	cd SERIAL/AMR;              $(MAKE) clean
	make -C FORTRAN clean
	make -C Cxx11 clean
	make -C C1z clean
	rm -f stats.json

veryclean: clean
	cd MPI1/Branch;             $(MAKE) veryclean
	cd OPENMP/Branch;           $(MAKE) veryclean
	cd SERIAL/Branch;           $(MAKE) veryclean
	cd MPI1/Stencil;            $(MAKE) veryclean
	cd MPI1/AMR;                $(MAKE) veryclean
	cd FENIX/Stencil;           $(MAKE) veryclean
	cd FENIX/AMR;               $(MAKE) veryclean
	cd OPENMP/Stencil;          $(MAKE) veryclean
	cd SERIAL/Stencil;          $(MAKE) veryclean
	cd SERIAL/AMR;              $(MAKE) veryclean
	cd MPIRMA/Stencil;          $(MAKE) veryclean
	cd MPISHM/Stencil;          $(MAKE) veryclean
	cd SHMEM/Stencil;           $(MAKE) veryclean
	cd FG_MPI/Stencil;          $(MAKE) veryclean
	cd MPIOPENMP/Stencil;       $(MAKE) veryclean
	cd GRAPPA/Stencil;          $(MAKE) veryclean
	cd CHARM++/Stencil;         $(MAKE) veryclean
	cd UPC/Stencil;             $(MAKE) veryclean
	cd FG_MPI/Branch;           $(MAKE) veryclean
	cd AMPI/Stencil;            $(MAKE) veryclean
	cd AMPI/Branch;             $(MAKE) veryclean
	cd AMPI/AMR;                $(MAKE) veryclean
	cd scripts/small;           $(MAKE) -f  Makefile_FG_MPI veryclean
	cd scripts/small;           $(MAKE) -f  Makefile_FENIX  veryclean
	cd scripts/wide;            $(MAKE) -f  Makefile_FG_MPI veryclean
	cd common; rm -f make.defs
