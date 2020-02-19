include ../../common/OPENMP.defs

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

#NTHREADFLAG  = -DMAXTHREADS=n
#description: default thread limit is 512

USERFLAGS     = 
#description: parameter to specify optional flags

EXTOBJS      = 
LIBS         = -lm
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef MAXTHREADS
  MAXTHREADS=512
endif
#description: default thread limit is 512

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

ifndef DEPENDENT
  DEPENDENT=0
endif
#description: default update of counter pair is independent

ifndef CONTENDED
  CONTENDED=1
endif
#description: default counter access type is contended

ifndef LOCK
  LOCK=2
endif
#description: default counter update is protected with a lock

ifndef INTEGER
  INTEGER=0
endif
#description: default counter type is floating point

ifndef STREAM
  STREAM=1
endif
#description: assume there will be some private work for each thread

ifeq ($(DEPENDENT),1)
  override INTEGER=0
endif
#dependent updates require floating point

ifeq ($(CONTENDED),1)
  ifeq ($(DEPENDENT),1)
    override LOCK=2
  endif
endif

PROGRAM      = refcount

DEPENDENTFLAG=-DDEPENDENT=$(DEPENDENT)
INTEGERFLAG  =-DINTEGER=$(INTEGER)
CONTENDEDFLAG=-DCONTENDED=$(CONTENDED)
LOCKFLAG     =-DLOCK=$(LOCK)
STREAMFLAG   =-DSTREAM=$(STREAM)
VERBOSEFLAG  =-DVERBOSE=$(VERBOSE)
NTHREADFLAG  =-DMAXTHREADS=$(MAXTHREADS)

OPTIONSSTRING="Make options:\n\
OPTION         MEANING                                    DEFAULT\n\
CONTENDED=0/1  uncontended/contended counters               [1]  \n\
LOCK=0/1/2     no locks/atomics/OpenMP locks                [2]  \n\
INTEGER=0/1    counter data type integer/floating point     [1]  \n\
DEPENDENT=0/1  independent/intertwined counter pair updates [0]  \n\
STREAM=0/1     disallow/allow independent thread work       [1]  \n\
MAXTHREADS=?   set maximum number of OpenMP threads         [512]\n\
VERBOSE=0/1    omit/include verbose run information         [0]"

TUNEFLAGS    = $(DEPENDENTFLAG) $(NTHREADFLAG) $(USERFLAGS)  $(LOCKFLAG)\
               $(INTEGERFLAG) $(CONTENDEDFLAG) $(STREAMFLAG) $(VERBOSEFLAG) \
               $(LOCKHINTFLAG)
# objects below are the default, used by "clean," if invoked
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
