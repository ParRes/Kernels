include ../../common/OPENMP.defs

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

#DEBUGFLAG    = -DVERBOSE
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
  ifeq ($(CONTENDED),1)
    override LOCK=2
  else
    ifeq ($(LOCK),1)
      override LOCK=2
    endif
  endif
endif

#NTHREADFLAG  = -DMAXTHREADS=n
#description: default thread limit is 256

USERFLAGS     = 
#description: parameter to specify optional flags

EXTOBJS      = 
LIBS         = -lm
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###
DEPENDENTFLAG=-DDEPENDENT=$(DEPENDENT)
INTEGERFLAG=-DINTEGER=$(INTEGER)
CONTENDEDFLAG=-DCONTENDED=$(CONTENDED)
LOCKFLAG=-DLOCK=$(LOCK)
STREAMFLAG=-DSTREAM=$(STREAM)

TUNEFLAGS    = $(DEPENDENTFLAG) $(DEBUGFLAG) $(NTHREADFLAG) $(USERFLAGS)\
               $(LOCKFLAG) $(INTEGERFLAG) $(CONTENDEDFLAG) $(STREAMFLAG)
PROGRAM     = refcount
# objects below are the default, used by "clean," if invoked
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
