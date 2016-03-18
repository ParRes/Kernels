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

ifeq ($(DEPENDENT),1)
  override INTEGER=0
  ifeq ($(CONTENDED),1)
    override LOCK=1
  endif
else
  ifndef INTEGER
    INTEGER=0
  endif
  ifndef LOCK
    LOCK=1
  endif
endif
#description: default counter update is protected with a lock
#description: default type is floating point

ifndef CONTENDED
  CONTENDED=1
endif
#description: default counter access type is contended

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
TUNEFLAGS    = $(DEPENDENTFLAG) $(DEBUGFLAG) $(NTHREADFLAG) $(USERFLAGS)\
               $(LOCKFLAG) $(INTEGERFLAG) $(CONTENDEDFLAG)
PROGRAM     = refcount
# objects below are the default, used by "clean," if invoked
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
