include ../../common/SHMEM.defs

##### User configurable options #####
#uncomment any of the following flags (and change values) to change defaults

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) 
#description: change above into something that is a decent optimization on you system

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = 
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef BARRIER_SYNCH
  BARRIER_SYNCH=0
endif
#description: default is not to use a barrier but point to point synchronization

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG     = -DVERBOSE=$(VERBOSE)
SYNCHFLAG       = -DBARRIER_SYNCH=$(BARRIER_SYNCH)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                             DEFAULT\n\
BARRIER_SYNCH=0/1  use point-to-point/barrier synchronization [0]  \n\
VERBOSE=0/1        omit/include verbose run information       [0]"

TUNEFLAGS    = $(VERBOSEFLAG)  $(SYNCHFLAG)
PROGRAM     = transpose
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common

