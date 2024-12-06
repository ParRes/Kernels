include ../../common/OPENMP.defs
##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         =
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

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
NTHREADFLAG = -DMAXTHREADS=$(MAXTHREADS)

OPTIONSSTRING="Make options:\n\
OPTION         MEANING                            DEFAULT\n\
MAXTHREADS=?   set maximum number of OpenMP threads [512]\n\
VERBOSE=0/1    omit/include verbose run information [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(NTHREADFLAG) $(USERFLAGS) 
PROGRAM     = global
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
