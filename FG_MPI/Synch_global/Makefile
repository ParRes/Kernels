include ../../common/FG_MPI.defs
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

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)

OPTIONSSTRING="Make options:\n\
OPTION         MEANING                            DEFAULT\n\
VERBOSE=0/1    omit/include verbose run information [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(USERFLAGS) 
PROGRAM     = global
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
