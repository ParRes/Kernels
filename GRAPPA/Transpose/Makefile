include ../../common/GRAPPA.defs
##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = $(GRAPPA_LDLIBS)
LIBPATHS     = $(GRAPPA_LDFLAGS)
INCLUDEPATHS = 

### End User configurable options ###

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG     = -DVERBOSE=$(VERBOSE)

# this is used by the C++ implicit compile rule
CXXFLAGS        = $(GRAPPA_CXXFLAGS)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                                  DEFAULT\n\
VERBOSE=0/1             omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG)  $(USERFLAGS) $(CXXFLAGS)
PROGRAM     = transpose
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
