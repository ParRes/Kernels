include ../../common/FENIX.defs
COMOBJS += random_draw.o
COMOBJS += timestep.o

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = -lm -lfenix -lmpi
LIBPATHS     = -L$(FENIXTOP)/lib -L$(ULFMTOP)/lib
INCLUDEPATHS = -I$(FENIXTOP)/include

### End User configurable options ###

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

ifndef SYNCHRONOUS
  SYNCHRONOUS=0
endif
#description: turn on synchronous (blocking) communications

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
BLOCKFLAG   = -DSYNCHRONOUS=$(SYNCHRONOUS)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                      DEFAULT\n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing)     [0]  \n\
VERBOSE=0/1            omit/include verbose run information           [0]  \n\
SYNCHRONOUS=0/1        Use a/synchronous communications               [0]"

TUNEFLAGS    = $(VERBOSEFLAG) $(USERFLAGS) $(RESTRICTFLAG) $(BLOCKFLAG)
PROGRAM      = transpose
OBJS         = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
