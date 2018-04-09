include ../../common/FENIX.defs
COMOBJS += random_draw.o
COMOBJS += timestep.o

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

ifndef FT_HARNESS
  FT_HARNESS=0
endif
#description: by default we do not emit output for the Fault Tolerance harness

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
HARNESSFLAG = -DFT_HARNESS=$(FT_HARNESS)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                                  DEFAULT\n\
RESTRICT_KEYWORD=0/1    disable/enable restrict keyword (aliasing) [0]  \n\
FT_HARNESS=0/1          do not/do emit FT harness output           [1]  \n\
VERBOSE=0/1             omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(USERFLAGS) $(RESTRICTFLAG) $(HARNESSFLAG)
PROGRAM     = p2p
# objects below are the default, used by "clean," if invoked
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
