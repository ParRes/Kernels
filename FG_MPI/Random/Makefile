include ../../common/FG_MPI.defs

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS     = 
#description: parameter to specify optional flags

EXTOBJS      = 
LIBS         =
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

ifndef LOOKAHEAD
  LOOKAHEAD=1024
endif
#description: the HPC Challenge rule is a lookahead of no more than 1024

ifndef LONG_IS_64BITS
  LONG_IS_64BITS=0
endif
#description: can use "long" for 64 bit integers instead of "long long"

VERBOSEFLAG      = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG     = -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
LOOKAHEADFLAG    = -DLOOKAHEAD=$(LOOKAHEAD)
LONG64FLAG       = -DLONG_IS_64BITS=$(LONG_IS_64BITS)

OPTIONSSTRING="Make options:\n\
OPTION               MEANING                                   DEFAULT\n\
LONG_IS_64BITS=0/1   do/do not set long type as 64 bits         [0]   \n\
RESTRICT_KEYWORD=0/1 disable/enable restrict keyword (aliasing) [0]   \n\
LOOKAHEAD=?          batch factor for generating table indices  [1024]\n\
VERBOSE=0/1          omit/include verbose run information       [0]"

TUNEFLAGS   = $(RESTRICTFLAG) $(LONG64FLAG) $(VERBOSEFLAG) $(NTHREADFLAG)\
              $(USERFLAGS)    $(LOOKAHEADFLAG)
PROGRAM     = random
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
