include ../../common/UPC.defs

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

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate
#             data accessed through pointers (requires -restrict compiler flag)

ifndef USE_BUPC_EXT
  USE_BUPC_EXT=0
endif
#description: default is to not enable BUPC extensions

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG     = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG    = -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
BUPCFLAG        = -DUSE_BUPC_EXT=$(USE_BUPC_EXT)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                                  DEFAULT\n\
RESTRICT_KEYWORD=0/1    disable/enable restrict keyword (aliasing) [0]  \n\
USE_BUPC_EXT=0/1        disable/enable Berkely UPC extensions      [0]  \n\
VERBOSE=0/1             omit/include verbose run information       [0]"

TUNEFLAGS    = $(RESTRICTFLAG) $(VERBOSEFLAG)$(USERFLAGS) $(BUPCFLAG)
PROGRAM     = p2p
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
