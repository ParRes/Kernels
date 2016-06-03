include ../../common/SERIAL.defs
 
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

ifndef SCRAMBLE
 SCRAMBLE=1
endif
#description: if flag is true, grid indices are scrambled to produce irregular stride

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
SCRAMBLEFLAG= -DSCRAMBLE=$(SCRAMBLE)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT\n\
SCRAMBLE=0/1           regular/irregular sparsity pattern         [1]  \n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]  \n\
VERBOSE=0/1            omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(USERFLAGS) $(SCRAMBLEFLAG) $(RESTRICTFLAG)
PROGRAM     = sparse
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
