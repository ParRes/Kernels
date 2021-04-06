include ../../common/FENIX.defs
COMOBJS += random_draw.o
COMOBJS += timestep.o

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) 
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS     = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = -lm -lfenix -lmpi
LIBPATHS     = -L$(FENIXTOP)/lib -L$(ULFMTOP)/lib
INCLUDEPATHS = -I$(FENIXTOP)/include

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

ifndef TESTDENSE
  TESTDENSE=0
endif
#description: if flag is set, sparse matrix will be embedded in dense matrix

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
SCRAMBLEFLAG= -DSCRAMBLE=$(SCRAMBLE)
DENSEFLAG   = -DTESTDENSE=$(TESTDENSE)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT\n\
SCRAMBLE=0/1           regular/irregular sparsity pattern         [1]  \n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]  \n\
TESTDENSE=0/1          store sparse matrix in sparse/dense formet [0]  \n\
VERBOSE=0/1            omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(USERFLAGS) $(SCRAMBLEFLAG) $(DENSEFLAG) \
              $(RESTRICTFLAG)
PROGRAM     = sparse
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
