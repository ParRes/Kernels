ifndef NUMBER_OF_FUNCTIONS
  NUMBER_OF_FUNCTIONS=40
endif

ifndef MATRIX_RANK
  MATRIX_RANK=10
endif

include ../../common/MPI.defs
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

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT\n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]  \n\
VERBOSE=0/1            omit/include verbose run information       [0]"

TUNEFLAGS   = $(RESTRICTFLAG) $(VERBOSEFLAG) $(USERFLAGS)
PROGRAM     = branch
OBJS        = $(PROGRAM).o $(COMOBJS) func.o

#default: usage
#	@echo "            \"make veryclean\" to remove custom built files as well"

include ../../common/make.common

func.c:
	@echo "############################################################"
	@echo "##### No file func.c -- invoking func_gen to create it #####"
	@echo "############################################################"
	./func_gen ${MATRIX_RANK} ${NUMBER_OF_FUNCTIONS}

veryclean:
	@rm -f func.c ___*
	make clean  
