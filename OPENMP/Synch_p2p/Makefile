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

ifndef SYNCHRONOUS
  SYNCHRONOUS=0
endif
#description: default handshake between threads is off

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

ifndef MAXTHREADS
  MAXTHREADS=512
endif
#description: default thread limit is 512

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG= -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
NTHREADFLAG = -DMAXTHREADS=$(MAXTHREADS)
SYNCHFLAG   = -DSYNCHRONOUS=$(SYNCHRONOUS)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                                  DEFAULT\n\
SYNCHRONOUS=0/1         disable/enable handshake between threads   [0]  \n\
RESTRICT_KEYWORD=0/1    disable/enable restrict keyword (aliasing) [0]  \n\
MAXTHREADS=?            set maximum number of OpenMP threads       [512]\n\
VERBOSE=0/1             omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG) $(NTHREADFLAG) $(USERFLAGS) $(SYNCHFLAG) \
              $(RESTRICTFLAG)
PROGRAM     = p2p
# objects below are the default, used by "clean," if invoked
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
