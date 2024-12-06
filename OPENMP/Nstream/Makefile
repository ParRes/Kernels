include ../../common/OPENMP.defs

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS     = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         =
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef MAXLENGTH
  MAXLENGTH=2000000
endif
#description: default vector length (if static) is 2000000

ifndef STATIC_ALLOCATION
  STATIC_ALLOCATION=0
endif
#description: default memory allocation is dynamic (using prk_malloc)

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
ALLOCFLAG   = -DSTATIC_ALLOCATION=$(STATIC_ALLOCATION)
LENGTHFLAG  = -DMAXLENGTH=$(MAXLENGTH)

OPTIONSSTRING="Make options:\n\
OPTION                   MEANING                                  DEFAULT    \n\
STATIC_ALLOC=0/1         set dynamic/static memory allocation       [0]      \n\
MAXLENGTH=?              setting maximum stream vector length       [2000000]\n\
RESTRICT_KEYWORD=0/1     disable/enable restrict keyword (aliasing) [0]      \n\
MAXTHREADS=?             set maximum number of OpenMP threads       [512]    \n\
VERBOSE=0/1              omit/include verbose run information       [0]"

TUNEFLAGS    = $(LENGTHFLAG) $(ALLOCFLAG) $(VERBOSEFLAG) $(NTHREADFLAG) \
               $(USERFLAGS)  $(RESTRICTFLAG)
PROGRAM      = nstream
OBJS         = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common

