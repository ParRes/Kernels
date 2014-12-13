include ../../common/OPENMP.defs
##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) -restrict -mcmodel=large
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

RESTRICTFLAG = -DRESTRICT_KEYWORD
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers

MEMFLAG      = -DMEGAMEMWORDS=4096L
#description: number of words to store all data related to reduction vectors

#DEBUGFLAG    = -DVERBOSE
#description: default diagnostic style is silent

#NTHREADFLAG  = -DMAXTHREADS=256
#description: default thread limit is 32

USERFLAGS     = -strict-ansi -vec-report3 
#description: parameter to specify optional flags

EXTOBJS      = 
LIBS         =
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

TUNEFLAGS   = $(MEMFLAG) $(DEBUGFLAG) $(NTHREADFLAG) $(USERFLAGS)
PROGRAM     = reduce
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
