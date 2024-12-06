include ../../common/OPENMP.defs

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

ifndef MAXTHREADS
  MAXTHREADS=512
endif
#description: default thread limit is 512

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

ifndef HPCC
  HPCC=1
endif
#description: Uses HPC Challenge rules: non-atomic updates, shared table, 
#             1% errors allowed; overrides ATOMICFLAG, CHUNKFLAG, ERRORPERCENTFLAG

ifndef ATOMIC
  ATOMIC=0
endif
#description: protect threads from simultaneous access to same memory location

ifndef CHUNKED
  CHUNKED=0
endif
#description: assign chunks of Table to different threads, so no atomic needed

ifndef ERRORPERCENT
  ERRORPERCENT=1
endif
#description: percent errors allowed in solution; if unset, no errors allowed

ifndef LONG_IS_64BITS
  LONG_IS_64BITS=0
endif
#description: can use "long" for 64 bit integers instead of "long long"

VERBOSEFLAG      = -DVERBOSE=$(VERBOSE)
RESTRICTFLAG     = -DRESTRICT_KEYWORD=$(RESTRICT_KEYWORD)
NTHREADFLAG      = -DMAXTHREADS=$(MAXTHREADS)
HPCCFLAG         = -DHPCC=$(HPCC)
ATOMICFLAG       = -DATOMIC=$(ATOMIC)
CHUNKEDFLAG      = -DCHUNKED=$(CHUNKED)
ERRORPERCENTFLAG = -DERRORPERCENT=$(ERRORPERCENT)
LONG64FLAG       = -DLONG_IS_64BITS=$(LONG_IS_64BITS)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT\n\
HPCC=0/1               do/do not impose HPCC rules                [1]  \n\
ATOMIC=0/1             use atomic access to update table elements [0]  \n\
CHUNKED=0/1            do/do not assign table chunks to threads   [0]  \n\
ERRORPERCENT=?         specify percentage of errors allowed            \n\
LONG64=0/1             do/do not set long type as 64 bits         [0]  \n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]  \n\
MAXTHREADS=?           set maximum number of OpenMP threads       [512]\n\
VERBOSE=0/1            omit/include verbose run information       [0]"

TUNEFLAGS   = $(RESTRICTFLAG) $(LONG64FLAG) $(VERBOSEFLAG) $(NTHREADFLAG)\
              $(USERFLAGS)    $(ATOMICFLAG) $(CHUNKFLAG)   $(HPCCFLAG)   \
              $(ERRORPERCENTFLAG)
PROGRAM     = random
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
