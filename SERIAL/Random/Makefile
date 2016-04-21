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

ifndef RESTRICT_KEYWORD
  RESTRICT_KEYWORD=0
endif
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

ifndef HPCC
  HPCC=1
endif
#description: Uses HPC Challenge rules: non-atomic updates, shared table, 
#             1% errors allowed; overrides ATOMICFLAG, CHUNKFLAG, ERRORPERCENTFLAG

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
HPCCFLAG         = -DHPCC=$(HPCC)
ERRORPERCENTFLAG = -DERRORPERCENT=$(ERRORPERCENT)
LONG64FLAG       = -DLONG_IS_64BITS=$(LONG_IS_64BITS)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT\n\
HPCC=0/1               do/don't impose HPCC rules                 [1]  \n\
ERRORPERCENT=?         specify percentage of errors allowed            \n\
LONG64=0/1             do/do not set long type as 64 bits         [0]  \n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]  \n\
VERBOSE=0/1            omit/include verbose run information       [0]"

TUNEFLAGS   = $(RESTRICTFLAG) $(LONG64FLAG) $(VERBOSEFLAG) \
              $(USERFLAGS)    $(HPCCFLAG)   $(ERRORPERCENTFLAG)
PROGRAM     = random
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
