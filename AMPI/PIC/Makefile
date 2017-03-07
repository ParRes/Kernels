include ../../common/AMPI.defs
COMOBJS += random_draw.o

##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) 
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS     = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = -lm -module CommonLBs 
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef NO_MIGRATE
  NO_MIGRATE=0
endif
#description: by default we allow migration, but can turn if off

ifndef USE_PUPER
  USE_PUPER=1
endif
#description: use PUP routines for migration by default

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
USE_PUPERFLAG    = -DUSE_PUPER=$(USE_PUPER)
NO_MIGRATEFLAG   = -DNO_MIGRATE=$(NO_MIGRATE)

OPTIONSSTRING="Make options:\n\
OPTION                 MEANING                                  DEFAULT \n\
RESTRICT_KEYWORD=0/1   disable/enable restrict keyword (aliasing) [0]   \n\
VERBOSE=0/1            omit/include verbose run information       [0]   \n\
NO_MIGRATE=0/1         no rank migration, no stats collecting     [0]   \n\
USE_PUPER=0/1          use PUP routines for migration             [1]"

TUNEFLAGS    = $(VERBOSEFLAG) $(USERFLAGS)  $(RESTRICTFLAG) \
               $(USE_PUPERFLAG) $(NO_MIGRATEFLAG)
PROGRAM      = pic
OBJS         = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
