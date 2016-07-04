include ../../common/OPENMP.defs
COMOBJS += random_draw.o

##### User configurable options #####
#uncomment any of the following flags (or change values) to change defaults

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#RESTRICTFLAG = -DRESTRICT_KEYWORD
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers

#DEBUGFLAG   = -DVERBOSE
#description: default diagnostic style is silent

#NTHREADFLAG = -DMAXTHREADS=n
#description: default thread limit is 256

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = -lm
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###
TUNEFLAGS    = $(RESTRICTFLAG) $(DEBUGFLAG)  $(USERFLAGS) $(NTHREADFLAG)
PROGRAM     = pic
OBJS        = $(PROGRAM).o $(COMOBJS) 

include ../../common/make.common

