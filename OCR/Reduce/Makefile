include ../../common/OCR.defs

##### User configurable options #####

#OPTFLAGS    = $(DEFAULT_OPT_FLAGS) -restrict
OPTFLAGS    = -O0 -g -restrict

#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

RESTRICTFLAG = -DRESTRICT_KEYWORD
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers (requires -restrict compiler flag)

DEBUGFLAG    = -DVERBOSE
#description: default diagnostic style is silent

USERFLAGS     = -strict-ansi -vec-report3
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = -locr
LIBPATHS     = -L${OCR_ROOT}/lib
INCLUDEPATHS = 

### End User configurable options ###

TUNEFLAGS   = $(DEBUGFLAG) $(USERFLAGS) $(RESTRICTFLAG)
PROGRAM     = reduce
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
