include ../../common/UPC.defs

##### User configurable options #####

#description: change this into something that is a decent optimization on your system
OPTFLAGS    = $(DEFAULT_OPT_FLAGS)

#uncomment any of the following flags (and change values) to change defaults
#description: default diagnostic style is silent
DEBUGFLAG    = -DVERBOSE

#description: parameter to specify optional flags
USERFLAGS     =

#set the following variables for custom libraries and/or other objects
EXTOBJS      =
LIBS         =
LIBPATHS     =
INCLUDEPATHS =

### End User configurable options ###

TUNEFLAGS   = $(BLOCKFLAG) $(DEBUGFLAG) $(USERFLAGS)
PROGRAM     = transpose
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
