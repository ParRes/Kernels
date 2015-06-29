include ../../common/GRAPPA.defs
##### User configurable options #####

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) 
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

#RESTRICTFLAG = -DRESTRICT_KEYWORD
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers

#DEBUGFLAG   = -DVERBOSE
#description: default diagnostic style is silent

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = $(GRAPPA_LDLIBS)
LIBPATHS     = $(GRAPPA_LDFLAGS)

### End User configurable options ###

# this is used by the C++ implicit compile rule
<<<<<<< 3472f59a2d7d2e53cf51993eac3df6b024afc727
CXXFLAGS= $(GRAPPA_CXXFLAGS)
=======
CXXFLAGS= $(GRAPPA_CXXFLAGS)  
>>>>>>> Changes to GRAPPA Makefiles and GRAPPA.defs from Jacob's pull request.

TUNEFLAGS    = $(RESTRICTFLAG) $(DEBUGFLAG)  $(USERFLAGS) $(CXXFLAGS)
PROGRAM     = p2p
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common
