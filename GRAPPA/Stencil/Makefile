include ../../common/GRAPPA.defs

##### User configurable options #####
#uncomment any of the following flags (and change values) to change defaults

OPTFLAGS    = $(DEFAULT_OPT_FLAGS)
#description: change above into something that is a decent optimization on you system

#RESTRICTFLAG = -DRESTRICT_KEYWORD
#description: the "restrict" keyword can be used on IA platforms to disambiguate  
#             data accessed through pointers

ifndef RADIUS
  RADIUS=2
endif
#description: default radius of filter to be applied is 2

ifndef LOOPGEN
  LOOPGEN=0
endif
#description: default is compact (non-expanded) form of loop body

STARFLAG   = -DSTAR
#description: default stencil is star-shaped

#DEBUGFLAG   = -DVERBOSE
#description: default diagnostic style is silent

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = $(GRAPPA_LDLIBS)
LIBPATHS     = $(GRAPPA_LDFLAGS)

### End User configurable options ###
RADIUSFLAG = -DRADIUS=$(RADIUS)
LOOPGENFLAG= -DLOOPGEN=$(LOOPGEN)

# this is used by the C++ implicit compile rule
CXXFLAGS= $(GRAPPA_CXXFLAGS)

TUNEFLAGS    = $(RESTRICTFLAG) $(DEBUGFLAG) $(RADIUSFLAG)  \
               $(USERFLAGS) $(CXXFLAGS) $(STARFLAG) \
               $(LOOPGENFLAG) 
PROGRAM     = stencil
OBJS        = $(PROGRAM).o $(COMOBJS)

include ../../common/make.common

$(PROGRAM).o: loop_body_star.incl 

loop_body_star.incl:
	@echo "#########################################################################"
	@echo "##### No file loop_body_star.incl -- invoking loop_gen to create it #####"
	@echo "#########################################################################"
	../../common/Stencil/loop_gen $(RADIUS) 1

veryclean:
	@rm -f loop_body_star.incl
	make clean

