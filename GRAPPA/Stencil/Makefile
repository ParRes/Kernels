include ../../common/GRAPPA.defs

##### User configurable options #####
#uncomment any of the following flags (and change values) to change defaults

OPTFLAGS    = $(DEFAULT_OPT_FLAGS) -O3
#description: change above into something that is a decent optimization on you system

USERFLAGS    = 
#description: parameter to specify optional flags

#set the following variables for custom libraries and/or other objects
EXTOBJS      = 
LIBS         = $(GRAPPA_LDLIBS)
LIBPATHS     = $(GRAPPA_LDFLAGS)
INCLUDEPATHS = 

### End User configurable options ###

ifndef DOUBLE
  DOUBLE=1
endif
#description: default data type is single precision

ifndef STAR
  STAR=1
endif
#description: default stencil is compact (dense, square)

ifndef RADIUS
  RADIUS=2
endif
#description: default radius of filter to be applied is 2

ifndef LOOPGEN
  LOOPGEN=0
endif
#description: default is compact (non-expanded) form of loop body

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

VERBOSEFLAG     = -DVERBOSE=$(VERBOSE)
RADIUSFLAG      = -DRADIUS=$(RADIUS)
LOOPGENFLAG     = -DLOOPGEN=$(LOOPGEN)
STARFLAG        = -DSTAR=$(STAR)

# this is used by the C++ implicit compile rule
CXXFLAGS        = $(GRAPPA_CXXFLAGS)

OPTIONSSTRING="Make options:\n\
OPTION                  MEANING                                  DEFAULT\n\
RADIUS=?                radius of stencil                          [2]  \n\
LOOPGEN=0/1             compact/expanded stencil loop body         [0]  \n\
STAR=0/1                box/star shaped stencil                    [1]  \n\
VERBOSE=0/1             omit/include verbose run information       [0]"

TUNEFLAGS   = $(VERBOSEFLAG)  $(USERFLAGS)  $(LOOPGENFLAG)\
              $(RADIUSFLAG)   $(STARFLAG)   $(CXXFLAGS)
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

