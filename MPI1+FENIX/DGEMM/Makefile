include ../../common/MPI.defs

##### User configurable options #####

OPTFLAGS     = $(DEFAULT_OPT_FLAGS) 
#description: change above into something that is a decent optimization on you system

#uncomment any of the following flags (and change values) to change defaults

USERFLAGS     =    
#description: parameter to specify optional flags

EXTOBJS      = 
LIBS         = -lm
LIBPATHS     = 
INCLUDEPATHS = 

### End User configurable options ###

ifndef VERBOSE
  VERBOSE=0
endif
#description: default diagnostic style is silent

ifndef BOFFSET
  BOFFSET=12
endif
#description: set this flag to some value to override default first array 
#             dimension padding (12) of tiles used in non-MKL version

PROGRAM    = dgemm
VERBOSEFLAG= -DVERBOSE=$(VERBOSE)
OFFSETFLAG = -DBOFFSET=$(BOFFSET)

OPTIONSSTRING="Make options:\n\
OPTION         MEANING                                              DEFAULT \n\
BOFFSET=?      override default first array dimension padding of tiles [12] \n\
VERBOSE=0/1    omit/include verbose run information                    [0]"

TUNEFLAGS   = $(OFFSETFLAG) $(VERBOSEFLAG) $(USERFLAGS)
OBJS        = $(PROGRAM).o $(COMOBJS) 

include ../../common/make.common
