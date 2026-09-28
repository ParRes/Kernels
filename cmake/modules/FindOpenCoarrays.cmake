#[=======================================================================[.rst:
FindOpenCoarrays
----------------

Finds the OpenCoarrays compiler wrapper (``caf``/``cafrun``) and the
``caf_mpi``/``caf_mpich``/``caf_openmpi`` runtime library used by gfortran's
``-fcoarray=lib`` mode. Corresponds to ``COARRAYFLAG=-fcoarray=lib -lcaf_mpi``
in common/make.defs.gcc.

Result variables:
  OpenCoarrays_FOUND    - TRUE if caf, cafrun, and the runtime library were found
  OpenCoarrays_CAF       - path to the caf compiler wrapper
  OpenCoarrays_CAFRUN    - path to the cafrun launcher wrapper
  OpenCoarrays_LIBRARY   - path to the caf runtime library
#]=======================================================================]

find_program(OpenCoarrays_CAF caf)
find_program(OpenCoarrays_CAFRUN cafrun)
find_library(OpenCoarrays_LIBRARY NAMES caf_openmpi caf_mpich caf_mpi)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCoarrays
  REQUIRED_VARS OpenCoarrays_CAF OpenCoarrays_CAFRUN OpenCoarrays_LIBRARY
)

mark_as_advanced(OpenCoarrays_CAF OpenCoarrays_CAFRUN OpenCoarrays_LIBRARY)
