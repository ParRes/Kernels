#[=======================================================================[.rst:
FindGlobalArrays
----------------

Finds the Global Arrays (GA) toolkit, used by the ``*-ga`` Fortran targets
(``GAFLAG`` in common/make.defs.*). GA also requires ARMCI-MPI; both are
autotools projects with no upstream CMake config, so this is a manual probe
rather than a find_dependency() forward.

Search hints: the ``GADIR`` environment variable (used by ci/install-ga.sh
and every make.defs.* file), falling back to standard prefixes.

Result variables:
  GlobalArrays_FOUND         - TRUE if headers and the ga library were found
  GlobalArrays_INCLUDE_DIR   - directory containing ga.h
  GlobalArrays_LIBRARY       - path to libga
#]=======================================================================]

find_path(GlobalArrays_INCLUDE_DIR ga.h
  HINTS ENV GADIR
  PATH_SUFFIXES include
)
find_library(GlobalArrays_LIBRARY
  NAMES ga
  HINTS ENV GADIR
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GlobalArrays
  REQUIRED_VARS GlobalArrays_LIBRARY GlobalArrays_INCLUDE_DIR
)

mark_as_advanced(GlobalArrays_INCLUDE_DIR GlobalArrays_LIBRARY)
