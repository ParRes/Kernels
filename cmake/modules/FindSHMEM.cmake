#[=======================================================================[.rst:
FindSHMEM
---------

Finds an OpenSHMEM implementation via its ``oshcc``/``oshrun`` compiler
wrapper (Sandia OpenSHMEM, Open MPI's built-in OSHMEM, OSSS-UCX, Cray
SHMEM, ...). Corresponds to ``SHMEMCC``/``common/SHMEM.defs`` in the
legacy Makefile build.

There is no shmem-config/pkg-config across implementations to query, so
this mirrors what CMake's own FindMPI does for mpicc: ask the wrapper
what flags it would use (``-show``, falling back to ``--showme`` for
wrappers that don't support ``-show``) and parse them into an IMPORTED
target, rather than assuming a fixed library/include layout.

Result variables:
  SHMEM_FOUND          - TRUE if oshcc and oshrun were found and -show parsed
  SHMEM_C_COMPILER      - path to the oshcc compiler wrapper
  SHMEM_RUN_EXECUTABLE  - path to the oshrun launcher wrapper
  SHMEM_INCLUDE_DIRS    - include directories reported by oshcc -show
  SHMEM_LIBRARIES       - libraries reported by oshcc -show

Imported target:
  SHMEM::SHMEM
#]=======================================================================]

find_program(SHMEM_C_COMPILER oshcc)
find_program(SHMEM_RUN_EXECUTABLE oshrun)

if(SHMEM_C_COMPILER)
  execute_process(
    COMMAND "${SHMEM_C_COMPILER}" -show
    OUTPUT_VARIABLE _shmem_show_output
    RESULT_VARIABLE _shmem_show_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT _shmem_show_result EQUAL 0 OR NOT _shmem_show_output)
    # Sandia OpenSHMEM / OSSS-UCX style wrappers use --showme instead of -show.
    execute_process(
      COMMAND "${SHMEM_C_COMPILER}" --showme
      OUTPUT_VARIABLE _shmem_show_output
      RESULT_VARIABLE _shmem_show_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  if(_shmem_show_output)
    separate_arguments(_shmem_show_args NATIVE_COMMAND "${_shmem_show_output}")
    set(SHMEM_INCLUDE_DIRS "")
    set(SHMEM_LIBRARIES "")
    set(_shmem_link_dirs "")
    foreach(_arg IN LISTS _shmem_show_args)
      if(_arg MATCHES "^-I(.+)$")
        list(APPEND SHMEM_INCLUDE_DIRS "${CMAKE_MATCH_1}")
      elseif(_arg MATCHES "^-L(.+)$")
        list(APPEND _shmem_link_dirs "${CMAKE_MATCH_1}")
      elseif(_arg MATCHES "^-l(.+)$")
        list(APPEND SHMEM_LIBRARIES "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    # Resolve bare -l names against the -L dirs oshcc reported (plus the
    # default system search path) so SHMEM::SHMEM carries full paths, same
    # as CMake's own IMPORTED targets for MPI/BLAS/etc.
    set(_shmem_resolved_libs "")
    foreach(_lib IN LISTS SHMEM_LIBRARIES)
      find_library(_shmem_lib_${_lib} NAMES ${_lib} HINTS ${_shmem_link_dirs})
      if(_shmem_lib_${_lib})
        list(APPEND _shmem_resolved_libs "${_shmem_lib_${_lib}}")
      else()
        list(APPEND _shmem_resolved_libs "${_lib}")
      endif()
    endforeach()
    set(SHMEM_LIBRARIES "${_shmem_resolved_libs}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SHMEM
  REQUIRED_VARS SHMEM_C_COMPILER SHMEM_RUN_EXECUTABLE SHMEM_LIBRARIES
)

if(SHMEM_FOUND AND NOT TARGET SHMEM::SHMEM)
  add_library(SHMEM::SHMEM INTERFACE IMPORTED)
  set_target_properties(SHMEM::SHMEM PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SHMEM_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SHMEM_LIBRARIES}"
  )
endif()

mark_as_advanced(SHMEM_C_COMPILER SHMEM_RUN_EXECUTABLE)
