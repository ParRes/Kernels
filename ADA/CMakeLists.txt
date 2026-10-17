#
# CMake port of ADA/Makefile. CMake has no native Ada language support
# (no LANGUAGES Ada for project()/enable_language()), so this drives
# gnatmake directly via add_custom_command(), the same approach used
# elsewhere in this repo for build steps CMake can't natively express
# (e.g. common/Stencil/loop_gen code generation in the C-family ports).
#
# Standalone project: `cd ADA && cmake -S . -B build` works on its own.
# Also add_subdirectory()'d from the repo root.
#

cmake_minimum_required(VERSION 3.18)
project(PRK_ADA NONE)
enable_testing()

find_program(GNATMAKE_EXECUTABLE gnatmake)

if(NOT GNATMAKE_EXECUTABLE)
  message(STATUS "ADA: skipped, no gnatmake (GNAT Ada compiler) found")
  return()
endif()

message(STATUS "ADA: gnatmake found at ${GNATMAKE_EXECUTABLE}")

# prk_ada_executable(<name>)
# Compiles <name>.adb with gnatmake into ${CMAKE_CURRENT_BINARY_DIR}/<name>,
# keeping generated .o/.ali files in the build tree (gnatmake -D) rather
# than polluting the source directory. The custom target is named
# ada_<name>, not <name>: target names must be unique across the whole
# combined root project (every PRK language directory has its own p2p/
# stencil/etc.), same reasoning as every other language's CMakeLists.txt
# in this repo -- but since this is a custom target/command, not a real
# add_executable(), there's no OUTPUT_NAME to reconcile: the produced
# binary is simply named <name> directly.
function(prk_ada_executable name)
  set(_src "${CMAKE_CURRENT_LIST_DIR}/${name}.adb")
  set(_out "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${GNATMAKE_EXECUTABLE}" -D "${CMAKE_CURRENT_BINARY_DIR}" -p
            "${_src}" -o "${_out}"
    DEPENDS "${_src}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Ada ${name} with gnatmake")
  add_custom_target(ada_${name} ALL DEPENDS "${_out}")
endfunction()

function(prk_ada_test test_name)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(PAT "" "" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${test_name} COMMAND ${PAT_COMMAND})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

prk_ada_executable(nstream)
prk_ada_executable(p2p)
prk_ada_executable(stencil)
prk_ada_executable(transpose)

# --- CTest: same invocations used to verify each kernel by hand ---------

prk_ada_test(ada_nstream         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/nstream" 10 16000000)
prk_ada_test(ada_p2p             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/p2p" 10 1000 1000)
prk_ada_test(ada_transpose       COMMAND "${CMAKE_CURRENT_BINARY_DIR}/transpose" 10 1000)
prk_ada_test(ada_stencil_star    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/stencil" 10 1000)
prk_ada_test(ada_stencil_grid    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/stencil" 10 1000 grid)
prk_ada_test(ada_stencil_star_r4 COMMAND "${CMAKE_CURRENT_BINARY_DIR}/stencil" 10 1000 star 4)
