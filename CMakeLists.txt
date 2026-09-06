cmake_minimum_required(VERSION 3.16)
project(opencl_filter LANGUAGES CXX)

# CPU reference + validation (no OpenCL required).
add_executable(ktgmc_cpu_ref sim/ktgmc_cpu_ref.cpp)
set_target_properties(ktgmc_cpu_ref PROPERTIES CXX_STANDARD 17)

# --- Optional: OpenCL host harness -----------------------------------------
# Only configured when an OpenCL implementation is available (find_package
# finds the headers + loader from e.g. ocl-icd-opencl-dev / pocl-opencl-icd).
# This sandbox has none, so the target is skipped by default.
# find_package(OpenCL QUIET)
# if(OpenCL_FOUND)
#   add_executable(ktgmc_opencl_host src/host/run_opencl.cpp)
#   target_link_libraries(ktgmc_opencl_host PRIVATE OpenCL::OpenCL)
# endif()

enable_testing()
add_test(NAME ktgmc_validation
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_validation.py)
add_test(NAME ktgmc_motion_core
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_motion_core.py)
add_test(NAME ktgmc_mv_aux
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_aux.py)
add_test(NAME ktgmc_mv_interp
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_interp.py)
add_test(NAME ktgmc_mv_mean
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_mean.py)
add_test(NAME ktgmc_mv_searchprep
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_searchprep.py)
