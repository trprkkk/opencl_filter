cmake_minimum_required(VERSION 3.16)
project(opencl_filter LANGUAGES CXX)

# CPU reference + validation (no OpenCL required).
add_executable(ktgmc_cpu_ref sim/ktgmc_cpu_ref.cpp)
set_target_properties(ktgmc_cpu_ref PROPERTIES CXX_STANDARD 17)

# --- Optional: OpenCL host harness -----------------------------------------
# Only configured when an OpenCL implementation is available (find_package
# finds the headers + loader from e.g. ocl-icd-opencl-dev / pocl-opencl-icd).
# This sandbox has none, so the target is skipped by default.
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
  add_executable(ktgmc_opencl_host src/host/run_opencl.cpp)
  target_link_libraries(ktgmc_opencl_host PRIVATE OpenCL::OpenCL)
  set_target_properties(ktgmc_opencl_host PROPERTIES CXX_STANDARD 17)
  # Bring-up smoke test: builds both .cl at both PX instantiations on a rig.
  # (Needs a device ICD; not run in CI here.)
  add_test(NAME ktgmc_cl_smoke
           COMMAND ktgmc_opencl_host
                   ${CMAKE_SOURCE_DIR}/src/opencl/ktgmc/kernels/ktgmc_simple.cl
                   ${CMAKE_SOURCE_DIR}/src/opencl/ktgmc/kernels/ktgmc_motion.cl)
  # AvsCUDA Conditional rig harness (docs/RIG_HANDOFF_AVSCUDA_CONDITIONAL.md
  # §5): dispatches ka_sum_pixels_f32 / ka_sad_f32 per the §4.4 launch
  # contract and prints counter bits + f64 reference. Device-required, so no
  # add_test here (the handoff's Phase A/B driver invokes it explicitly).
  add_executable(avscuda_rig_host src/host/run_avscuda_rig.cpp)
  target_link_libraries(avscuda_rig_host PRIVATE OpenCL::OpenCL)
  set_target_properties(avscuda_rig_host PROPERTIES CXX_STANDARD 17)
endif()

# --- Optional: GRunT (verbatim upstream CPU plugin, Windows-only) ------------
# GRunT has no CUDA upstream, so it is vendored as-is (third_party/grunt/)
# instead of ported. Builds only on Windows with AviSynthPlus headers.
# Untested here (no Windows, no cmake in this sandbox).
if(WIN32)
  find_path(AVISYNTH_INCLUDE_DIR avisynth.h
            PATHS "$ENV{AVISYNTH_SDK}/include"
            DOC "AviSynthPlus headers (for the GRunT plugin)")
  if(AVISYNTH_INCLUDE_DIR)
    add_library(GRunT SHARED third_party/grunt/GRunT.cpp)
    target_include_directories(GRunT PRIVATE ${AVISYNTH_INCLUDE_DIR})
    set_target_properties(GRunT PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "GRunT")
  endif()
endif()

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
add_test(NAME ktgmc_mv_io
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_io.py)
add_test(NAME ktgmc_mv_rb2b
         COMMAND ${CMAKE_COMMAND} -E env
                 python3 ${CMAKE_SOURCE_DIR}/python/run_mv_rb2b.py)
# Structural parse lint of the .cl kernels (real C parse via lint/oc_shim.h;
# gcc would otherwise treat .cl as a linker input and skip it). Needs gcc/bash.
add_test(NAME ktgmc_cl_lint
         COMMAND bash ${CMAKE_SOURCE_DIR}/lint/lint_opencl.sh)
