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

# --- Optional: KEdgeLevelOCL — first real AviSynth+ host plugin --------------
# Windows-only (like the cl.exe recipe in src/host/plugin/build_host_edgelevel.bat):
# builds the plugin DLL + its headless verification exe from the vendored
# Khronos headers/loader-import-lib (Q:\Downloads\openclfilter\sdk\opencl) and
# the AviSynth+ header (Q:\Downloads\openclfilter\sdk\avisynth.h).  The kernel
# source is embedded at configure/build time by embed_cl.py.  Device-required
# at run time, so no add_test.
if(WIN32 AND OpenCL_FOUND AND AVISYNTH_INCLUDE_DIR)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  set(KEL_PLUGIN_DIR ${CMAKE_SOURCE_DIR}/src/host/plugin)
  set(KEL_CL_EMBEDDED ${KEL_PLUGIN_DIR}/kfm_edgelevel_cl.h)
  if(Python3_FOUND)
    add_custom_command(
      OUTPUT ${KEL_CL_EMBEDDED}
      COMMAND ${Python3_EXECUTABLE} ${KEL_PLUGIN_DIR}/embed_cl.py
      DEPENDS ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_edgelevel.cl
      WORKING_DIRECTORY ${KEL_PLUGIN_DIR}
      COMMENT "Embedding kfm_edgelevel.cl")
  endif()
  add_library(kfm_edgelevel_avs SHARED
              ${KEL_PLUGIN_DIR}/kfm_edgelevel_avs.cpp
              src/host/ocl_host.cpp
              ${KEL_CL_EMBEDDED})
  target_include_directories(kfm_edgelevel_avs PRIVATE
                             ${CMAKE_SOURCE_DIR}/src/host ${AVISYNTH_INCLUDE_DIR})
  target_link_libraries(kfm_edgelevel_avs PRIVATE OpenCL::OpenCL)
  set_target_properties(kfm_edgelevel_avs PROPERTIES CXX_STANDARD 17
                        OUTPUT_NAME "kfm_edgelevel_avs")
  add_executable(test_kfm_edgelevel ${KEL_PLUGIN_DIR}/test_kfm_edgelevel.cpp)
  target_include_directories(test_kfm_edgelevel PRIVATE ${AVISYNTH_INCLUDE_DIR})
  set_target_properties(test_kfm_edgelevel PROPERTIES CXX_STANDARD 17)

  # --- KTemporalNROCL — second AviSynth+ host plugin (temporal, stateless
  # windowed over the child clip; see the plugin header's state model) -------
  set(KTN_CL_EMBEDDED ${KEL_PLUGIN_DIR}/kfm_temporalnr_cl.h)
  if(Python3_FOUND)
    add_custom_command(
      OUTPUT ${KTN_CL_EMBEDDED}
      COMMAND ${Python3_EXECUTABLE} ${KEL_PLUGIN_DIR}/embed_cl.py
              ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_temporalnr.cl
              ${KTN_CL_EMBEDDED} kfm_temporalnr_cl_src
      DEPENDS ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_temporalnr.cl
      WORKING_DIRECTORY ${KEL_PLUGIN_DIR}
      COMMENT "Embedding kfm_temporalnr.cl")
  endif()
  add_library(kfm_temporalnr_avs SHARED
              ${KEL_PLUGIN_DIR}/kfm_temporalnr_avs.cpp
              src/host/ocl_host.cpp
              ${KTN_CL_EMBEDDED})
  target_include_directories(kfm_temporalnr_avs PRIVATE
                             ${CMAKE_SOURCE_DIR}/src/host ${AVISYNTH_INCLUDE_DIR})
  target_link_libraries(kfm_temporalnr_avs PRIVATE OpenCL::OpenCL)
  set_target_properties(kfm_temporalnr_avs PROPERTIES CXX_STANDARD 17
                        OUTPUT_NAME "kfm_temporalnr_avs")
  add_executable(test_kfm_temporalnr ${KEL_PLUGIN_DIR}/test_kfm_temporalnr.cpp)
  target_include_directories(test_kfm_temporalnr PRIVATE ${AVISYNTH_INCLUDE_DIR})
  set_target_properties(test_kfm_temporalnr PROPERTIES CXX_STANDARD 17)

  # --- KDebandOCL — third AviSynth+ host plugin (single-pass random-sampling
  # debander; shared luma-dims XorShift rand stream, see the plugin header) --
  set(KDB_CL_EMBEDDED ${KEL_PLUGIN_DIR}/kfm_deband_cl.h)
  if(Python3_FOUND)
    add_custom_command(
      OUTPUT ${KDB_CL_EMBEDDED}
      COMMAND ${Python3_EXECUTABLE} ${KEL_PLUGIN_DIR}/embed_cl.py
              ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_deband.cl
              ${KDB_CL_EMBEDDED} kfm_deband_cl_src
      DEPENDS ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_deband.cl
      WORKING_DIRECTORY ${KEL_PLUGIN_DIR}
      COMMENT "Embedding kfm_deband.cl")
  endif()
  add_library(kfm_deband_avs SHARED
              ${KEL_PLUGIN_DIR}/kfm_deband_avs.cpp
              src/host/ocl_host.cpp
              ${KDB_CL_EMBEDDED})
  target_include_directories(kfm_deband_avs PRIVATE
                             ${CMAKE_SOURCE_DIR}/src/host ${AVISYNTH_INCLUDE_DIR})
  target_link_libraries(kfm_deband_avs PRIVATE OpenCL::OpenCL)
  set_target_properties(kfm_deband_avs PROPERTIES CXX_STANDARD 17
                        OUTPUT_NAME "kfm_deband_avs")
  add_executable(test_kfm_deband ${KEL_PLUGIN_DIR}/test_kfm_deband.cpp)
  target_include_directories(test_kfm_deband PRIVATE ${AVISYNTH_INCLUDE_DIR})
  set_target_properties(test_kfm_deband PROPERTIES CXX_STANDARD 17)

  # --- KTemporalDiffOCL + KMergeStaticOCL — fourth AviSynth+ host plugin
  # (both filters of upstream KFM MergeStatic.cu; stateless cross-frame clip
  # mapping, see the plugin header's state model) ----------------------------
  set(KMS_CL_EMBEDDED ${KEL_PLUGIN_DIR}/kfm_mergestatic_cl.h)
  if(Python3_FOUND)
    add_custom_command(
      OUTPUT ${KMS_CL_EMBEDDED}
      COMMAND ${Python3_EXECUTABLE} ${KEL_PLUGIN_DIR}/embed_cl.py
              ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_mergestatic.cl
              ${KMS_CL_EMBEDDED} kfm_mergestatic_cl_src
      DEPENDS ${CMAKE_SOURCE_DIR}/src/opencl/kfm/kernels/kfm_mergestatic.cl
      WORKING_DIRECTORY ${KEL_PLUGIN_DIR}
      COMMENT "Embedding kfm_mergestatic.cl")
  endif()
  add_library(kfm_mergestatic_avs SHARED
              ${KEL_PLUGIN_DIR}/kfm_mergestatic_avs.cpp
              src/host/ocl_host.cpp
              ${KMS_CL_EMBEDDED})
  target_include_directories(kfm_mergestatic_avs PRIVATE
                             ${CMAKE_SOURCE_DIR}/src/host ${AVISYNTH_INCLUDE_DIR})
  target_link_libraries(kfm_mergestatic_avs PRIVATE OpenCL::OpenCL)
  set_target_properties(kfm_mergestatic_avs PROPERTIES CXX_STANDARD 17
                        OUTPUT_NAME "kfm_mergestatic_avs")
  add_executable(test_kfm_mergestatic ${KEL_PLUGIN_DIR}/test_kfm_mergestatic.cpp)
  target_include_directories(test_kfm_mergestatic PRIVATE ${AVISYNTH_INCLUDE_DIR})
  set_target_properties(test_kfm_mergestatic PROPERTIES CXX_STANDARD 17)
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
