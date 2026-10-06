include(FetchContent)

# SketchSSM CUDA kernels, vendored as vllm.third_party.sketchssm_kernels.
# SKETCHSSM_SRC_DIR (env or cmake) selects a local SketchSSM checkout.
if(DEFINED ENV{SKETCHSSM_SRC_DIR})
  set(SKETCHSSM_SRC_DIR $ENV{SKETCHSSM_SRC_DIR})
endif()

set(SKETCHSSM_GIT_REPOSITORY "https://github.com/SNU-ARC/SketchSSM.git")
set(SKETCHSSM_GIT_TAG "v0.1.12")
set(SKETCHSSM_KERNELS_SUBDIR "sketchssm/kernels")
set(SKETCHSSM_KERNELS_DEST "vllm/third_party/sketchssm_kernels")

if(SKETCHSSM_SRC_DIR)
  set(_sketchssm_user_src "${SKETCHSSM_SRC_DIR}")
  cmake_path(ABSOLUTE_PATH _sketchssm_user_src
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    NORMALIZE)
  if(NOT IS_DIRECTORY "${_sketchssm_user_src}")
    message(FATAL_ERROR
      "SKETCHSSM_SRC_DIR is not an existing directory: '${_sketchssm_user_src}'")
  endif()
  set(sketchssm_SOURCE_DIR "${_sketchssm_user_src}")
else()
  set(_sketchssm_fc_root "${FETCHCONTENT_BASE_DIR}")
  if(NOT _sketchssm_fc_root)
    set(_sketchssm_fc_root "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(sketchssm_SOURCE_DIR "${_sketchssm_fc_root}/sketchssm-src")
  if(NOT EXISTS "${sketchssm_SOURCE_DIR}/${SKETCHSSM_KERNELS_SUBDIR}/__init__.py")
    FetchContent_Populate(
      sketchssm
      SUBBUILD_DIR "${_sketchssm_fc_root}/sketchssm-subbuild"
      SOURCE_DIR "${sketchssm_SOURCE_DIR}"
      BINARY_DIR "${_sketchssm_fc_root}/sketchssm-build"
      GIT_REPOSITORY "${SKETCHSSM_GIT_REPOSITORY}"
      GIT_TAG "${SKETCHSSM_GIT_TAG}"
      GIT_PROGRESS TRUE
    )
  endif()
endif()
message(STATUS "SketchSSM is available at ${sketchssm_SOURCE_DIR}")

set(_sketchssm_pkg "${sketchssm_SOURCE_DIR}/${SKETCHSSM_KERNELS_SUBDIR}")

file(GLOB _sketchssm_py "${_sketchssm_pkg}/*.py")
list(FILTER _sketchssm_py EXCLUDE REGEX "/build\\.py$")
install(FILES ${_sketchssm_py}
  DESTINATION ${SKETCHSSM_KERNELS_DEST}
  COMPONENT sketchssm_kernels)
install(DIRECTORY "${_sketchssm_pkg}/csrc" "${_sketchssm_pkg}/configs"
  DESTINATION ${SKETCHSSM_KERNELS_DEST}
  COMPONENT sketchssm_kernels
  PATTERN "__pycache__" EXCLUDE)

set(SKETCHSSM_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0)
  list(APPEND SKETCHSSM_SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND SKETCHSSM_SUPPORT_ARCHS "10.0f")
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
  list(APPEND SKETCHSSM_SUPPORT_ARCHS "10.0a" "10.3a")
endif()

cuda_archs_loose_intersection(
  SKETCHSSM_ARCHS "${SKETCHSSM_SUPPORT_ARCHS}" "${CUDA_ARCHS}")

set(_sketchssm_build_py "${_sketchssm_pkg}/build.py")
if(SKETCHSSM_ARCHS AND EXISTS "${_sketchssm_build_py}")
  message(STATUS "SketchSSM AOT kernel architectures: ${SKETCHSSM_ARCHS}")
  set(_sketchssm_aot "${CMAKE_CURRENT_BINARY_DIR}/sketchssm_aot")
  string(REPLACE ";" "$<SEMICOLON>" _sketchssm_arch_arg "${SKETCHSSM_ARCHS}")
  file(GLOB_RECURSE _sketchssm_srcs
    "${_sketchssm_pkg}/*.py" "${_sketchssm_pkg}/csrc/*")
  add_custom_command(
    OUTPUT "${_sketchssm_aot}/.built"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_sketchssm_aot}"
    COMMAND "${Python_EXECUTABLE}" "${_sketchssm_build_py}"
            --arch "${_sketchssm_arch_arg}" --out "${_sketchssm_aot}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_sketchssm_aot}/.built"
    DEPENDS ${_sketchssm_srcs}
    COMMENT "Building SketchSSM AOT kernels for ${SKETCHSSM_ARCHS}"
    VERBATIM)
  add_custom_target(sketchssm_kernels ALL DEPENDS "${_sketchssm_aot}/.built")
  install(DIRECTORY "${_sketchssm_aot}/"
    DESTINATION ${SKETCHSSM_KERNELS_DEST}/aot
    COMPONENT sketchssm_kernels
    PATTERN ".built" EXCLUDE)
else()
  message(STATUS "SketchSSM AOT kernels will not be built; "
    "the vendored kernels compile at runtime")
  add_custom_target(sketchssm_kernels)
endif()
