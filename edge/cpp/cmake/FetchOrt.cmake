# Pinned ONNX Runtime CPU builds. SHA256 is verified when SOUNDBUBBLE_ORT_SHA256 is set
# in this file for the active asset. Update both the version and the hash together.
set(SOUNDBUBBLE_ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime release tag")

if(WIN32)
  set(_ort_asset "onnxruntime-win-x64-${SOUNDBUBBLE_ORT_VERSION}.zip")
  set(_ort_sha256 "1c970f0c4b6c0c6a1b6a0a0e0c0e0c0e0c0e0c0e0c0e0c0e0c0e0c0e0c0e0c0e")
  set(_ort_sha_known FALSE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  set(_ort_asset "onnxruntime-linux-aarch64-${SOUNDBUBBLE_ORT_VERSION}.tgz")
  set(_ort_sha_known FALSE)
else()
  set(_ort_asset "onnxruntime-linux-x64-${SOUNDBUBBLE_ORT_VERSION}.tgz")
  set(_ort_sha_known FALSE)
endif()

set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${SOUNDBUBBLE_ORT_VERSION}/${_ort_asset}")
set(_ort_dir "${CMAKE_BINARY_DIR}/_deps/onnxruntime")

if(NOT EXISTS "${_ort_dir}/include/onnxruntime_cxx_api.h")
  message(STATUS "Fetching ONNX Runtime ${_ort_asset}")
  set(_archive "${CMAKE_BINARY_DIR}/_deps/${_ort_asset}")
  if(_ort_sha_known)
    file(DOWNLOAD "${_ort_url}" "${_archive}" SHOW_PROGRESS EXPECTED_HASH SHA256=${_ort_sha256} STATUS _st)
  else()
    file(DOWNLOAD "${_ort_url}" "${_archive}" SHOW_PROGRESS STATUS _st)
  endif()
  list(GET _st 0 _code)
  if(NOT _code EQUAL 0)
    message(WARNING "ONNX Runtime download failed (${_st}). Live/offline ONNX inference will be unavailable.")
    set(SOUNDBUBBLE_HAS_ORT FALSE)
    return()
  endif()
  file(SHA256 "${_archive}" _got)
  message(STATUS "ONNX Runtime archive SHA256=${_got}")
  file(MAKE_DIRECTORY "${_ort_dir}")
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
  file(GLOB _extracted "${CMAKE_BINARY_DIR}/_deps/onnxruntime-*")
  foreach(_d ${_extracted})
    if(IS_DIRECTORY "${_d}" AND NOT "${_d}" STREQUAL "${_ort_dir}")
      file(GLOB _kids "${_d}/*")
      file(COPY ${_kids} DESTINATION "${_ort_dir}")
    endif()
  endforeach()
endif()

if(EXISTS "${_ort_dir}/include/onnxruntime_cxx_api.h")
  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${_ort_dir}/lib/onnxruntime.lib"
      IMPORTED_LOCATION "${_ort_dir}/lib/onnxruntime.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${_ort_dir}/include")
    set(SOUNDBUBBLE_ORT_RUNTIME_DLL "${_ort_dir}/lib/onnxruntime.dll" CACHE INTERNAL "")
    # Providers shared DLL is optional but must match the same ORT build when present.
    if(EXISTS "${_ort_dir}/lib/onnxruntime_providers_shared.dll")
      set(SOUNDBUBBLE_ORT_PROVIDERS_DLL
        "${_ort_dir}/lib/onnxruntime_providers_shared.dll" CACHE INTERNAL "")
    endif()
  else()
    file(GLOB _ort_so "${_ort_dir}/lib/libonnxruntime.so*")
    list(GET _ort_so 0 _ort_so0)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${_ort_so0}"
      INTERFACE_INCLUDE_DIRECTORIES "${_ort_dir}/include")
    set(SOUNDBUBBLE_ORT_RUNTIME_LIB "${_ort_so0}" CACHE INTERNAL "")
  endif()
  set(SOUNDBUBBLE_HAS_ORT TRUE)
  set(SOUNDBUBBLE_ORT_ROOT "${_ort_dir}" CACHE INTERNAL "")
else()
  set(SOUNDBUBBLE_HAS_ORT FALSE)
endif()

# Windows loads System32\onnxruntime.dll (often older) unless the matching
# FetchOrt DLL sits beside the exe. Call after add_executable(...).
function(soundbubble_copy_ort_runtime tgt)
  if(NOT WIN32 OR NOT SOUNDBUBBLE_HAS_ORT OR NOT SOUNDBUBBLE_ORT_RUNTIME_DLL)
    return()
  endif()
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "soundbubble_copy_ort_runtime: unknown target ${tgt}")
  endif()
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${SOUNDBUBBLE_ORT_RUNTIME_DLL}"
      "$<TARGET_FILE_DIR:${tgt}>/onnxruntime.dll"
    COMMENT "Copy onnxruntime.dll beside ${tgt} (avoid System32 ORT mismatch)"
    VERBATIM)
  if(SOUNDBUBBLE_ORT_PROVIDERS_DLL)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SOUNDBUBBLE_ORT_PROVIDERS_DLL}"
        "$<TARGET_FILE_DIR:${tgt}>/onnxruntime_providers_shared.dll"
      COMMENT "Copy onnxruntime_providers_shared.dll beside ${tgt}"
      VERBATIM)
  endif()
endfunction()
