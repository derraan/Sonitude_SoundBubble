cmake_minimum_required(VERSION 3.24)
project(soundbubble_rt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SOUNDBUBBLE_BUILD_TESTS "Build gtest suite" ON)
option(SOUNDBUBBLE_WITH_ORT "Fetch and link ONNX Runtime" ON)
option(SOUNDBUBBLE_WITH_SHERPA "Fetch and link sherpa-onnx" OFF)
option(SOUNDBUBBLE_WITH_PORTAUDIO "Fetch and link PortAudio" ON)
option(SOUNDBUBBLE_WITH_SOXR "Fetch and link libsoxr" ON)
option(SOUNDBUBBLE_WITH_SENTENCEPIECE "Link sentencepiece when available" OFF)

if(MSVC)
  add_compile_options(/W3 /EHsc /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

include(FetchContent)
set(SOUNDBUBBLE_HAS_ORT FALSE)
set(SOUNDBUBBLE_HAS_SHERPA FALSE)
set(SOUNDBUBBLE_HAS_PORTAUDIO FALSE)
set(SOUNDBUBBLE_HAS_SOXR FALSE)
set(SOUNDBUBBLE_HAS_SP FALSE)

if(SOUNDBUBBLE_WITH_SOXR)
  find_path(SOXR_INCLUDE_DIR soxr.h)
  find_library(SOXR_LIBRARY NAMES soxr libsoxr)
  if(SOXR_INCLUDE_DIR AND SOXR_LIBRARY)
    add_library(soxr UNKNOWN IMPORTED GLOBAL)
    set_target_properties(soxr PROPERTIES IMPORTED_LOCATION "${SOXR_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${SOXR_INCLUDE_DIR}")
    set(SOUNDBUBBLE_HAS_SOXR TRUE)
  else()
    message(STATUS "libsoxr not found; streaming SRC uses the linear fallback until libsoxr is installed (apt install libsoxr-dev, or vcpkg install soxr).")
  endif()
endif()

if(SOUNDBUBBLE_WITH_PORTAUDIO)
  set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(portaudio
    URL https://github.com/PortAudio/portaudio/archive/refs/tags/v19.7.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(portaudio)
  if(TARGET portaudio)
    set(SOUNDBUBBLE_HAS_PORTAUDIO TRUE)
  elseif(TARGET PortAudio)
    add_library(portaudio ALIAS PortAudio)
    set(SOUNDBUBBLE_HAS_PORTAUDIO TRUE)
  endif()
endif()

if(SOUNDBUBBLE_WITH_ORT)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/FetchOrt.cmake)
endif()
if(SOUNDBUBBLE_WITH_SHERPA)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/FetchSherpa.cmake)
endif()

if(SOUNDBUBBLE_WITH_SENTENCEPIECE)
  find_package(sentencepiece CONFIG QUIET)
  if(sentencepiece_FOUND OR TARGET sentencepiece)
    set(SOUNDBUBBLE_HAS_SP TRUE)
  endif()
endif()

add_library(soundbubble_core STATIC
  src/realtime.cpp
  src/scheduling.cpp
  src/processing.cpp
  src/limiter.cpp
  src/contract.cpp
  src/policy.cpp
  src/pipeline.cpp
  src/onnx_streamer.cpp
  src/wake.cpp
  src/kws.cpp
  src/mixctl.cpp
  src/audio_io.cpp
  src/controller.cpp
  src/wav.cpp
)
target_include_directories(soundbubble_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(soundbubble_core PUBLIC
  $<$<BOOL:${SOUNDBUBBLE_HAS_ORT}>:SOUNDBUBBLE_HAS_ORT>
  $<$<BOOL:${SOUNDBUBBLE_HAS_SHERPA}>:SOUNDBUBBLE_HAS_SHERPA>
  $<$<BOOL:${SOUNDBUBBLE_HAS_PORTAUDIO}>:SOUNDBUBBLE_HAS_PORTAUDIO>
  $<$<BOOL:${SOUNDBUBBLE_HAS_SOXR}>:SOUNDBUBBLE_HAS_SOXR>
  $<$<BOOL:${SOUNDBUBBLE_HAS_SP}>:SOUNDBUBBLE_HAS_SP>
)
if(SOUNDBUBBLE_HAS_SOXR)
  target_link_libraries(soundbubble_core PUBLIC soxr)
endif()
if(SOUNDBUBBLE_HAS_PORTAUDIO)
  if(TARGET portaudio_static)
    target_link_libraries(soundbubble_core PUBLIC portaudio_static)
  else()
    target_link_libraries(soundbubble_core PUBLIC portaudio)
  endif()
endif()
if(SOUNDBUBBLE_HAS_ORT)
  target_link_libraries(soundbubble_core PUBLIC onnxruntime)
endif()
if(SOUNDBUBBLE_HAS_SHERPA)
  target_link_libraries(soundbubble_core PUBLIC sherpa-onnx)
endif()
if(SOUNDBUBBLE_HAS_SP AND TARGET sentencepiece)
  target_link_libraries(soundbubble_core PUBLIC sentencepiece)
endif()
if(WIN32)
  target_link_libraries(soundbubble_core PUBLIC ws2_32 winmm avrt)
  if(NOT MSVC)
    # MinGW does not treat the ONNX Runtime _stdcall token as a calling convention.
    target_compile_definitions(soundbubble_core PUBLIC _stdcall=__stdcall)
  endif()
else()
  target_link_libraries(soundbubble_core PUBLIC pthread)
endif()

add_executable(soundbubble-rt apps/soundbubble_rt.cpp)
target_link_libraries(soundbubble-rt PRIVATE soundbubble_core)
set_target_properties(soundbubble-rt PROPERTIES OUTPUT_NAME soundbubble-rt)

add_executable(soundbubble-ctl apps/soundbubble_ctl.cpp)
target_link_libraries(soundbubble-ctl PRIVATE soundbubble_core)
set_target_properties(soundbubble-ctl PROPERTIES OUTPUT_NAME soundbubble-ctl)

add_executable(soundbubble-hello-ort apps/hello_ort.cpp)
target_link_libraries(soundbubble-hello-ort PRIVATE soundbubble_core)

add_executable(soundbubble-mix-listen apps/mix_listen.cpp)
target_link_libraries(soundbubble-mix-listen PRIVATE soundbubble_core)

if(SOUNDBUBBLE_BUILD_TESTS)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(soundbubble_tests
    tests/test_realtime.cpp
    tests/test_policy.cpp
    tests/test_processing.cpp
    tests/test_contract.cpp
    tests/test_wake.cpp
    tests/test_mixctl.cpp
    tests/test_profile.cpp
    tests/test_controller.cpp
    tests/test_pipeline_offline.cpp
  )
  target_link_libraries(soundbubble_tests PRIVATE soundbubble_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(soundbubble_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 180)
endif()

install(TARGETS soundbubble-rt soundbubble-ctl soundbubble-hello-ort soundbubble-mix-listen
  RUNTIME DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/run.sh DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sonitude-soundbubble.service
  ${CMAKE_CURRENT_SOURCE_DIR}/packaging/soundbubble-controller.service
  DESTINATION lib/systemd/system)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sonitude-soundbubble.env
  DESTINATION /etc
  RENAME sonitude-soundbubble.env)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/packaging/postinst DESTINATION lib/soundbubble)

if(SOUNDBUBBLE_HAS_ORT AND SOUNDBUBBLE_ORT_ROOT)
  install(DIRECTORY ${SOUNDBUBBLE_ORT_ROOT}/lib/ DESTINATION lib/soundbubble
    FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll" PATTERN "*.dylib")
endif()
if(SOUNDBUBBLE_HAS_SHERPA AND SOUNDBUBBLE_SHERPA_ROOT)
  install(DIRECTORY ${SOUNDBUBBLE_SHERPA_ROOT}/lib/ DESTINATION lib/soundbubble
    FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll")
endif()

set(CPACK_PACKAGE_NAME "soundbubble-rt")
set(CPACK_PACKAGE_VENDOR "Sonitude")
set(CPACK_PACKAGE_CONTACT "sonitude")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libasound2")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/packaging/postinst")
if(WIN32)
  set(CPACK_GENERATOR "ZIP")
else()
  set(CPACK_GENERATOR "DEB")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
  else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  endif()
endif()
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/soundbubble")
include(CPack)
