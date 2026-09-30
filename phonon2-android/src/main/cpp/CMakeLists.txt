cmake_minimum_required(VERSION 3.22.1)
project("phonon2_android")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture specific compiler flags
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_definitions(-DARCH_ARM64=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
elseif(ANDROID_ABI STREQUAL "x86_64")
    add_definitions(-DARCH_X86_64=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -msse4.2 -fPIC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
endif()

add_library(
    phonon2_android
    SHARED
    phonon2_runtime.cpp
)

find_library(log-lib log)

target_link_libraries(
    phonon2_android
    ${log-lib}
)
