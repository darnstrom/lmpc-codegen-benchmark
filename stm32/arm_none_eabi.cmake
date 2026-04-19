# CMake toolchain file for arm-none-eabi cross-compilation.
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=<path>/arm_none_eabi.cmake ..

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

# Resolve the toolchain prefix from an env variable or fall back to the
# standard name on PATH.
if(DEFINED ENV{TOOLCHAIN_PREFIX})
    set(_TC "$ENV{TOOLCHAIN_PREFIX}")
else()
    set(_TC "arm-none-eabi-")
endif()

find_program(CMAKE_C_COMPILER   "${_TC}gcc"      REQUIRED)
find_program(CMAKE_CXX_COMPILER "${_TC}g++"      REQUIRED)
find_program(CMAKE_ASM_COMPILER "${_TC}gcc"      REQUIRED)
find_program(CMAKE_OBJCOPY      "${_TC}objcopy"  REQUIRED)
find_program(CMAKE_SIZE_UTIL    "${_TC}size"     REQUIRED)

# Prevent CMake from testing the bare compiler (it would fail for cross builds).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
