cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(HybridChessEngine C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ON: optimise for this machine's CPU (fastest). OFF: portable build that runs on any
# x86-64 CPU with AVX2 (Haswell / 2013+), e.g. for CI or sharing the binary.
option(ENGINE_NATIVE "Optimise for the CPU that builds the engine" ON)

# No LibTorch any more: the NNUE and the CNN policy are evaluated in plain C++.
add_executable(engine src/main.cpp)

# Fathom: Syzygy tablebase probing (MIT, include/fathom).
add_library(fathom STATIC include/fathom/tbprobe.c)
set_target_properties(fathom PROPERTIES C_STANDARD 11)
# Fathom includes <tbconfig.h> with angle brackets.
target_include_directories(fathom PUBLIC include/fathom)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fathom PRIVATE -O3 -w)
    if(ENGINE_NATIVE)
        target_compile_options(fathom PRIVATE -march=native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_compile_options(fathom PRIVATE -march=x86-64-v3)
    endif()
endif()
target_link_libraries(engine PRIVATE fathom)

find_package(Threads REQUIRED)
target_link_libraries(engine PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(engine PRIVATE -O3 -fno-math-errno)
    if(ENGINE_NATIVE)
        target_compile_options(engine PRIVATE -march=native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_compile_options(engine PRIVATE -march=x86-64-v3)
    endif()
    # Profile-guided build (scripts/build_pgo.py): GEN builds an instrumented engine that
    # writes profiles to ENGINE_PGO_DIR, USE rebuilds with them.
    set(ENGINE_PGO "" CACHE STRING "Profile-guided optimisation: empty, GEN or USE")
    set(ENGINE_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE PATH "Where the PGO profiles go")
    if(ENGINE_PGO STREQUAL "GEN")
        target_compile_options(engine PRIVATE -fprofile-generate=${ENGINE_PGO_DIR})
        target_link_options(engine PRIVATE -fprofile-generate=${ENGINE_PGO_DIR})
    elseif(ENGINE_PGO STREQUAL "USE")
        target_compile_options(engine PRIVATE -fprofile-use=${ENGINE_PGO_DIR} -fprofile-correction -Wno-missing-profile)
        target_link_options(engine PRIVATE -fprofile-use=${ENGINE_PGO_DIR})
    endif()
    if(MINGW)
        # Bundle the runtime so engine.exe also runs outside CLion: static
        # libgcc/libstdc++, and copy the one remaining DLL (winpthread) next to it.
        target_link_options(engine PRIVATE -static-libgcc -static-libstdc++)
        get_filename_component(MINGW_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
        if(EXISTS "${MINGW_BIN}/libwinpthread-1.dll")
            add_custom_command(TARGET engine POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MINGW_BIN}/libwinpthread-1.dll" "$<TARGET_FILE_DIR:engine>")
        endif()
    endif()
elseif(MSVC)
    target_compile_options(engine PRIVATE /O2 /arch:AVX2)
endif()
