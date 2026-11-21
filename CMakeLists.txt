cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(HybridChessEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# No LibTorch any more: the NNUE and the CNN policy are evaluated in plain C++.
add_executable(engine src/main.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -march=native: build on the machine that runs the engine (AVX2 matters a lot).
    target_compile_options(engine PRIVATE -O3 -march=native -fno-math-errno)
    if(MINGW)
        # Self-contained .exe (no dependency on the MinGW runtime DLLs).
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
