cmake_minimum_required(VERSION 3.10)
project(AkerReplay LANGUAGES C CXX)

# Builds the vendored Aker sources straight into two replay drivers, one per
# compile-time mode (standard, proximity). No shared library, so no RPATH issues.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(AKER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/aker" CACHE PATH "Vendored Aker")
set(BOOST_ROOT "" CACHE PATH "Boost root directory (optional)")
set(FAISS_ROOT "" CACHE PATH "FAISS installation prefix")
set(LIBOMP_ROOT "" CACHE PATH "libomp prefix (needed on macOS, where omp.h is not on the default path)")

find_package(Threads REQUIRED)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.80 REQUIRED COMPONENTS program_options log log_setup date_time thread
    system filesystem regex atomic chrono)
find_path(FAISS_INCLUDE_DIR NAMES faiss/Index.h
    PATHS "${FAISS_ROOT}/include" /usr/local/include /usr/include REQUIRED)
find_library(FAISS_LIBRARY NAMES faiss
    PATHS "${FAISS_ROOT}/lib" "${FAISS_ROOT}/lib64" /usr/local/lib /usr/lib /usr/lib64 REQUIRED)

if(LIBOMP_ROOT)
    set(OMP_INCLUDE "${LIBOMP_ROOT}/include")
    set(OMP_LIBS "${LIBOMP_ROOT}/lib/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    find_package(OpenMP REQUIRED)
    set(OMP_INCLUDE "")
    set(OMP_LIBS OpenMP::OpenMP_CXX)
endif()

file(GLOB_RECURSE AKER_SRC CONFIGURE_DEPENDS "${AKER_DIR}/src/*.c" "${AKER_DIR}/src/*.cc")

# The YCSB-C based key-sequence generator is not used (streams come from workloads).
list(FILTER AKER_SRC EXCLUDE REGEX "ak_sequence_gen")

function(add_replay name proximity)
    add_executable(${name} ak_replay.cc ${AKER_SRC} "${AKER_DIR}/extern/xxHash/xxhash.c")
    target_include_directories(${name} PRIVATE "${AKER_DIR}/inc" "${AKER_DIR}/extern"
        "${FAISS_INCLUDE_DIR}" ${OMP_INCLUDE})
    target_compile_definitions(${name} PRIVATE BOOST_LOG_DYN_LINK BOOST_LOG_SETUP_DYN_LINK
        AKER_ENABLE_PROXIMITY_MODE=${proximity} AKER_ENABLE_POTLUCK_MODE=0
        AKER_STANDARD_MODE=$<IF:$<BOOL:${proximity}>,0,1>)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-w>)
    target_link_libraries(${name} PRIVATE Threads::Threads ${FAISS_LIBRARY} ${OMP_LIBS}
        Boost::program_options Boost::log Boost::log_setup Boost::date_time Boost::thread
        Boost::system Boost::filesystem Boost::regex Boost::atomic Boost::chrono)
endfunction()

add_replay(ak_replay_standard 0)
add_replay(ak_replay_proximity 1)
