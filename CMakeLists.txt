cmake_minimum_required(VERSION 3.10)
project(TestCUDA VERSION 1.0)
enable_language(CXX)
if(NOT APPLE)
    enable_language(CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "cuda compiler: " ${CMAKE_CUDA_COMPILER})
    message(STATUS "cuda compiler id: " ${CMAKE_CUDA_COMPILER})
    message(STATUS "cuda compiler version: " ${CMAKE_CUDA_COMPILER_VERSION})
    message(STATUS "cuda include dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    # cuda CXX version
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    find_package(CUDAToolkit)
    # if(CUDAToolkit_FOUND)
    #     target_link_libraries(YourProjectName PRIVATE CUDA::cublas)
    # endif()

endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_DEBUG_POSTFIX "d")
include_directories("${PROJECT_SOURCE_DIR}/includes")
include_directories("${PROJECT_SOURCE_DIR}")

if(WIN32)
    # add_definitions(-D _USE_MATH_DEFINES)
    # add_definitions(-D _CRT_NO_VA_START_VALIDATION)
    ###### set cpp flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

    ###### set cuda flags
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G  -Xcompiler=\"-MDd -O0 -Ob0\" ")
    set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=\"-MD -O2 -Ob2\" -DNDEBUG")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4061 /wd4244 /wd4246 /wd4305 /wd4267 /wd4711 /wd4710 /wd4514 /wd4477")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Zi")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    # use sanitzer
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
    # add_definitions("/fsanitize=address")
else()
    # add_definitions(-std=c++17)
    # message(FATAL_ERROR add unix c++17 definitions)
    # ---------------------enable debug symbols---------------------
    add_definitions(-g)
    set(CMAKE_CUDA_FLAGS_DEBUG "-G")
endif()


set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

message("CMAKE_INSTALL_PREFIX = " ${CMAKE_INSTALL_PREFIX})
add_subdirectory(./gpu_utils)
add_subdirectory(./utils)
add_subdirectory(./geometries)

add_library(
    net_lib
    FullConnectedNetSingleScalar.cpp
    FullConnectedNetSingleScalarGPU.cpp
    FullConnectedNetSingleScalarGPU.cu
    FullConnectedNetSingleScalarGPU2.cu
    build_net.cpp
    test_gpu_net.cpp
    bake_net.cpp
    load_bake_net.cpp
)
target_link_libraries(net_lib PUBLIC utils_lib cuda_lib)

# target_compile_options(net_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w -maxrregcount=32>)
target_compile_options(net_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)

add_executable(main ./main.cpp)
add_executable(reduction ./gpu_reduction.cpp gpu_reduction.cu)

target_link_libraries(main PUBLIC cuda_lib net_lib CUDA::cudart CUDA::cublas)
target_link_libraries(reduction PUBLIC cuda_lib CUDA::cudart CUDA::cublas)

install(TARGETS main RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})