cmake_minimum_required(VERSION 3.10)
project(pig_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_DIR "D:/Users/INFO-H-503/Documents/NiccoloBenedetto/lib/eigen-3.4.1")

set(OpenCV_DIR "D:/Users/INFO-H-503/Documents/NiccoloBenedetto/lib/opencv/install")


# Dependencies
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/lib/opencv-4.12.0/build")
endif()


find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

# CUDA setup
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CUDA toolkit found — GPU acceleration enabled.")

# Get Sources recursively
#file(GLOB CPP_SOURCES "src/*.cpp")
#file(GLOB_RECURSE CUDA_SOURCES "src/gpu/*.cu")
#file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
# Get Sources
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/gpu/*.cu
)

# Safety filter (evita roba di CMake)
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/build/.*")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*_deps.*")

# Rimuovi roba dentro build/_deps
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*_deps.*")
file(GLOB_RECURSE CUDA_SOURCES "src/gpu/*.cu")

#list(APPEND CPP_SOURCES ${CPP_STAGE_SOURCES})

add_definitions(-DOPENCV_IO_ENABLE_OPENEXR=1)

#add_executable(${PROJECT_NAME} "main.cpp" ${CPP_SOURCES} ${CUDA_SOURCES})

add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CPP_SOURCES} 
    ${CUDA_SOURCES}
)

# CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${CMAKE_SOURCE_DIR}/lib/opencv-4.12.0/build/x64/vc16/bin"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Definitions and includes
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-warnings-as-errors=*")
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_DIR}
)



# To disable warnings on library builds.
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /external:W0>)

# Linking
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    nlohmann_json::nlohmann_json
    #Eigen3::Eigen
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
        $<$<COMPILE_LANGUAGE:CUDA>:--line-info>
    )
endif()
