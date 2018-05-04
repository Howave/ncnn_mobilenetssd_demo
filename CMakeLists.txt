cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pie -fPIE -fPIC -fopenmp -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")

set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -Wl,--gc-sections")
set(CMAKE_LDFLAGS "${CMAKE_LDFLAGS} -fopenmp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
add_executable(ssdmobilenet ssdmobilenet.cpp)
target_link_libraries(ssdmobilenet ncnn ${OpenCV_LIBS})
