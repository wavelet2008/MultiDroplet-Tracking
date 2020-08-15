cmake_minimum_required(VERSION 2.8)

set(PROJ MultiDropletTracking)
set(CMAKE_CXX_STANDARD 11)
project(${PROJ})

set(OpenCV_DIR "../opencv/MinGWbuild")
find_package(OpenCV 4 REQUIRED)

add_subdirectory(camera)
add_subdirectory(preprocessing)
add_subdirectory(detection)
add_subdirectory(tracking)

add_library(DropletTracking_LIB multidroplet_tracking.cpp)
target_link_libraries(DropletTracking_LIB ${OpenCV_LIBS} Camera_LIB PreProcessing_LIB Detection_LIB Tracking_LIB)

add_executable(${PROJ} main.cpp multidroplet_tracking.cpp)
target_link_libraries(${PROJ} DropletTracking_LIB)