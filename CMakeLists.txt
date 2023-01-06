cmake_minimum_required(VERSION 3.14)
project(dyros_practice_1)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
add_compile_options(-O2)

set(Torch_DIR "/home/jhk/libtorch/share/cmake/Torch/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  std_msgs
  eigen_conversions
)
find_package(Boost REQUIRED COMPONENTS system)

find_package(Torch REQUIRED PATH /usr/local/include/libtorch/share/cmake/Torch/)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ${PROJECT_NAME}
#  CATKIN_DEPENDS roscpp roslib rospy eigen_conversions
# DEPENDS system_lib
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include_directories(
  #  include
    ${catkin_INCLUDE_DIRS}
)


add_executable(dyros_practice src/dyros_practice_1.cpp)
add_dependencies(dyros_practice ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


target_link_libraries(dyros_practice
  ${catkin_LIBRARIES}
 -lcrocoddyl
 -lpinocchio
 ${TORCH_LIBRARIES}
)
