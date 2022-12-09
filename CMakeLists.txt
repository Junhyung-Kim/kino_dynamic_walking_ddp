cmake_minimum_required(VERSION 3.14)
project(dyros_practice_1)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  std_msgs
  eigen_conversions
)

find_package(Boost REQUIRED COMPONENTS system)

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
)

#install(TARGETS ${PROJECT_NAME}
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
#)

#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#  FILES_MATCHING PATTERN "*.h"
#  PATTERN ".svn" EXCLUDE
#)
