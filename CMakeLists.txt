cmake_minimum_required(VERSION 2.8.3)
project(icl_ros_segmentation)

find_package(cmake_modules REQUIRED)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  image_transport
  sensor_msgs
  pcl_conversions
  cv_bridge
  segmentation_msgs
)

find_package(ICL REQUIRED)
include_directories( ${ICL_INCLUDE_DIRS} )
add_definitions( ${ICL_DEFINITIONS} )

#find_package(PCL 1.7 REQUIRED COMPONENTS common io)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES icl_ros_bridge
  CATKIN_DEPENDS  image_transport sensor_msgs pcl_conversions cv_bridge segmentation_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a C++ library
# add_library(icl_ros_bridge
#   src/${PROJECT_NAME}/icl_ros_bridge.cpp
# )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++14")

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories(src)

set(LIB_SRC
	src/Kinect.cpp
	src/preSegmentation.cpp
  src/HighLevelObjectSegmentation.cpp
  src/SurfaceFeature.cpp
  src/CurvatureFeature.cpp
  src/CutfreeAdjacencyFeature.cpp
  src/FeatureGraphSegmenter.cpp
  src/PlanarRansac.cpp
  src/RemainingPointsFeature.cpp
  src/SegmenterHelper.cpp
  src/helper/icl_cv_helper.cpp
  src/GraphCut.cpp
  src/CoPlanarityFeature.cpp
	)
set(LIB_INC
	src/Kinect.h
	src/preSegmentation.h
  src/HighLevelObjectSegmentation.h
  src/SurfaceFeature.h
  src/CurvatureFeature.h
  src/CutfreeAdjacencyFeature.h
  src/FeatureGraphSegmenter.h
  src/PlanarRansac.h
  src/RemainingPointsFeature.h
  src/SegmenterHelper.h
  src/RegionGrowing.h
  src/helper/icl_cv_helper.h
  src/GraphCut.h
  src/CoPlanarityFeature.h
)

set(EXEC
  src/SurfaceFeature.h
  src/CurvatureFeature.h
  src/CutfreeAdjacencyFeature.h
  src/FeatureGraphSegmenter.h
  src/PlanarRansac.h
  src/RemainingPointsFeature.h
  src/SegmenterHelper.h
  src/RegionGrowing.h
  src/helper/icl_cv_helper.h
  src/GraphCut.h
  src/CoPlanarityFeature.h
  src/SurfaceFeature.cpp
  src/CurvatureFeature.cpp
  src/CutfreeAdjacencyFeature.cpp
  src/FeatureGraphSegmenter.cpp
  src/PlanarRansac.cpp
  src/RemainingPointsFeature.cpp
  src/SegmenterHelper.cpp
  src/helper/icl_cv_helper.cpp
  src/GraphCut.cpp
  src/CoPlanarityFeature.cpp
)

## Declare a C++ library
add_library( icl_ros_segmentation ${LIB_SRC} ${LIB_INC} )

target_link_libraries(icl_ros_segmentation
	${catkin_LIBRARIES}
	${ICL_LIBRARIES}
	${PCL_COMMON_LIBRARIES}
	${PCL_IO_LIBRARIES}
)

## Declare a C++ executable  
# NEED to add the configurabledepthimagesegmenter here, out of the lib, otherwise ICL version of this file is linked and used)
add_executable(icl_ros_segmentation_node src/icl_ros_segmentation_node.cpp ${EXEC} src/ConfigurableDepthImageSegmenter.cpp src/ConfigurableDepthImageSegmenter.h src/FeatureGraphSegmenter.cpp src/FeatureGraphSegmenter.h) 

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(icl_ros_segmentation_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(icl_ros_segmentation_node
  ${ICL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  icl_ros_segmentation
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Mark executables and/or libraries for installation
install(TARGETS icl_ros_segmentation_node icl_ros_segmentation
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#install(DIRECTORY launch/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )


#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_icl_ros_bridge.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
