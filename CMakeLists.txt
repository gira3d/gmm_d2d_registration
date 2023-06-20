cmake_minimum_required(VERSION 3.13)

project(gmm_d2d_registration
	VERSION 0.0.0
	DESCRIPTION
	"GMM Registration"
	LANGUAGES CXX C)

find_package(dlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gmm REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-march=native -ffast-math")
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

find_package(Eigen3 REQUIRED)

include_directories(include
		    ${dlib_INCLUDE_DIR}
		    ${gmm_INCLUDE_DIR}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
)

install(DIRECTORY cmake/
  DESTINATION "share/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.cmake"
)
