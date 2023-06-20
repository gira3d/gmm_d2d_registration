# - Try to find gmm_d2d_registration header files and libraries
#
# Once done this will define
#
# GMM D2D Registration

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
set(gmm_d2d_registration_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/gmm_d2d_registration/include")
