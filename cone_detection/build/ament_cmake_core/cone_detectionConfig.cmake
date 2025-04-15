# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_cone_detection_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED cone_detection_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(cone_detection_FOUND FALSE)
  elseif(NOT cone_detection_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(cone_detection_FOUND FALSE)
  endif()
  return()
endif()
set(_cone_detection_CONFIG_INCLUDED TRUE)

# output package information
if(NOT cone_detection_FIND_QUIETLY)
  message(STATUS "Found cone_detection: 0.0.0 (${cone_detection_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'cone_detection' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${cone_detection_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(cone_detection_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${cone_detection_DIR}/${_extra}")
endforeach()
