// generated from rosidl_generator_cpp/resource/rosidl_generator_cpp__visibility_control.hpp.in
// generated code does not contain a copyright notice

#ifndef CONE_DETECTION__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_
#define CONE_DETECTION__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_

#ifdef __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROSIDL_GENERATOR_CPP_EXPORT_cone_detection __attribute__ ((dllexport))
    #define ROSIDL_GENERATOR_CPP_IMPORT_cone_detection __attribute__ ((dllimport))
  #else
    #define ROSIDL_GENERATOR_CPP_EXPORT_cone_detection __declspec(dllexport)
    #define ROSIDL_GENERATOR_CPP_IMPORT_cone_detection __declspec(dllimport)
  #endif
  #ifdef ROSIDL_GENERATOR_CPP_BUILDING_DLL_cone_detection
    #define ROSIDL_GENERATOR_CPP_PUBLIC_cone_detection ROSIDL_GENERATOR_CPP_EXPORT_cone_detection
  #else
    #define ROSIDL_GENERATOR_CPP_PUBLIC_cone_detection ROSIDL_GENERATOR_CPP_IMPORT_cone_detection
  #endif
#else
  #define ROSIDL_GENERATOR_CPP_EXPORT_cone_detection __attribute__ ((visibility("default")))
  #define ROSIDL_GENERATOR_CPP_IMPORT_cone_detection
  #if __GNUC__ >= 4
    #define ROSIDL_GENERATOR_CPP_PUBLIC_cone_detection __attribute__ ((visibility("default")))
  #else
    #define ROSIDL_GENERATOR_CPP_PUBLIC_cone_detection
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // CONE_DETECTION__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_
