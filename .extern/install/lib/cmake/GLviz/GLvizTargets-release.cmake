#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GLviz::glviz" for configuration "Release"
set_property(TARGET GLviz::glviz APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GLviz::glviz PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/glviz.lib"
  )

list(APPEND _cmake_import_check_targets GLviz::glviz )
list(APPEND _cmake_import_check_files_for_GLviz::glviz "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/glviz.lib" )

# Import target "GLviz::shader" for configuration "Release"
set_property(TARGET GLviz::shader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GLviz::shader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/shader.lib"
  )

list(APPEND _cmake_import_check_targets GLviz::shader )
list(APPEND _cmake_import_check_files_for_GLviz::shader "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/shader.lib" )

# Import target "GLviz::embed_resource" for configuration "Release"
set_property(TARGET GLviz::embed_resource APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GLviz::embed_resource PROPERTIES
  IMPORTED_LOCATION_RELEASE "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/bin/embed_resource.exe"
  )

list(APPEND _cmake_import_check_targets GLviz::embed_resource )
list(APPEND _cmake_import_check_files_for_GLviz::embed_resource "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/bin/embed_resource.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
