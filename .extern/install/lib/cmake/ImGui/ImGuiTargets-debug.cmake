#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ImGui::imgui" for configuration "Debug"
set_property(TARGET ImGui::imgui APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ImGui::imgui PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imguid.lib"
  )

list(APPEND _cmake_import_check_targets ImGui::imgui )
list(APPEND _cmake_import_check_files_for_ImGui::imgui "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imguid.lib" )

# Import target "ImGui::imgui_sdl" for configuration "Debug"
set_property(TARGET ImGui::imgui_sdl APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ImGui::imgui_sdl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui_sdld.lib"
  )

list(APPEND _cmake_import_check_targets ImGui::imgui_sdl )
list(APPEND _cmake_import_check_files_for_ImGui::imgui_sdl "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui_sdld.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
