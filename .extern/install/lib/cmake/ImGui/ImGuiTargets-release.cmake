#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ImGui::imgui" for configuration "Release"
set_property(TARGET ImGui::imgui APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ImGui::imgui PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui.lib"
  )

list(APPEND _cmake_import_check_targets ImGui::imgui )
list(APPEND _cmake_import_check_files_for_ImGui::imgui "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui.lib" )

# Import target "ImGui::imgui_sdl" for configuration "Release"
set_property(TARGET ImGui::imgui_sdl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ImGui::imgui_sdl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui_sdl.lib"
  )

list(APPEND _cmake_import_check_targets ImGui::imgui_sdl )
list(APPEND _cmake_import_check_files_for_ImGui::imgui_sdl "C:/Users/CanonUser/Documents/GitHub/surface_splatting/.extern/install/lib/imgui_sdl.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
