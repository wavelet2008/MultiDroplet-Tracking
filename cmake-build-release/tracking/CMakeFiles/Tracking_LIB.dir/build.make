# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\Program Files (x86)\CLion 2019.3.4\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\Program Files (x86)\CLion 2019.3.4\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release"

# Include any dependencies generated for this target.
include tracking/CMakeFiles/Tracking_LIB.dir/depend.make

# Include the progress variables for this target.
include tracking/CMakeFiles/Tracking_LIB.dir/progress.make

# Include the compile flags for this target's objects.
include tracking/CMakeFiles/Tracking_LIB.dir/flags.make

tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj: tracking/CMakeFiles/Tracking_LIB.dir/flags.make
tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj: tracking/CMakeFiles/Tracking_LIB.dir/includes_CXX.rsp
tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj: ../tracking/tracking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj"
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && D:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Tracking_LIB.dir\tracking.cpp.obj -c "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\tracking\tracking.cpp"

tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tracking_LIB.dir/tracking.cpp.i"
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && D:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\tracking\tracking.cpp" > CMakeFiles\Tracking_LIB.dir\tracking.cpp.i

tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tracking_LIB.dir/tracking.cpp.s"
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && D:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\tracking\tracking.cpp" -o CMakeFiles\Tracking_LIB.dir\tracking.cpp.s

# Object files for target Tracking_LIB
Tracking_LIB_OBJECTS = \
"CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj"

# External object files for target Tracking_LIB
Tracking_LIB_EXTERNAL_OBJECTS =

tracking/libTracking_LIB.a: tracking/CMakeFiles/Tracking_LIB.dir/tracking.cpp.obj
tracking/libTracking_LIB.a: tracking/CMakeFiles/Tracking_LIB.dir/build.make
tracking/libTracking_LIB.a: tracking/CMakeFiles/Tracking_LIB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libTracking_LIB.a"
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && $(CMAKE_COMMAND) -P CMakeFiles\Tracking_LIB.dir\cmake_clean_target.cmake
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Tracking_LIB.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tracking/CMakeFiles/Tracking_LIB.dir/build: tracking/libTracking_LIB.a

.PHONY : tracking/CMakeFiles/Tracking_LIB.dir/build

tracking/CMakeFiles/Tracking_LIB.dir/clean:
	cd /d "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" && $(CMAKE_COMMAND) -P CMakeFiles\Tracking_LIB.dir\cmake_clean.cmake
.PHONY : tracking/CMakeFiles/Tracking_LIB.dir/clean

tracking/CMakeFiles/Tracking_LIB.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\tracking" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-release\tracking\CMakeFiles\Tracking_LIB.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : tracking/CMakeFiles/Tracking_LIB.dir/depend

