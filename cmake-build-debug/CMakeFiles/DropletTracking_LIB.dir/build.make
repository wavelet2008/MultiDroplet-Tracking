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
CMAKE_BINARY_DIR = "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/DropletTracking_LIB.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DropletTracking_LIB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DropletTracking_LIB.dir/flags.make

CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj: CMakeFiles/DropletTracking_LIB.dir/flags.make
CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj: CMakeFiles/DropletTracking_LIB.dir/includes_CXX.rsp
CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj: ../multidroplet_tracking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj"
	D:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\DropletTracking_LIB.dir\multidroplet_tracking.cpp.obj -c "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\multidroplet_tracking.cpp"

CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.i"
	D:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\multidroplet_tracking.cpp" > CMakeFiles\DropletTracking_LIB.dir\multidroplet_tracking.cpp.i

CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.s"
	D:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\multidroplet_tracking.cpp" -o CMakeFiles\DropletTracking_LIB.dir\multidroplet_tracking.cpp.s

# Object files for target DropletTracking_LIB
DropletTracking_LIB_OBJECTS = \
"CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj"

# External object files for target DropletTracking_LIB
DropletTracking_LIB_EXTERNAL_OBJECTS =

libDropletTracking_LIB.a: CMakeFiles/DropletTracking_LIB.dir/multidroplet_tracking.cpp.obj
libDropletTracking_LIB.a: CMakeFiles/DropletTracking_LIB.dir/build.make
libDropletTracking_LIB.a: CMakeFiles/DropletTracking_LIB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libDropletTracking_LIB.a"
	$(CMAKE_COMMAND) -P CMakeFiles\DropletTracking_LIB.dir\cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\DropletTracking_LIB.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DropletTracking_LIB.dir/build: libDropletTracking_LIB.a

.PHONY : CMakeFiles/DropletTracking_LIB.dir/build

CMakeFiles/DropletTracking_LIB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\DropletTracking_LIB.dir\cmake_clean.cmake
.PHONY : CMakeFiles/DropletTracking_LIB.dir/clean

CMakeFiles/DropletTracking_LIB.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug" "E:\Digital Microfluidics\MV4DMF\MultiDroplet Tracking\cmake-build-debug\CMakeFiles\DropletTracking_LIB.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/DropletTracking_LIB.dir/depend

