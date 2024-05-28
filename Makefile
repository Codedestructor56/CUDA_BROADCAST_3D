# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/richardM/pytorch-dir/cust_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/richardM/pytorch-dir/cust_cuda

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/richardM/pytorch-dir/cust_cuda/CMakeFiles /home/richardM/pytorch-dir/cust_cuda//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/richardM/pytorch-dir/cust_cuda/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -P /home/richardM/pytorch-dir/cust_cuda/CMakeFiles/VerifyGlobs.cmake
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named cuda-app

# Build rule for target.
cuda-app: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cuda-app
.PHONY : cuda-app

# fast build rule for target.
cuda-app/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/build
.PHONY : cuda-app/fast

src/cuda-app.o: src/cuda-app.cpp.o
.PHONY : src/cuda-app.o

# target to build an object file
src/cuda-app.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-app.cpp.o
.PHONY : src/cuda-app.cpp.o

src/cuda-app.i: src/cuda-app.cpp.i
.PHONY : src/cuda-app.i

# target to preprocess a source file
src/cuda-app.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-app.cpp.i
.PHONY : src/cuda-app.cpp.i

src/cuda-app.s: src/cuda-app.cpp.s
.PHONY : src/cuda-app.s

# target to generate assembly for a file
src/cuda-app.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-app.cpp.s
.PHONY : src/cuda-app.cpp.s

src/cuda-kernel.o: src/cuda-kernel.cu.o
.PHONY : src/cuda-kernel.o

# target to build an object file
src/cuda-kernel.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-kernel.cu.o
.PHONY : src/cuda-kernel.cu.o

src/cuda-kernel.i: src/cuda-kernel.cu.i
.PHONY : src/cuda-kernel.i

# target to preprocess a source file
src/cuda-kernel.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-kernel.cu.i
.PHONY : src/cuda-kernel.cu.i

src/cuda-kernel.s: src/cuda-kernel.cu.s
.PHONY : src/cuda-kernel.s

# target to generate assembly for a file
src/cuda-kernel.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda-app.dir/build.make CMakeFiles/cuda-app.dir/src/cuda-kernel.cu.s
.PHONY : src/cuda-kernel.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... cuda-app"
	@echo "... src/cuda-app.o"
	@echo "... src/cuda-app.i"
	@echo "... src/cuda-app.s"
	@echo "... src/cuda-kernel.o"
	@echo "... src/cuda-kernel.i"
	@echo "... src/cuda-kernel.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -P /home/richardM/pytorch-dir/cust_cuda/CMakeFiles/VerifyGlobs.cmake
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

