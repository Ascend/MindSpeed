# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project configuration (explicitly declare C++ project)
project(QOS_TOOL CXX)

# ==============================================================================
# 1. Compiler Configuration
# ==============================================================================
# Specify C++ compiler path
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# Enforce C++11 standard (no extensions, mandatory compliance)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 2. Python 3 Binding Configuration
# ==============================================================================
# Find Python 3 (require interpreter and development components)
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "✅ Found Python 3: ${Python_EXECUTABLE} (version: ${Python_VERSION})")

# ==============================================================================
# 3. pybind11 Configuration
# ==============================================================================
# Set pybind11 CMake config directory (update this path to your actual install dir)
set(pybind11_DIR "/pybind11_install_dir/share/cmake/pybind11")
# Find pybind11 (use explicit path to avoid system-wide lookup)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
message(STATUS "✅ Found pybind11: ${pybind11_DIR}")

# ==============================================================================
# 4. DCMI Library Configuration
# ==============================================================================
# Include header directories for DCMI and Ascend driver
include_directories(
    /usr/local/Ascend/driver/include    # Ascend driver headers
    /usr/local/dcmi/include             # DCMI SDK headers
    ${Python_INCLUDE_DIRS}              # Python 3 development headers
)

# Link directories for DCMI and Ascend driver libraries
link_directories(
    /usr/local/Ascend/driver/lib64/driver  # Ascend driver libraries
    /usr/local/dcmi                        # DCMI SDK libraries
)

# ==============================================================================
# 5. Source File Configuration
# ==============================================================================
# Define source file path
set(SRC_FILE "aiQos.c")

# Validate source file existence (fail fast if missing)
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "❌ Source file not found: ${CMAKE_SOURCE_DIR}/${SRC_FILE}")
endif()

# Critical fix: Force compile .c file as C++ (ignore file extension)
set_source_files_properties(${SRC_FILE} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-x c++"
)

# ==============================================================================
# 6. Build Python Extension Module
# ==============================================================================
# Create pybind11-based Python extension module
pybind11_add_module(aiQos ${CMAKE_SOURCE_DIR}/${SRC_FILE})

# ==============================================================================
# 7. Link Libraries
# ==============================================================================
# Link required libraries to the extension module
target_link_libraries(aiQos PRIVATE
    /usr/local/dcmi/libdcmi.so  # DCMI core library
    m                           # Math library
    pthread                     # POSIX thread library
    dl                          # Dynamic linking library
    ${Python_LIBRARIES}         # Python 3 runtime libraries
)

# ==============================================================================
# 8. Runtime Configuration (Critical Fixes)
# ==============================================================================
set_target_properties(aiQos PROPERTIES
    LINKER_LANGUAGE CXX                          # Explicitly set linker language to C++
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output  # Output module to ./output
    INSTALL_RPATH "/usr/local/dcmi:/usr/local/Ascend/driver/lib64/driver"  # Runtime library path
    BUILD_WITH_INSTALL_RPATH ON                  # Use INSTALL_RPATH during build
    C_VISIBILITY_PRESET default                  # Disable symbol hiding (C)
    CXX_VISIBILITY_PRESET default                # Disable symbol hiding (C++) - ensure PyInit_aiQos export
)

# ==============================================================================
# 9. Compilation Options
# ==============================================================================
# Add global compile options (enable warnings + force symbol export)
add_compile_options(
    -Wall           # Enable all warning messages
    -O2             # Optimization level 2 (balance speed/size)
    -fvisibility=default  # Fallback: force export all symbols (ensure Python module entry point is visible)
)