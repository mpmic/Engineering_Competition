cmake_minimum_required(VERSION 3.15)

# Set the output directories for built binaries, libraries, and runtime files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")       # Output directory for executable files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")      # Output directory for library files

# On Windows, set the library output directory to the bin directory
if(WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")  # Output directory for library files (Windows)
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")  # Output directory for library files (non-Windows)
endif()

# Set configuration-specific output directories
foreach(CFG IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${CFG}" CFG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG} "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endforeach()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the project with name "EC2023" and language C++
project(EC2023 LANGUAGES CXX)

# Add the main executable target with source files main.cpp and solution.h
add_executable(${PROJECT_NAME}
        ${CMAKE_CURRENT_LIST_DIR}/main.cpp
        ${CMAKE_CURRENT_LIST_DIR}/solution.h)

# Add additional source files to the target
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec2023.h                    # Additional source files in ec2023 directory
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_measurement.h
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_float.h
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_vec_hw.h
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_stream_hw.h
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_signal_reader.h
        ${CMAKE_CURRENT_LIST_DIR}/ec2023/ec_math.h
        ${CMAKE_CURRENT_LIST_DIR}/solution.h)

# Add the current directory as a private include directory for the target
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR})

# Add a custom command to copy files after the build
# Copies input_signal.txt and reference_spectrum.txt from the current directory to the bin directory
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/input_signal.txt"                 # Source file location
        "${CMAKE_BINARY_DIR}/bin/"                                   # Destination directory
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/reference_spectrum.txt"           # Source file location
        "${CMAKE_BINARY_DIR}/bin/")                                  # Destination directory
