cmake_minimum_required(VERSION 3.10)
project(PatternRecognitionProject)

# Set the C standard
set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(src/model src/file src/sequential)

# Add source files
set(SOURCES
    src/model/dataset.c
    src/file/loader.c
    src/file/dumper.c
    src/sequential/sequential.c
    src/sequential/setup.c
    src/sequential/sad.c
)

# Add executable
add_executable(sequential ${SOURCES})