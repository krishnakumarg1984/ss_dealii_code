##
#  CMake script for the solid_diffusion program:
##

# Set the name of the project and target:
SET(PROJECT_NAME "solid_diffusion")
# SET(TARGET "solid_diffusion")
SET(TARGET ${PROJECT_NAME}.run)

SET(TARGET_SRC
    ${PROJECT_NAME}.cc
  )

# Define the output that should be cleaned:
SET(CLEAN_UP_FILES *.vtu *.pvd)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.2.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
