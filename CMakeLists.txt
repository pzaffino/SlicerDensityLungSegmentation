cmake_minimum_required(VERSION 3.13.4)

project(LungDensitySegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/LungDensitySegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy), Maria Francesca Spadea (Magna Graecia University of Catanzaro, Italy)")
set(EXTENSION_DESCRIPTION "This extenson labels lung tissues on basis of density")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/pzaffino/SlicerLungDensitySegmentation/main/LungDensitySegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/pzaffino/SlicerLungDensitySegmentation/main/LungDensitySegmentation_screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(LungDensitySegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
