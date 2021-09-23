cmake_minimum_required(VERSION 3.13.4)

project(DensityLungSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/pzaffino/SlicerDensityLungSegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy), Maria Francesca Spadea (Magna Graecia University of Catanzaro, Italy)")
set(EXTENSION_DESCRIPTION "This extension labels lung tissues on basis of density (air, healthy, ground-glass opacities, consolidation, and denser tissues). Voxel-per-voxel label is provided alongside with an averaged volume.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/pzaffino/SlicerDensityLungSegmentation/main/DensityLungSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/pzaffino/SlicerDensityLungSegmentation/main/DensityLungSegmentation_screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(LungCTGMMSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
