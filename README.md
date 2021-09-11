# SlicerLungDensitySegmentation
This extension labels lung tissues on basis of density.
Voxel-per-voxel label is provided alongside with an averaged volume.

The implemented workflow is described in:
Zaffino, Paolo, et al. "An Open-Source COVID-19 CT Dataset with Automatic Lung Tissue Classification for Radiomics." Bioengineering 8.2 (2021): 26.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7919807/

The license used is APACHE 2.0.

![screenshot](https://raw.githubusercontent.com/pzaffino/SlicerDensityLungSegmentation/main/LungDensitySegmentation_screenshot.png)

## Module description

### Lung CT GMM Segmentation

This module, given a chest CT, labels lung tissue by fitting the intensities with a Gaussian Mixture Model already available. It can be used for pneumonia (COVID-19 too).

Tutorial

1. Load chest CT (COVID-19 CTs can be download from https://www.imagenglab.com/newsite/covid-19/ )
2. Select Chest CT GMM Segmentation module (in the segmentation menu)
3. Select/create a labelmap for the result
4. Select/create a labelmap for the averaged result
5. Click Apply button
