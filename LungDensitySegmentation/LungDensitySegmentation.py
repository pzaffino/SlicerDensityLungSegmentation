import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

from slicer.util import setSliceViewerLayers
import numpy as np
import SimpleITK as sitk
import sitkUtils
import scipy.ndimage

#
# LungDensitySegmentation
#

class LungDensitySegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Density Lung Segmentation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (Magna Graecia University of Catanzaro, Italy)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = '''
This module labels lung tissues on basis of intensities.
The full validation workflow is described in Zaffino, Paolo, et al. "An Open-Source COVID-19 CT Dataset with Automatic Lung Tissue Classification for Radiomics." Bioengineering 8.2 (2021): 26.
'''
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """ """ # replace with organization, grant and thanks.

#
# LungDensitySegmentationWidget
#

class LungDensitySegmentationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """


  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # GT CT volume selector
    #
    self.CTSelector = slicer.qMRMLNodeComboBox()
    self.CTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.CTSelector.selectNodeUponCreation = True
    self.CTSelector.addEnabled = False
    self.CTSelector.removeEnabled = False
    self.CTSelector.noneEnabled = False
    self.CTSelector.showHidden = False
    self.CTSelector.showChildNodeTypes = False
    self.CTSelector.setMRMLScene(slicer.mrmlScene)
    self.CTSelector.setToolTip( "Select the CT" )
    parametersFormLayout.addRow("CT volume: ", self.CTSelector)

    #
    # output volume selector
    #

    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene(slicer.mrmlScene)
    self.outputSelector.setToolTip("Select or create a labelmap for lung tissue classification")
    parametersFormLayout.addRow("Output labelmap: ", self.outputSelector)

    #
    # Averaged output volume selector
    #

    self.averagedOutputSelector = slicer.qMRMLNodeComboBox()
    self.averagedOutputSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.averagedOutputSelector.selectNodeUponCreation = True
    self.averagedOutputSelector.addEnabled = True
    self.averagedOutputSelector.removeEnabled = True
    self.averagedOutputSelector.noneEnabled = True
    self.averagedOutputSelector.showHidden = False
    self.averagedOutputSelector.showChildNodeTypes = False
    self.averagedOutputSelector.setMRMLScene(slicer.mrmlScene)
    self.averagedOutputSelector.setToolTip("Select or create a labelmap for averaged lung tissue classification")
    parametersFormLayout.addRow("Averaged output labelmap: ", self.averagedOutputSelector)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply (it can takes a few minutes)")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.CTSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.averagedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

    # Create logic object
    self.logic = LungDensitySegmentationLogic()

  def onSelect(self):
    self.applyButton.enabled = self.CTSelector.currentNode() and self.outputSelector.currentNode() and self.averagedOutputSelector.currentNode()


  def onApplyButton(self):
    self.logic.run(self.CTSelector.currentNode().GetName(), self.outputSelector.currentNode(), self.averagedOutputSelector.currentNode())
#
# LungDensitySegmentationLogic
#

class LungDensitySegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def extract_only_lungs_islands(self, thr_img):
    """
    Extract only lung islands from patient's binary image
    """

    # Create final mask
    final_mask = np.zeros_like(thr_img, dtype=np.uint8)

    # Compute islands
    label_im, nb_labels = scipy.ndimage.label(thr_img)
    sizes = scipy.ndimage.sum(thr_img, label_im, range(nb_labels + 1))

    # investigate each island
    for i in range(nb_labels):

      # discard small islands
      if sizes[i] < 5.0e5:
        continue

      # Check if island is background (bbox overlapping with image corner)
      img_coords = np.zeros_like(thr_img, dtype=np.uint8)
      img_coords[label_im==i]=1
      coords = self.bbox(img_coords, margin=0)

      if (coords[2] != 0 and coords[4]!=0 and
         coords[3] != thr_img.shape[1]-1 and coords[5] != thr_img.shape[2]-1): # non background, set as lung

        final_mask[img_coords==1]=1

    return final_mask

  def bbox(self, img, margin=20):
    """
    Compute bounding box of a binary mask and add a maring (only in axial plane).
    """

    coords=[0,img.shape[0],0,img.shape[1],0,img.shape[2]]

    # i
    for i in range(img.shape[0]):
      if 1 in img[i,:,:]:
        coords[0]=i
        break
    for i in range(img.shape[0]-1,-1,-1):
      if 1 in img[i,:,:]:
        coords[1]=i
        break
    # j
    for j in range(img.shape[1]):
      if 1 in img[:,j,:]:
        coords[2]=j - margin
        break
    for j in range(img.shape[1]-1,-1,-1):
      if 1 in img[:,j,:]:
        coords[3]=j + margin
        break
    # k
    for k in range(img.shape[2]):
      if 1 in img[:,:,k]:
        coords[4]=k - margin
        break
    for k in range(img.shape[2]-1,-1,-1):
      if 1 in img[:,:,k]:
        coords[5]=k + margin
        break

    assert coords[0] >= 0 and coords[2] >= 0 and coords[4] >= 0
    assert coords[1] <= img.shape[0]-1 and coords[3] <= img.shape[1]-1 and coords[5] <= img.shape[2]-1

    return coords


  def binary_closing_sitk(self, img_np, radius_list):
    """
    SimpleITK much faster and less compute-intesive than skimage
    """

    img_sitk = sitk.GetImageFromArray(img_np)

    for radius in radius_list:
      img_sitk = sitk.BinaryMorphologicalClosing(img_sitk, [radius, radius, radius])

    return sitk.GetArrayFromImage(img_sitk).astype(np.uint8)


  def threshold_image(self, ct, intensity_thr=-155):
    """
    Execute a threshold based segmentation and fill holes
    """

    thr_img = np.zeros_like(ct, dtype=np.uint8)
    thr_img[ct>=intensity_thr]=1
    thr_img = 1 - thr_img
    thr_img = scipy.ndimage.binary_opening(thr_img, iterations=3)

    return thr_img

  def close_lungs_mask(self, lungs_mask):
    """
    Close lungs binary mask.
    """

    # Do bounding box (to sepped up morph filters)
    coords = self.bbox(lungs_mask)
    bb_lungs_mask = lungs_mask[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]

    # Binary closing
    closed_bb_lung_mask = self.binary_closing_sitk(bb_lungs_mask, [30, 20])

    assert closed_bb_lung_mask.sum() > 1000

    # Undo bounding box
    closed_lung_mask = np.zeros_like(lungs_mask, dtype=np.uint8)
    closed_lung_mask[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]] = closed_bb_lung_mask

    return closed_lung_mask


  def run(self, CTVolumeName, outputVolume, averagedOutputVolume):
    """
    Run intensity labeling
    """

    # Import the required libraries
    try:
      import joblib
    except ModuleNotFoundError:
      slicer.util.pip_install("joblib")
      import joblib

    try:
      import sklearn
    except ModuleNotFoundError:
      slicer.util.pip_install("scikit-learn")
      import sklearn

    # Get sitk/numpy images from Slicer
    CT_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(CTVolumeName), sitk.sitkFloat32)
    CT_np = sitk.GetArrayFromImage(CT_sitk)
    CT_np[CT_np<-1000]=-1000

    # Compute lung mask
    thr_CT = self.threshold_image(CT_np, -155)
    lungs_mask = self.extract_only_lungs_islands(thr_CT)
    closed_lungs_mask = self.close_lungs_mask(lungs_mask)

    CT_np[closed_lungs_mask==0]=-1000
    CT_flatten = CT_np.flatten()

    # Remove background
    indexes_to_remove = np.argwhere(closed_lungs_mask.flatten()==0)
    lungs = np.delete(CT_flatten, indexes_to_remove)

    # Run GMM
    gmm_model_fn = __file__.replace("LungDensitySegmentation.py", "Resources%sGMM_parameters_COVID-19.joblib" % (os.sep))
    gmm = joblib.load(gmm_model_fn)
    gmm_labels = gmm.predict(lungs.reshape(-1,1)).reshape(lungs.shape)

    # Make label values fixed
    sorted_label = np.zeros_like(lungs, dtype=np.uint8)
    sorted_gmm_means = np.argsort([i[0] for i in gmm.means_])

    sorted_label[gmm_labels==[sorted_gmm_means[0]]]=1
    sorted_label[gmm_labels==[sorted_gmm_means[1]]]=2
    sorted_label[gmm_labels==[sorted_gmm_means[2]]]=3
    sorted_label[gmm_labels==[sorted_gmm_means[3]]]=4
    sorted_label[gmm_labels==[sorted_gmm_means[4]]]=5

    # Restore background voxels
    indexes_to_leave = np.argwhere(closed_lungs_mask.flatten()==1)
    indexes_to_leave_list = [i[0] for i in indexes_to_leave]

    final_label = np.zeros_like(CT_flatten, dtype=np.uint8)

    counter = 0
    for i in indexes_to_leave_list:
        final_label[i] = sorted_label[counter]
        counter += 1

    # Reshape array labels. From 1D to 3D
    final_label = final_label.reshape(CT_np.shape)
    final_label_sitk = sitk.GetImageFromArray(final_label)
    final_label_sitk.CopyInformation(CT_sitk)

    # Average label
    filtered_label = np.rint(scipy.ndimage.median_filter(final_label, 4)).astype(np.uint8)
    filtered_label_sitk = sitk.GetImageFromArray(filtered_label)
    filtered_label_sitk.CopyInformation(CT_sitk)

    # Show labelmap
    outputVolume = sitkUtils.PushVolumeToSlicer(final_label_sitk, outputVolume)
    averagedOutputVolume = sitkUtils.PushVolumeToSlicer(filtered_label_sitk, averagedOutputVolume)
    setSliceViewerLayers(background=outputVolume)

