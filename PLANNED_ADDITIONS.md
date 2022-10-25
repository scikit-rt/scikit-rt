# Scikit-rt: planned additions

## 01 July 2022 - 31 December 2022

Planned additions to Scikit-rt for the period 1 July 2022 to 31 December 2022
are as outlined below.

### Reading and writing of DICOM datasets

In the original implementation of the `skrt.patient.Patient` class, DICOM data
are read and linked under the assumption that files have been pre-sorted
to follow the organisation used in the VoxTox study
 ([http://dx.doi.org/10.1088/1742-6596/898/7/072048](http://dx.doi.org/10.1088/1742-6596/898/7/072048)).  Additions to the `skrt.patient.Patient` class
are to allow:
- reading and linking of DICOM files grouped by patient but otherwise
unsorted;
- writing out all or part of a DICOM dataset accoding to the VoxTox model.

The `skrt.patient.Patient` class provides generic data-handling
capabilities, but individual studies may benefit from subclasses providing
more-specialised functionality.  An example is to be provided:
- example `skrt.patient.Patient` subclass for a study of local
recurrence following radiotherapy for breast canser.

## Data preparation for auto-segmentation

Tools are to be added to enable, or simplify, data preprocessing for
auto-segmentation:
- standardisation/redefinition of voxel sizes;
- standardisation of image sizes;
- masking/cropping to within a specified margin of image foregrounds;
- writing data organised as required for training/inference using
the [Inner-Eye Deep-Learning framework](https://innereye-deeplearning.readthedocs.io/).

## Registration-based auto-segmentation

Tools and example workflows are to be added to help with registration-based
auto-segmentation:
- possibilities for initial alignment of fixed and moving image:
  - alignment at image top, centre, bottom, along each axis;
  - alignment to a specified slice of a previously defined structure;
  - user-specified translation;
- variation of registration parameters, for performance optimisation;
- registration of multiple moving images to a single fixed image;
- registration of a single moving image to multiple fixed images;
- simplified workflow for pushing ROI contour points from coordinate system
  of fixed image to coordinate system of moving image;
- simplified workflow for pulling ROI masks from coordinate system of
  moving image to coordinate system of fixed image.

## Evaluation of auto-segmentation performance

The possibilities available for evaluating auto-segmentation performance
are to be extended:
- calculation of conformity indicies for comparison of multiple ROIs ([https://doi.org/10.1088/0031-9155/54/9/018](https://doi.org/10.1088/0031-9155/54/9/018));
- calculation of distances to conformity ([https://doi.org/10.1259/bjr/27674581](https://doi.org/10.1259/bjr/27674581));
- calculation of elements of confusion matrix, and metrics based on these;
- slice-by-slice comparisons for all image slices through compared ROIs;
- visualisation of posterior probabilities and uncertianties (Shannon entropy)
  from auto-segmentation with [Inner-Eye Deep-Learning framework](https://innereye-deeplearning.readthedocs.io/).

## Analysis of radiation doses

Possibilities are to be added to enable, or simplify, analysis of radiation
doses, and in particular doses to ROIs:
- summation of dose fields;
- matching of dose fields to image volumes;
- creation of population dose-volume histograms.
