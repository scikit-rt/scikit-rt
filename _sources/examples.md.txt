# Scikit-rt by examples

[Jupyter](https://jupyter.org/) notebooks are available that demonstrate
scikit-rt functionality.  These may be obtained through either of the
following:

1. Click on any of the notebook links given below.  On the linked page,
   which shows the rendered notebook, click on the button marked `Raw` (top
   right, just above the notebook).  Right click on the resulting code
   document, select `Save As...`, and save with extension `.ipynb`.  (Some
   browsers may try to add the extension `.txt`.)

2. Clone the scikit-rt repository:

   ```
   git clone https://github.com/scikit-rt/scikit-rt
   ```

    Navigate to the example notebooks, in the directory
    `scikit-rt/examples/notebooks`.

The [scikit-rt installation](installation.md) includes both
[Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/) and
[JupyterLab](https://github.com/jupyterlab/jupyterlab).  With the
environment for using scikit-rt activated, a downloaded notebook can
be loaded with, for example:

```
jupyter lab path/to/notebook.ipynb
```

## Notebooks using public datasets, or not requiring data

The following notebooks either use public datasets or don't require
input data.  They can be run as they are, possibly after modifying paths
to directories for copying data and saving outputs.

- [plotting_demo.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/plotting_demo.ipynb) :
  plotting capabilities.
- [application_demo.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/application_demo.ipynb) :
  definition and running of a scikit-rt application, for (non-interactive)
  processing of datasets for multiple patients.
- [roi_intensities.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/roi_intensities.ipynb) :
  hisogramming voxel intensities inside an ROI.
- [dose_volume_rois.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/dose_volume_rois.ipynb) :
  creation of ROI objects corresponding to volumes receiving
  a specified radiation dose.
- [eqd.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/eqd.ipynb) :
  calculation of equivalent dose, for specified dose per fraction,
  and of biologically effective dose.
- [grid_creation.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/grid_creation.ipynb) :
  creation of a grid image, which can be useful, for example,
  for understanding the effect of applying a registration transform.
- [image_registration_checks.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/image_registration_checks.ipynb):
  registration of images featuring geometrical shapes, and qualitative
  checks of registration performance.

## Notebooks using non-public datasets

The following were run using data from the [VoxTox](https://www.cancerresearchuk.org/about-cancer/find-a-clinical-trial/a-study-to-collect-detailed-information-about-side-effects-of-radiotherapy-for-cancers-of-the-prostate-head-and-neck-or-central-nervous-system-voxtox) study.  They can't be rerun without equivalent
data, but may be useful for code examples.

- [workshop_26_09_22.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/workshop_26_09_22.ipynb) :
  overview of interactive functionality;
  presentation by K. Harrison, 26th September 2022.
- [kvct_to_mvct.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/kvct_to_mvct.ipynb):
  generation of a megavoltage (MV) computed-tomography (CT) guidance scan,
  starting from a (downsampled) kilovoltage (kV) CT scan, used in
  radiotherapy planning.
- [workshop_19_01_22.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/workshop_19_01_22.ipynb) :
  overview of interactive functionality;
  presentation by H. Pullen, 19th January 2022.
