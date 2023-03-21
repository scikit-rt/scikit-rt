# Scikit-rt demos

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

    Navigate to the demo notebooks, in the directory
    `scikit-rt/examples/notebooks`.

The [scikit-rt installation](docs/installation.md) includes both
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

- [plotting_demo.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/plotting_demo.ipynb)
 demonstrates plotting capabilities;
- [application_demo.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/application_demo.ipynb)
demonstrates definition and running of a scikit-rt application,
for (non-interactive) processing of datasets for multiple patients;
- [dose_volume_rois.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/dose_volume_rois.ipynb)
demonstrates how to create ROI objects corresponding to volumes receiving
a specified radiation dose.
- [grid_creation.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/grid_creation.ipynb)
demonstrates creation of a grid image, which can be useful, for example,
for understanding the effect of applying a registration transform.

## Notebooks using non-public datasets

The following were run using data from the [VoxTox](https://www.cancerresearchuk.org/about-cancer/find-a-clinical-trial/a-study-to-collect-detailed-information-about-side-effects-of-radiotherapy-for-cancers-of-the-prostate-head-and-neck-or-central-nervous-system-voxtox) study.  They can't be rerun without equivalent
data, but may be useful as code examples.

- [workshop_26_09_22.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/workshop_26_09_22.ipynb)
demonstrates general interactive functionality;
- [kvct_to_mvct.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/kvct_to_mvct.ipynb) demonstrates generation of a megavoltage (MV)
computed-tomography (CT) guidance scan, starting from a (downsampled)
kilovoltage (kV) CT scan, used in radiotherapy planning;
- [rtplan.ipynb](https://github.com/scikit-rt/scikit-rt/blob/master/examples/notebooks/rtplan.ipynb) demonstrates how to access some of the information in a radiotherapy plan.
