# Usage and data model

## Usage
Scikit-rt can be used in scripts, to be run from the command line
or to be executed on a batch system; and can be used in
[Jupyter](https://jupyter.org/) notebooks, where interactive features
are enabled for image viewing.  The Scikit-rt installation includes both
[Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/)
 and [JupyterLab](https://github.com/jupyterlab/jupyterlab).

Scikit-rt is able to load [DICOM](https://www.dicomstandard.org/current)
 data (images, RTDOSE, RTSTRUCT, RTPLAN) and
[NIfTI](https://nifti.nimh.nih.gov/) data (images, dose images,
regions of interest (ROIs) represented as masks), and to convert between them.

## Data model

The different types of radiotherapy data are represented in scikit-rt by
different classes:

- [Image](https://scikit-rt.github.io/scikit-rt/skrt.image.html#skrt.image.Image);
- [Dose](https://scikit-rt.github.io/scikit-rt/skrt.dose.html#skrt.dose.Dose);  
- [StructureSet](https://scikit-rt.github.io/scikit-rt/skrt.structures.html#skrt.structures.StructureSet);
- [ROI](https://scikit-rt.github.io/scikit-rt/skrt.structures.html#skrt.structures.ROI) (region of interest);
- [Plan](https://scikit-rt.github.io/scikit-rt/skrt.dose.html#skrt.dose.Plan).

There are associations between class instances:

- a StructureSet is composed of ROIs, and relates to a specific Image;
- a Plan has constraints relating to a StructureSet;
- a Dose is calculated on the basis of a Plan and the material densities
  represented by an Image.

Data relating to an indidual patient are typically grouped into one or
more studies.  This is represented by the classes:
- [Patient](https://scikit-rt.github.io/scikit-rt/skrt.patient.html#skrt.patient.Patient);
- [Study](https://scikit-rt.github.io/scikit-rt/skrt.patient.html#skrt.patient.Study).

A class diagram for the data model is shown below.

<img src="../images/scikit-rt_data_model.png" alt="Scikit-rt data model" style="width:80%">