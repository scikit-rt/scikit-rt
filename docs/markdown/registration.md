# Image registration

## Installation and setup

One or more image-registration packages should be installed,
as outlined in the instructions:
[installation and setup for image registration](installation.md#installation-and-setup-for-image-registration)

The currently supported image-registration packages are:

- [elastix](https://elastix.lumc.nl/)
- [matlab-skrt](https://github.com/kh296/matlab-skrt/)
- [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)

## Example registration

The example code below is intended for use inside a Jupyter notebook,
in particular for the interactive image viewing.  It covers the following:

- perform a registration;
- view registration results for qualitative evaluation;
- transform to the space of the fixed image a structure set defined 
  for the moving image;
- quantitatively compare transformed structure set and structure set
  defined for the fixed image.

```
from skrt.registration import Registration, get_default_pfiles

# Specify the registration engine.
engine = "elastix"

# Create registration object.
reg = Registration(
  engine=engine,
  engine_dir="some/location/elastix-5.0.1-mac,
  fixed="some/location/fixed_image_dicoms",
  moving="some/location/moving_image_dicoms",
  pfiles={
    "rigid": get_default_pfiles("*Rigid*", engine=engine),
    "bspline": get_default_pfiles("*BSpline15*", engine=engine), 
  },
  outdir="my_results"
)

# Perform registration.
reg.register()

# View initial and final images.
reg.view_init()
reg.view_result()

# Transform moving-image structures.
from skrt import StructureSet
structs_moving = StructureSet("some/location/moving_structs.dcm")
structs_fixed = StructureSet("some/location/fixed_structs.dcm")
structs_transformed = reg.transform(structs_moving)

# Obtain comparison metrics for transformed vs. fixed-image structures.
df = structs_fixed.get_comparison(structs_transformed)
df.to_csv("metrics.csv")
```

Explanation of each part of the code:

1. Import the image-registration class, and the function
for locating example parameter files included in the scikit-rt package.
```
from skrt.registration import Registration, get_default_pfiles
```

3. Create a Registration object.
```
reg = Registration(
  engine="elastix",
  engine_dir="some/location/elastix-5.0.1-mac,
  fixed="some/location/fixed_image_dicoms",
  moving="some/location/moving_image_dicoms",
  pfiles={
    "rigid": get_default_pfiles("*Rigid*", engine=engine),
    "bspline": get_default_pfiles("*BSpline15*", engine=engine), 
  },
  outdir="my_results"
)
```
The arguments to the `Registration` class initialisation are:
- `engine`: Name of registration engine ("elastix", "matlab", or "niftyreg").
- `engine_dir` Path to location of registration software.  Parameter
  not needed if environment setup for registration software performed
  prior to using scikit-rt.
- `fixed`: Path to a directory containing the dicom files for the fixed image. Can also be a numpy array or the path to a single nifti file.
- `moving`: Path to a directory containing the dicom files for the moving image. Can also be a numpy array or the path to a single nifti file.
- `pfiles`: Dictionary where keys are names to be associated with registration
  steps, and values are paths to parameter file(s).  The example uses
  parameter files included in the scikit-rt package.
- `outdir`: Path to directory in which to save the results of the registration. The registration object can later be reloaded from this directory.

For each parameter file, a subdirectory will be created in `"my_results"` with the step name as specified by the associated key in the `pfiles` dictionary. If a registration-transform file is already found in that directory, the registration will not be rerun for that parameter file, and instead the existing transform will be used.  To completely overwrite `"my_results"` and start from scratch, set `overwrite=True` when initialising the `Registration` class.

5. Run the registration.
```
reg.register()
```
This step can also be performed automatically by setting `auto=True` when creating the `Registration` object. By default, registration steps will only be performed if there is no existing output transform in the directory for that step; to force rerunning of the registration, set `force=True` when running the `register()` function.

4. Viewing the images
A `Registration` object can be used to launch an interactive viewer showing either the initial or final images side-by-side. To view the initial fixed and moving images, run
```
reg.view_init()
```
To view the fixed image and transformed moving image, run:
```
reg.view_result()
```
You can also view an intermediate step by setting `step` to the number or name of the registration step. To view the list of steps for this registration, print
`reg.steps`. E.g. to view the results of the first step:
```
reg.view_result(step=0)
```

5. Transforming structure sets
The transform found by the registration can be used to transform images, ROIs, and structure sets via the `transform()` function. Structure sets can be loaded from dicom with the `StructureSet` class imported from `skrt`:
```
from skrt import StructureSet
structs_moving = StructureSet("some/location/moving_structs.dcm")
structs_fixed = StructureSet("some/location/fixed_structs.dcm")
```
A structure set can then be transformed by running
```
structs_transformed = reg.transform(structs_moving)
```
which returns a new structure set object. This can be written to a file by running `structs_transformed.write()`.

StructureSet objects have a built-in function to calculate comparison metrics with respect to another structure set, e.g. to compare the transformed structure set to the fixed image structure set:
```
df = structs_fixed.get_comparison(structs_transformed)
```
Here, `df` is a [pandas](https://pandas.pydata.org/) DataFrame. By default, this will contain the dice scores and centroid distances for each structure in the structure sets. The desired metrics can be set via the `metrics` argument; see documentation for `skrt.ROI.get_comparison` for more details. The DataFrame is written to a .csv file by calling
```
df.to_csv("metrics.csv")
```
