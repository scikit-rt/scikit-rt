# scikit-rt

A suite of tools for loading, plotting, and analysing medical imaging data.

This work was supported by Cancer Research UK RadNet Cambridge [C17918/A28870].

## Installation

Package installation provides full code access.  Docker installation
allows for scikit-rt code to be used from a jupyter notebook.

### Package installation

Installation via a minconda environment is recommended. If not using miniconda, ensure that you have at least python 3.8 and [pip](https://pypi.org/project/pip/).

#### 1. Miniconda setup (recommended)

1. Install miniconda by downloading from https://docs.conda.io/en/latest/miniconda.html .
2. Open the Anaconda Prompt app (indows) or a terminal (Mac/Linux).
3. Create a new environment called `skrt` (can replace this with a name of your choice) with python 3.9 by running:
```
conda create --name skrt python=3.9
```
4. Activate the environment by running:
```
conda activate skrt
```

#### 2. Package installation via `pip`

1. Open Anaconda Prompt/a terminal, if not already open.
2. If using miniconda, ensure that you have first activated your environment by running step 1.4 from above.
3. Install scikit-rt by running:
```
pip install scikit-rt
```
4. If using Windows, you will also need to install shapely by running:
```
conda install -c conda-forge shapely
```
5. If you want to be able to calculate the STAPLE consensus of contours, you must also install SimpleITK:
```
pip install simpleitk
```


#### 3. Using scikit-rt in code

- If using miniconda, ensure you have opened Anaconda Prompt/a terminal and activated your environment (step 1.4).
- Options for running python are:
    - Launch a jupyter notebook server by typing `jupyter notebook`.
    - Launch an iPython session by typing `ipython`.
    - Create a python script and run it by typing `python my_script.py`.
- To check that scikit-rt is correctly installed, trying running the command `import skrt` via any of the three methods above.

#### 4. Elastix installation (optional -- needed for image registration functionality)

1. Download the version of elastix matching your operating system from https://github.com/SuperElastix/elastix/releases/tag/5.0.1
2. Unzip the downloaded folder and place it somewhere. Make a note of where you stored it.
3. On Mac/Linux, you can also add the location of the elastix install to your environment by adding the following lines to your `~.zshrc` or `.bashrc`:

On Mac:
```
export PATH=${PATH}:/path/to/elastix/directory/bin
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/path/to/elastix/directory/lib
```
On Linux:
```
export PATH=${PATH}:/path/to/elastix/directory/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/elastix/directory/lib
```
4. If you are on Windows or didn't do step 3, any time you want to use image registration in scikit-rt, you will need to run:
```
from skrt.registration import set_elastix_dir
set_elastix_dir('path/to/elastix/directory')
```

### Updating scikit-rt

Scikit-rt is in active development. To install the lastest version, run:
```
pip install --upgrade scikit-rt
```

### Docker installation

You need to have an installation of [Docker](https://www.docker.com/).

Download the Scikit-rt docker image with:

```
docker pull ghcr.io/scikit-rt/scikit-rt:latest
```

Choose a work directory, copy here any data that you would want to be able to
access with Scikit-rt, then start a docker container:

```
docker run -v /path/to/work/directory:/home/jovyan/workdir -p 8888:8888 ghcr.io/scikit-rt/scikit-rt
```

Copy the last URL listed (starting: http://127.0.0.1:8888), and point a browser to this URL.  This should open a jupyter lab session, where you import
Scikit-rt modules.

Notes on arguments passed to `docker run`:
- The argument to `-v` maps a local work directory (/path/to/work/directory) to 
the work directory of the docker container (/home/jovyan/workdir).  The former
is chosen by the user; the latter is fixed.  Paths should always be absolute
 (not relative).
- The argument to `-p` maps the server port (8888) on the local machine to the
server port (8888) on the container side.

## Table of contents
1) [Images](#1-images)
2) [Regions of interest (ROIs) and structure sets](#2-rois-and-structure-sets)
3) [Patients and studies](#3-patients)
4) [Synthetic data](#4-synthetic-data)
5) [Image registration](docs/image_registration.md)

## 1. Images

The Image class can be used to read and write medical images from DICOM, nifti,
or NumPy format. These images can be plotted and compared.

To load the Image class:
```
from skrt import Image
```

Images will be processed into a consistent format:

- The `Image.data` property contains a numpy array, which stores (y, x, z) in (row, column, slice) respectively. Note that numpy elements are indexed in order (row, column, slice); so if you did `Image.data[i, j, k]`, `i` would correspond to y index, `j` would correspond to x index, `k` would correspond to z index.
- The `Image.affine` property contains a 4x4 matrix that can convert a (row, column, slice) index to an (x, y, z) position. This will always be diagonal, so (0, 0) contains x voxel size etc, (0, 3) contains x origin.
- The `voxel_size` and `origin` properties are the diagonal and third column, respectively; they give voxel sizes and origin position in order (x, y, z).
- The `n_voxels` property containins the number of voxels in the (x, y, z) directions (same as `Image.data.shape`, but with 0 and 1 swapped).

In the standard dicom-style configuration (Left, Posterior, Superior):
- The x-axis increases along each column, and points towards the patient's left (i.e. towards the heart, away from the liver).
- The y-axis increase down each row, and points from the patient's front to back (posterior).
- The z-axis increases along the slice index, and points from the patient's feet to head (superior).

A canonical nifti-style array and affine can be obtained by running `Image.get_nifti_array_and_affine()`. By convention, this points in the same z direction but has x and y axes reversed (Right, Anterior, Superior). In the affine matrix, the x and y origins are therefore defined as being at the opposite end of the scale.

Note that positions can also be specified in terms of slice number:
- For x and y, slice number is just numpy array index + 1 (slice number ranges from 1 - n_voxels, whereas array index ranges from 0 - n_voxels-1)
- For z, by convention the slice numbers increases from 1 at the head to n_voxels at the feet, so it is in the opposite direction to the array index (convert as n_voxels[2] - idx).

### Loading from a file

An image can be loaded from a dicom, nifti, or numpy file via:
```
im = Image(filepath)
```

If the dicom file is part of a series, any other files in that series in the same directory will also be loaded. Alternatively, you can give the path to a directory containing multiple dicom files. The first dicom file in that directory alphabetically will be loaded along with any other files in its series.

### Loading from an array

An image can also be loaded from a numpy array. By default, it will be taken to have origin (0, 0, 0) and voxel sizes (1, 1, 1) mm; otherwise, these can be set manually, either via:
```
im = Image(array, voxel_size=(1.5, 1.5, 3), origin=(-100, -100, 40))
```
where the origin/voxel size lists are in order (x, y, z).

The origin and voxel sizes can also be specified via an affine matrix, e.g.
```
affine = np.array([
    [1.5, 0, 0, -100],
    [0, 1.5, 0, -100],
    [0, 0, 3, 40],
    [0, 0, 0, 1]
])
im = Image(array, affine=affine)
```
where the first row of the affine matrix contains the `x` voxel size and origin, second row contains `y`, third row contains `z`.

### Plotting
To plot a slice of the image, you need to specify the orientation (`x-y`, `y-z`, or `x-z`; default `x-y`) and either the slice number, array index, or position in mm (by default, the central slice in the chosen orientation will be plotted).
e.g.
```
im.plot('y-z', idx=5)
```

### Writing out image data
Images can be written out with the `Image.write(filename)` function. The output filetype will be inferred from the filename.

#### Writing to dicom
If `filename` ends in `.dcm` or is a directory (i.e. has no extension), the image will be written in dicom format. Each `x-y` slice will be written to a separate file labelled by slice number, e.g. slice 1 (corresponding to `[:, :, -1]` in the image array) would be saved to `1.dcm`.

The path to a dicom file from which to take the header can be specified via the `header_source` parameter. If no path is given but the input source for the Image was a dicom file, the header will be taken from the source. Otherwise (e.g. if the file was loaded from a nifti and no `header_source` is given), a brand new header with new UIDs will be created. In that case, you can set the following info for that header:
- `patient_id`
- `modality`
- `root_uid` (an ID unique to your institue that will prefix the generated dicom UIDs so that they are globally unique; one can be obtained here: https://www.medicalconnections.co.uk/FreeUID/)

#### Writing to nifti
If `filename` ends in `.nii` or `.nii.gz`, the image will be written to nifti. The nifti will be in canonical format, i.e. in Right, Anterior, Superior configuration. (Note that this means the nifti you write out may not be the same as the one you read in).

#### Writing to a numpy array
If `filename` ends in `.npy`, the image array will be written to a numpy binary file. To write a canonical nifti-style array instead of the dicom-style array, set `nifti_array=True`. If `set_geometry` is `True` (which it is by default), a text file will be written to the same directory as the `.npy` file containing the voxel sizes and origin.

## 2. Regions of interest (ROIs) and structure sets

A structure/region of interest (ROI) can be represented by either a set of contour points or a binary mask. The `ROI` class allows an ROI to be loaded from either of these sources and converted from one to the other.

### The ROI class

An `ROI` object, which contains a single ROI, can be created via:
```
from skrt import ROI
roi = ROI(source)
```

The `source` can be from any of the following:
- A dicom structure set file containing one or more sets of contours;
- A dictionary of contours, where the keys are slice positions in mm and the values are lists of lists of contour points for each contour on that slice;
- A nifti file containing a binary mask;
- A numpy file containing a binary mask;
- A numpy array.

If the input object is a dicom file, the name of the ROI within that file must be given. In addition, if the input source is a dicom file or contour dictionary, in order to convert the contours to a mask, the user must provide either of the following arguments to the `ROI` creation:
- `image`: Assign an `Image` object associated with this ROI. The created mask will have the same dimensions as this image.
- `shape`, `origin`, and `voxel_sizes`: This tells the ROI the dimensions of the pixel array to create when making the mask.

Additional useful arguments are:
- `color`: set the colour of this ROI for plotting. If not specified, this will either be read from the dicom file (if given) or the ROI will be assigned a unique colour.
- `name`: set the name of this ROI. If the input is a dicom structure set file, this name will be used to select the correct ROI from the file. If not given, the ROI will be given a unique name ("ROI 1" etc)
- `load`: if `True` (default), the input ROI will be converted into a mask/contours automatically upon creation. This can be time consuming if you are creating many ROIs, so setting it to `False` can save time and allow ROIs to be processed later on-demand.

An `Image` object associated with this ROI can be set at any time via:
```
roi.image = image
```
This image can optionally be plotted in the background when the ROI is plotted. 

#### Plotting an ROI

ROIs can be plotted as contours or a mask or both, and with or without an associated image behind them. The simplest plot command is:
```
roi.plot()
```
This will plot a contour in the x-y plane, e.g.:

<img src="docs/images/struct_contour.png" alt="ROI contour plot" height="500"/>

Additional options are:
- `include_image`: boolean, set to `True` to plot the associated image (if one is assigned) in the background; 
- `view`: specify the orientation (`x-y`, `y-z`, or `x-z`);
- `sl`, `pos`, or `idx`: specify the slice number, slice position in mm, or slice index;
- `zoom`: amount by which to zoom in (will automatically zoom in on the centre of the ROI).

The following plot types are available, set via the `plot_type` argument:
- `contour`: plot a contour
- `mask`: plot a binary mask
- `filled`: plot a contour on top of a semi-transparent mask
- `centroid`: plot a contour with the centroid marked by a cross
- `filled centroid`: plot a contour and centroid on top of a semi-transparent mask

Example contour plot with `include_image=True`:

<img src="docs/images/struct_contour_with_image.png" alt="ROI contour plot with image" height="500"/>

Example mask plot:

<img src="docs/images/struct_mask.png" alt="ROI binary mask plot" height="500"/>


#### Writing out an ROI

An ROI can be written to a nifti or numpy file as a binary mask. This can be done either by specifying a filename with an appropriate extension, e.g.
```
roi.write('my_roi.nii')
```
would automatically write to a nifti file.

Alternatively, the ROI's name can be used to create the filename. This is done by just providing an extension, e.g. if an ROI were called "heart", the following command:
```
roi.write(ext='.nii')
```
would produce a file called `heart.nii` containing the ROI as a binary mask. An output directory can also be optionally provided, e.g.
```
roi.write(outdir='my_rois', ext='.npy')
```
would produce a binary numpy file at `my_rois/heart.npy`.

#### Getting geometric properties

The `ROI` class has various methods for obtaining geometric properties of the ROI:

- **Centroid**: get the 3D centroid coordinates (i.e. the centre-of-mass of the ROI) in mm via `roi.get_centroid()`, or get the 2D coordinates for a single slice via `roi.get_centroid(view='x-y', sl=10)`, for example (the slice position in mm `pos` or slice index `idx` can be given instead of slice number `sl`)

- **Centre**: get the 3D midpoint coordinates (i.e. the mean of the maximum extents in each direction, rather than centre of mass) via `roi.get_centre()`. The 2D midpoint of a slice is obtained in a similar way to the centroid, e.g. `roi.get_centre(view='x-y', sl=10)`.

- **Volume**: `roi.get_volume(units)`, where `units` can either be `mm` or `voxels` (default `mm`).

- **Area**: get the area of a given slice of an ROI by running e.g. `roi.get_area(view='x-y', sl=10, units='mm')`. To get the area of the central slice, can simply run `roi.get_area()`.

- **Length**: get ROI length along a given axis by running `roi.get_length(axis, units)` where `axis` is `x`, `y`, or `z` and `units` is `mm` or `voxels`.

### Structure Sets: the StructureSet class

A structure set is an object that contains multiple ROIs. This is implemented via the `StructureSet` class. 

#### Loading a structure set

A structure set is created via
```
from skrt import StructureSet
structure_set = StructureSet(source)
```

The source can be:
- The path to a dicom structure set file containing multiple ROIs;
- The path to a directory containing one or more nifti or numpy files, each containing a binary ROI mask;
- A list of paths to nifti or numpy ROI mask files.

In addition, more ROIs can be added later via:
```
structure_set.add_rois(source)
```
where `source` is any of the above source types.

Alternatively, single ROIs can be added at a time via any of the valid `ROI` sources (see above), and with any of the `ROI` initialisation arguments. An empty `StructureSet` can be created and then populated, e.g.
```
structure_set = StructureSet()
structure_set.add_roi('heart.nii', color='red')
structure_set.add_roi('some_structs.dcm', name='lung')
```

The `StructureSet` can also be associated with an `Image` object by specifying the `image` argument upon creation, or running `structure_set.set_image(image)`. This image will be assigned to all ROIs in the structure set.

#### Creating a filtered copy of a structure set

Sometimes you may wish to load many ROIs (e.g. from a dicom structure set file) and then filter them by name. This is done via:
```
structure_set.filter_rois(to_keep, to_remove)
```
where `to_keep` and `to_remove` are optional lists containing ROI names, or wildcards with the `*` character. First, all of the ROIs belonging to `structure_set` are checked and only kept if they match the names or wildcards in `to_keep`. The remaining ROIs are then removed if their names match the names or wildcards in `to_remove`.

To restore a structure set to its original state (i.e. reload it from its source), run `structure_set.reset()`.

#### Renaming ROIs

ROIs can be renamed by mapping from one or more possible original names to a single final name. In this way, multiple structure sets where the same ROI might have different names can be standardised to have the same ROI names.

For example, let's say you wish to rename the right parotid gland to `right_parotid`, but you know that it has a variety of names across different structure sets. You could do this with the following (assuming `my_structure_sets` is a list of `StructureSet` objects:
```
names_map = {
    'right_parotid': ['right*parotid', 'parotid*right', 'R parotid', 'parotid_R']
}
for structure_set in my_structure_sets:
    structure_set.rename_rois(names_map)
```

By default, only one ROI per structure set will be renamed; for example, if a structure set for some reason contained both `right parotid` and `R parotid`, only the first in the list (`right parotid`) would be renamed. This behaviour can be turned off by setting `first_match_only=False`; beware this could lead to duplicate ROI names.

You can also choose to discard any ROIs that aren't in your renaming map by setting `keep_renamed_only=True`.

#### Getting ROIs

Get a list of the `ROI` objects belonging to the structure set:
```
structure_set.get_rois()
```

Get a list of names of the `ROI` objects:
```
structure_set.get_roi_names()
```

Get a dictionary of `ROI` objects with their names as keys:
```
structure_set.get_roi_dict()
```

Get an `ROI` object with a specific name:
```
structure_set.get_roi(name)
```

Print the `ROI` names:
```
structure_set.print_rois()
```


#### Filtering a structure set

A `StructureSet` object can be copied to a new `StructureSet`, optionally with some ROIs filtered/renamed (you might want to do this if you want to preserve the original structure set, while making a filtered version too), via:
```
filtered = structure_set.filter(names, to_keep, to_remove, keep_renamed_only)
```

#### Writing a structure set

The ROIs in a structure set can be written to a directory of nifti or numpy files, via:
```
structure_set.write(outdir, ext)
```
where `outdir` is the output directory and `ext` is either `.nii`, `.nii.gz` or `.npy`. Dicom writing will be supported in future.


#### Assigning StructureSets to an Image

Just as ROIs and structure sets can be associated with an image, the `Image` object can be associated with one or more `StructureSet` objects. This is done via:
```
from skrt import Image, StructureSet

image = Image("some_image.nii")
structure_set = StructureSet("roi_directory")

image.add_structure_set(structure_set)
```

Now, when the `Image` is plotted, the ROIs in its structure set(s) can be plotted on top. To plot all structure sets, run:
```
image.plot(structure_set='all')
```
Note that this could be slow if there are many structure sets containing many structures.

To plot just one structure set, you can also provide the index of the structure set in the list of structure sets belonging to the image, e.g. to plot the most recently added structure set:
```
image.plot(structure_set=-1)
```

To add a legend to the plot, set `roi_legend=True`.

The image's structure sets can be cleared at any time via
```
image.clear_structure_sets()
```

## 3. Patients and studies

The `Patient` and `Study` classes allow multiple medical images and structure sets associated with a single patient to be read into one object. 

### Expected file structure

#### Patient and study file structure

The files for a patient should be sorted into a specific structure in order for the `Patient` class to be able to read them in correctly.

The top level directory represents the entire **patient**; this should be a directory whose name is the patient's ID.

The next one or two levels represent **studies**. A study is identified by a directory whose name is a timestamp, which is a string with format `YYYYMMDD_hhmmss`. These directories can either appear within the patient directory, or be nested in a further level of directories, for example if you wished to separate groups of studies.

A valid file structure could look like this:
```
mypatient1
--- 20200416_120350
--- 20200528_100845
```
This would represent a patient with ID `mypatient1`, with two studies, one taken on 16/04/2020 and one taken on 28/05/2020.

Another valid file structure could be:
```
mypatient1
--- planning
------ 20200416_120350
------ 20200528_100845
--- relapse
------ 20211020_093028
```
This patient would have three studies, two in the "planning" category and two in the "relapse" category.

#### Files within a study

Each study can contain images of various imaging modalities, and associated structure sets. Within a study directory, there can be three "special" directories, named `RTSTRUCT`, `RTDOSE`, and `RTPLAN` (currently only `RTSTRUCT` does anything; the others will be updated soon), containing structure sets, dose fields, and radiotherapy plans, respectively.

Any other directories within the study directory are taken to represent an **imaging modality**. The structure sets associated with this modality should be nested inside the `RTSTRUCT` directory inside directories with the same name as the image directories. The images and structure sets themselves should be further nested inside a timestamp directory representing the time at which that image was taken.

For example, if a study containined both CT and MR images, as well as two structure sets associated with the CT image, the file structure should be as follows:
```
20200416_120350
--- CT
------ 20200416_120350
--------- 1.dcm
--------- 2.dcm ... etc
--- MR
------ 20200417_160329
--------- 1.dcm
--------- 2.dcm ... etc
--- RTSTRUCT
------ CT
---------- 20200416_120350
-------------- RTSTRUCT_20200512_134729.dcm
-------------- RTSTRUCT_20200517_162739.dcm
```

### The Patient class

A `Patient` object is created by providing the path to the top-level patient directory:
```
from skrt import Patient
p = Patient('mypatient1')
```

A list of the patient's associated studies is stored in `p.studies`.

Additional properties can be accessed:
- Patient ID: `p.id`
- Patient sex: `p.get_sex()`
- Patient age: `p.get_age()`
- Patient birth date: `p.get_birth_date()`

#### Writing a patient tree

A patient's files can be written out in nifti or numpy format. By default, files will be written to compress nifti (`.nii.gz`) files. This is done via the `write` method:
```
p.write('some_dir')
```
where `some_dir` is the directory in which the new patient folder will be created (if not given, it will be created in the current directory).

By default, all imaging modalities will be written. To ignore a specific modality, a list `to_ignore` can be provided, e.g. to ignore any MR images:
```
p.write('some_dir', to_ignore=['MR'])
```

To write out structure sets as well as images, the `structure_set` argument should be set. This can be either:
- `'all'`: write all structure sets.
- The index of the structure set (e.g. to write only the newest structure set for each image, set `structure_set=-1`)
- A list of indices of structure sets to write (e.g. to write only the oldest and newest, set `structure_set=[0, -1]`)

By default, no structure sets will be written, as conversion of structures from contours to masks can be slow for large structures.


### The Study class

A `Study` object stores images and structure sets. A list of studies can be extracted from a `Patient` object via the property `Patient.studies`. You can access the study's path via `Study.path`. If the study was nested inside a subdirectory, the name of that subdirectory is accessed via `Study.subdir`.

#### Images

For each imaging modalitiy subdirectory inside the study, a new class property will be created to contain a list of images of that modality, called `{modality}_images`, where the modality is taken from the subdirectory name (note, this is always converted to lower case). E.g. if there were directories called `CT` and `MR`, the `Study` object would have properties `ct_images` and `mr_images` containing lists of `Image` objects.

#### Structure sets

The study's structure sets can be accessed in two ways. Firstly, the structure sets associated with an image can be extracted from the `structs` property of the `Image` itself; this is a list of `StructureSet` objects. E.g. to get the newest structure set for the oldest CT image in the oldest study, you could run:
```
p = Patient('mypatient1')
s = p.studies[0]
structure_set = s.ct_images[0].structure_sets[-1]
```

In addition, structure sets associated with each imaginging modality will be stored in a property of the `Study` object called `{modality}_structure_sets`. E.g. to get the oldest CT-related structure set, you could run:
```
structure_set = s.ct_structure_sets[0]
```

The `StructureSet` object also has an associated `image` property (`structure_set.image`), which can be used to find out which `Image` is associated with that structure set.

## 4. Synthetic data

### 4.1 Synthetic images

The `SyntheticImage` class enables the creation of images containing simple geometric shapes.

<img src="docs/images/synthetic_example.png" alt="synthetic image with a sphere and a cube" height="500"/>

#### Creating a synthetic image

To create an empty image, load the `SyntheticImage` class and specify the desired image shape in order (x, y, z), e.g.

```
from skrt.simulation import SyntheticImage

sim = SyntheticImage((250, 250, 50))
```

The following arguments can be used to adjust the image's properties:
- `voxel_size`: voxel sizes in mm in order (x, y, z); default (1, 1, 1).
- `origin`: position of the top-left voxel in mm; default (0, 0, 0).
- `intensity`: value of the background voxels of the image.
- `noise_std`: standard deviation of Gaussian noise to apply to the image. This noise will also be added on top of any shapes. (Can be changed later by altering the `sim.noise_std` property).

##### Adding shapes

The `SyntheticImage` object has various methods for adding geometric shapes. Each shape has the following arguments:
- `intensity`: intensity value with which to fill the voxels of this shape.
- `above`: if `True`, this shape will be overlaid on top of all existing shapes; otherwise, it will be added below all other shapes.

The available shapes and their specific arguments are:
- **Sphere**: `sim.add_sphere(radius, centre=None)`
    - `radius`: radius of the sphere in mm.
    - `centre`: position of the centre of the sphere in mm (if `None`, the sphere will be placed in the centre of the image).
- **Cuboid**: `sim.add_cuboid(side_length, centre=None)`
    - `side_length`: side length in mm. Can either be a single value (to create a cube) or a list of the (x, y, z) side lengths.
    - `centre`: position of the centre of the cuboid in mm (if `None`, the cuboid will be placed in the centre of the image).
- **Cube**: `sim.add_cube(side_length, centre=None)`
    - Same as `add_cuboid`.
- **Cylinder**: `sim.add_cylinder(radius, length, axis='z', centre=None)`
    - `radius`: radius of the cylinder in mm.
    - `length`: length of the cylinder in mm.
    - `axis`: either `'x'`, `'y'`, or `'z'`; direction along which the length of the cylinder should lie.
    - `centre`: position of the centre of the cylinder in mm (if `None`, the cylinder will be placed in the centre of the image).
- **Grid**: `sim.add_grid(spacing, thickness=1, axis=None)`
    - `spacing`: grid spacing in mm. Can either be a single value, or list of (x, y, z) spacings.
    - `thickenss`: gridline thickness in mm. Can either be a single value, or list of (x, y, z) thicknesses.
    - `axis`: if None, gridlines will be created in all three directions; if set to `'x'`, `'y'`, or `'z'`, grids will only be created in the plane orthogonal to the chosen axis, such that a solid grid runs through the image along that axis.

To remove all shapes from the image, run
```
sim.reset()
```

##### Plotting

The `SyntheticImage` class inherits from the `Image` class, and can thus be plotted in the same way by calling

```
sim.plot()
```

along with any of the arguments available to the `Image` plotting method.

##### Rotations and translations

Rotations and translations can be applied to the image using:

```
sim.translate(dx, dy, dz)
```

or 
```
sim.rotate(pitch, yaw, roll)
```

Rotations and translations can be removed by running
```
sim.reset_transforms()
```

##### Getting the image array

To obtain a NumPy array containing the image data, run
```
array = sim.get_data()
```

This array will contain all of the added shapes, as well as having any rotations, translations, and noise applied.

##### Saving

The synthetic image can be written to an image file by running
```
sim.write(outname)
```

The output name `outname` can be:
- A path ending in `.nii` or `.nii.gz`: image will be written to a nifti file.
- A path ending in `.npy`: image will be written to a binary NumPy file.
- A path to a directory: image will be written to dicom files (one file per slice) inside the directory.

The `write` function can also take any of the arguments of the `Image.write()` function.


#### Adding structures

When shapes are added to the image, they can also be set as ROIs. This allows you to:
- Plot them as contours or masks on top of the image;
- Access a `StructureSet` object containing the ROIs;
- Write ROIs out separately as masks or as a dicom structure set file.

##### Single ROIs

To assign a shape as an ROI, you can either give it a name upon creation, e.g.:

```
sim.add_sphere(50, name='my_sphere')
```

or set the `is_roi` property to `True` (the ROI will then be given an automatic named based on its shape type):
```
sim.add_sphere(50, is_roi=True)
```

When `sim.plot()` is called, its ROIs will automatically be plotted as contours. Some useful extra options to the `plot` function are:
- `roi_plot_type`: set plot type to any of the valid `ROI` plotting types (mask, contour, filled, centroid, filled centroid).
- `centre_on_roi`: name of ROI on which the plotted image should be centred.
- `roi_legend`: set to `True` to draw a legend containing the strutcure names.

##### Grouped shapes

Multiple shapes can be combined to create a single ROI. To do this, set the `group` argument to some string when creating shapes. Any shapes created with the same `group` name will be added to this group.

E.g. to create a single ROI called "two_cubes" out of two cubes centred at different positions, run:
```
sim.add_cube(10, centre=(5, 5, 5), group='two_cubes')
sim.add_cube(10, centre=(7, 7, 5)), group='two_cubes')
```

##### Getting ROIs

To get a `StructureSet` object containing all of the ROIs belonging to the image, run:
```
structure_set = sim.get_structure_set()
```

You can also access a single ROI as an `ROI` object by running:
```
roi = sim.get_roi(name)
```

##### Saving ROIs

When the `SyntheticImage.write()` function is called, the ROIs belonging to that image will also be written. If the provided `outname` is that of a nifti or NumPy file, the ROIs will be written to nifti or Numpy files, respectively, inside a directory with the same name as `outname` but with the extension removed.

### 4.2 Synthetic Patient objects

The Patient class can be used to create a custom patient object from scratch, rather than loading a prexisting patient.

To do this, first create a blank Patient object with your chosen ID:
```
from skrt import Patient

p = Patient("my_id")
```

Studies can then be added to this object. Optional arguments for adding a study are:
- `subdir`: a custom subdirectory in which the study will be nested when the Patient tree is written;
- `timestamp`: a custom timestamp (if not set, this will be automatically generated)
- `images`: a list of Image objects to associate with this study (note, Images can also be added to the Study later)
- `image_type`: the type of image correspoding to the Images in `images`, if used (e.g. `CT`); this will determine the name of the directory in which the images are saved when the Patient tree is written.

E.g. to add one study containing a single synthetic image with one ROI:
```
from skrt.simulation import SyntheticImage

im = SyntheticImage((100, 100, 30))
im.add_sphere(radius=10, name="my_sphere")

p.add_study("my_study", images=[im], image_type="MR")
```

The patient tree can then be written out:
```
p.write(outdir="some_dir")
```

This will create a patient directory `somedir/my_id` containing the added study and its corresponding image and structure set.
