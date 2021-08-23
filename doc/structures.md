# Structures and structure sets

A structure/region of interest (ROI) can be represented by either a set of contour points or a binary mask. The `ROI` class allows a structure to be loaded from either of these sources and converted from one to the other.

## The ROI class

An `ROI` object, which contains a single structure, can be created via:
```
from quickviewer.prototype import ROI
roi = ROI(source)
```

The `source` can be from any of the following:
- A dicom RTStruct file containing one or more sets of contours;
- A dictionary of contours, where the keys are slice positions in mm and the values are lists of lists of contour points for each contour on that slice;
- A nifti file containing a binary mask;
- A numpy file containing a binary mask;
- A numpy array.

If the input object is a dicom file, the name of the ROI within that file must be given. In addition, if the input source is a dicom file or contour dictionary, in order to convert the contours to a mask, the user must provide either of the following arguments to the `ROI` creation:
- `image`: Assign an `Image` object associated with this ROI. The created mask will have the same dimensions as this image.
- `shape`, `origin`, and `voxel_sizes`: This tells the ROI the dimensions of the pixel array to create when making the mask.

Additional useful arguments are:
- `color`: set the colour of this ROI for plotting. If not specified, this will either be read from the dicom file (if given) or the ROI will be assigned a unique colour.
- `name`: set the name of this ROI. If the input is a dicom RTStruct file, this name will be used to select the correct ROI from the file. If not given, the ROI will be given a unique name ("ROI 1" etc)
- `load`: if `True` (default), the input ROI will be converted into a mask/contours automatically upon creation. This can be time consuming if you are creating many ROIs, to setting it to `False` can save time and allow ROIs to be processed later on-demand.

An `Image` object associated with this ROI can be set at any time via:
```
roi.image = image
```
This image can optionally be plotted in the background when the ROI is plotted. 

### Plotting an ROI

ROIs can be plotted as contours or a mask or both, and with or without an associate image behind them. The simplest plot command is:
```
roi.plot()
```
This will plot a contour in the x-y plane, e.g.:

<img src="images/struct_contour.png" alt="ROI contour plot" height="500"/>

Additional options are:
- `include_image`: boolean, set to `True` to plot the associated image (if one is assigned) in the background; 
- `view`: specify the orientation (`x-y`, `y-z`, or `x-z`);
- `sl`, `pos`, or `idx`: specify the slice number, slice position in mm, or slice index;
- `zoom`: amount by which to zoom in (will automatically zoom in on the centre of the structure).

The following plot types are available, set via the `plot_type` argument:
- `contour`: plot a contour
- `mask`: plot a binary mask
- `filled`: plot a contour on top of a semi-transparent mask
- `centroid`: plot a contour with the centroid marked by a cross
- `filled centroid`: plot a contour and centroid on top of a semi-transparent mask

Example contour plot with `include_image=True`:

<img src="images/struct_contour_with_image.png" alt="ROI contour plot with image" height="500"/>

Example mask plot:

<img src="images/struct_mask.png" alt="ROI binary mask plot" height="500"/>


### Writing out an ROI

An ROI can be written to a nifti or numpy file as a binary mask. This can either be done by specifying a filename with an appropriate extension, e.g.
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

### Getting geometric properties

The `ROI` class has various methods for obtaining geometric properties of the ROI:

- **Centroid**: get the 3D centroid coordinates (i.e. the centre-of-mass of the ROI) in mm via `roi.get_centroid()`, or get the 2D coordinates for a single slice via `roi.get_centroid(view='x-y', sl=10)`, for example (the slice position in mm `pos` or slice index `idx` can be given instead of slice number `sl`)

- **Centre**: get the 3D midpoint coordinates (i.e. the mean of the maximum extents in each direction, rather than centre of mass) via `roi.get_centre()`. The 2D midpoint of a slice is obtained in a similar way to the centroid, e.g. `roi.get_centre(view='x-y', sl=10)`.

- **Volume**: `roi.get_volume(units)`, where `units` can either be `mm` or `voxels` (default `mm`).

- **Area**: get the area of a given slice of a structure by running e.g. `roi.get_area(view='x-y', sl=10, units='mm')`. To get the area of the central slice, can simply run `roi.get_area()`.

- **Length**: get structure length along a given axis by running `roi.get_length(axis, units)` where `axis` is `x`, `y`, or `z` and `units` is `mm` or `voxels`.

## Structure Sets: the RtStruct class

A structure set is an object that contains multiple ROIs. This is done via the `RtStruct` class. 

### Loading a structure set

A structure set is created via
```
from quickviewer.prototype import RtStruct
rtstruct = RtStruct(source)
```

The source can be:
- The path to a dicom RtStruct file containing multiple ROIs;
- The path to a directory containing one or more nifti or numpy files, each containing a binary ROI mask;
- A list of paths to nifti or numpy ROI mask files.

In addition, more ROIs can be added later via:
```
rtstruct.add_structs(source)
```
where `source` is any of the above source types.

Alternatively, single ROIs can be added at a time via any of the valid `ROI` sources (see above), and with any of the `ROI` initialisation arguments. An empty can `RtStruct` can be created and then populated, e.g.
```
rtstruct = RtStruct()
rtstruct.add_struct('heart.nii', color='red')
rtstruct.add_struct('some_structs.dcm', name='lung')
```

The `RtStruct` can also be associated with an `Image` object by specifying the `image` argument upon creation, or running `rtstruct.set_image(image)`. This image will be assigned to all ROIs in the structure set.

### Filtering a structure set

Sometimes you may wish to load many ROIs (e.g. from a dicom RtStruct file) and then filter them by name. This is done via:
```
rtstruct.filter_structs(to_keep, to_remove)
```
where `to_keep` and `to_remove` are optional lists containing structure names, or wildcards with the `*` character. First, all of the ROIs belonging to `rtstruct` are checked and only kept if they match the names or wildcards in `to_keep`. The remaining ROIs are then removed if their names match the names or wildcards in `to_remove`.

To restore a structure set to its original state (i.e. reload it from its source), run `rtstruct.reset()`.

### Renaming ROIs

ROIs can be renamed by mapping from one or more possible original names to a single final name. In this way, multiple structure sets where the same ROI might have different names can be standardised to have the same ROI names.

For example, let's say you wish to rename the right parotid gland to `right_parotid`, but you know that it has a variety of names across different structure sets. You could do this with the following (assuming my_rtstructs is a list of `RtStruct` objects:
```
names_map = {
    'right_parotid': ['right*parotid', 'parotid*right', 'R parotid', 'parotid_R']
}
for rtstruct in my_rtstructs:
    rtstruct.rename_structs(names_map)
```

By default, only one ROI per structure set will be renamed; for example, if a structure set for some reason contained both `right parotid` and `R parotid`, onl the first in the list (`right parotid`) would be renamed. This behaviour can be turned off by setting `first_match_only=False`; beware this could lead to duplicate structure names.

You can also choose to discard any structures that aren't in your renaming map by setting `keep_renamed_only=True`.

### Getting ROIs

Get a list of the `ROI` objects belonging to the structure set:
```
rtstruct.get_structs()
```

Get a list of names of the `ROI` objects:
```
rtstruct.get_struct_names()
```

Get a dictionary of `ROI` objects with their names as keys:
```
rtstruct.get_struct_dict()
```

Get an `ROI` object with a specific name:
```
rtstruct.get_struct(name)
```

Print the `ROI` names:
```
rtstruct.print_structs()
```


### Copying a structure set

An `RtStruct` object can be copied to a new `RtStruct`, optionally with some structures filtered/renamed (you might want to do this if you want to preserve the original structure set, while making a filtered version too), via:
```
rtstruct_filtered = rtstruct.copy(names, to_keep, to_remove, keep_renamed_only)
```

### Writing a structure set

The ROIs in a structure set can be written to a directory of nifti or numpy files, via:
```
rtstruct.write(outdir, ext)
```
where `outdir` is the output directory and `ext` is either `.nii`, `.nii.gz` or `.npy`. Dicom writing will be supported in future.


### Assigning RtStructs to an Image

Just as ROIs and structure sets can be associated with an image, the `Image` object can be associated with one or more `RtStruct` objects. This is done via:
```
from quickviewer.prototype import Image, RtStruct

image = Image("some_image.nii")
rtstruct = RtStruct("roi_directory")

image.add_structs(rtstruct)
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

To add a legend to the plot, set `struct_legend=True`.

The image's structure sets can be cleared at any time via
```
image.clear_structs()
```
