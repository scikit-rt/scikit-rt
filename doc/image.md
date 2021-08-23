# The Image class

The Image class can be imported from the prototype submodule:
```
from quickviewer.prototype import Image
```

Images can be loaded from dicom files, nifti files, or numpy arrays, and will be put into a consistent format. This format is:

- The `Image.data` property contains a numpy array, which stores (y, x, z) in (row, column, slice) respectively. Note that numpy elements are indexed in order (row, column, slice); so if you did `Image.data[i, j, k]`, `i` would correspond to y index, `j` would correspond to x index, `k` would correspond to z index.
- The `Image.affine` property contains a 4x4 matric that can convert a (row, column, slice) index to an (x, y, z) position. This will always be diagonal, so (0, 0) contains x voxel size etc, (0, 3) contains x origin.
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

## Loading from a file

An image can be loaded from a dicom, nifti, or numpy file via:
```
im = Image(filepath)
```

If the dicom file is part of a series, any other files in that series in the same directory will also be loaded. Alternatively, you can give the path to a directory containing multiple dicom files. The first dicom file in that directory alphabetically will be loaded along with any other files in its series.

## Loading from an array

An image can also be loaded from a numpy array. By default, it will be taken to have origin (0, 0, 0) and voxel sizes (1, 1, 1)mm; otherwise, these can be set manually, either via:
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

## Plotting
To plot a slice of the image, you need to specify the orientation (`x-y`, `y-z`, or `x-z`; default `x-y`) and either the slice number, array index, or position in mm (by default, the central slice in the chosen orientation will be plotted).
e.g.
```
im.plot('y-z', idx=5)
```

## Writing out image data
Images can be written out with the `Image.write(filename)` function. The output filetype will be inferred from the filename.

### Writing to dicom
If `filename` ends in `.dcm` or is a directory (i.e. has no extension), the image will be written in dicom format. Each `x-y` slice will be written to a separate file labelled by slice number, e.g. slice 1 (corresponding to `[:, :, -1]` in the image array) would be saved to `1.dcm`.

The path to a dicom file from which to take the header can be specified via the `header_source` parameter. If no path is given but the input source for the Image was a dicom file, the header will be taken from the source. Otherwise (e.g. if the file was loaded from a nifti and no `header_source` is given), a brand new header with new UIDs will be created. In that case, you can set the following info for that header:
- `patient_id`
- `modality`
- `root_uid` (an ID unique to your institue while will prefix the generated dicom UIDs so that they are globally unique; one can be obtained here: https://www.medicalconnections.co.uk/FreeUID/)

### Writing to nifti
If `filename` ends in `.nii` or `.nii.gz`, the image will be written to nifti. The nifti will be in canonical format, i.e. in Right, Anterior, Superior configuration. (Note that this means the nifti you write out may not be the same as the one you read in).

### Writing to a numpy array
If `filename` ends in `.npy`, the image array will be written to a numpy binary file. To write a canonical nifti-style array instead of the dicom-style array, set `nifti_array=True`. If `set_geometry` is `True` (which it is by default), a text file will be written to the same directory as the `.npy` file containing the voxel sizes and origin.
