"""Classes for loading and comparing medical images."""

import numbers
import pathlib
from inspect import signature

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pydicom.dataset import FileDataset, FileMetaDataset
import scipy.ndimage
import scipy.interpolate
import copy
import datetime
import glob
import functools
import logging
import math
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import os
import shutil
import pydicom
import tempfile
import time
import uuid
import skimage.transform

import skrt.core
from skrt.dicom_writer import DicomWriter

try:
    import mahotas
    _has_mahotas = True
except ModuleNotFoundError:
    _has_mahotas = False

_axes = ["x", "y", "z"]
_slice_axes = {"x-y": 2, "y-z": 0, "x-z": 1}
_plot_axes = {"x-y": [0, 1], "y-z": [2, 1], "x-z": [2, 0]}
_default_figsize = 6
_default_stations = {"0210167": "LA3", "0210292": "LA4"}
_default_bolus_names = ["planning bolus", "virtual bolus", "bolus",
        "fo p/bolus", "for bolus", "for p-bolus", "for virtual bolus",
        "plan bolus", "planning  bolus", "planningl bolus", "pretend bolus",
        "temp-for bolus", "temp- for bolus", "Temp for p/bolus",
        "treatment bolus", "t1-for bolus", "0.5CM BOLUS"]

# Matplotlib settings
mpl.rcParams["figure.figsize"] = (7.4, 4.8)
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14.0

skrt.core.Defaults().foreground_name = "foreground"
skrt.core.Defaults().foreground_threshold = -150

class Image(skrt.core.Archive):
    """
    Class representing a medical image.

    Attributes of an Image object should usually be accessed via
    their getter methods, rather than directly, to ensure that
    attribute values have been loaded.
    """

    def __init__(
        self,
        path="",
        load=True,
        title=None,
        affine=None,
        voxel_size=(1, 1, 1),
        origin=(0, 0, 0),
        nifti_array=False,
        downsample=None,
        dtype=None,
        auto_timestamp=False,
        default_intensity=(-200, 300),
        log_level=None,
        rgb_weights=(0.299, 0.587, 0.114),
        rgb_rescale_slope=100,
        rgb_rescale_intercept=0
    ):
        """
        Initialise from a medical image source.

        **Parameters:**
        
        path : str/array/Nifti1Image, default = ""
            Source of image data. Can be either:

                (a) A string, optionally with wildcards,
                    containing the path to one or multiple dicom files,
                    the path to a directory containg dicom files,
                    or the path to a single nifti file;
                (b) A list of paths, optionally with wildcards, to dicom files;
                (c) A string, optionally containing wildcards, containing
                    the path to a single numpy file containing a 2D or 3D array;
                (d) A 2D or 3D numpy array;
                (e) A nibabel.nifti1.Nifti1Image object;
                (f) An existing Image object to be cloned; in this case, all 
                    other input args except <title> will be ignored, as these 
                    will be taken from the existing Image.

            Notes:

            1. If path points to a single file, all files in the same
               directory as this file are considered also.

            2. When path resolves to multiple dicom files, only files
            that match the values of the first file for the dicom
            attributes: "StudyInstanceUID", "SeriesNumber", "Modality",
            "ImageOrientationPatient".  When path points to a single file
            in a directory with others, this file is taken as the first
            file.  Otherwise, files are sorted according to natural sort
            order (so "2.dcm" before "11.dcm").  To load images from files
            in a single directory (or in a directory tree) that may have
            different values for these, it could be better to use the
            skrt.patient.Patient class:

            from skrt import Patient
            p = Patient("path/to/directory", unsorted_dicom=True)

            For more details, see documentation of Patient class.

        load : bool, default=True
            If True, the image data will be immediately loaded. Otherwise, it
            can be loaded later with the load() method.

        title : str, default=None
            Title to use when plotting the image. If None and <source> is a
            path, a title will be automatically generated from the filename.

        affine : 4x4 array, default=None
            Array containing the affine matrix to use if <source> is a numpy
            array or path to a numpy file. If not None, this takes precendence
            over <voxel_size> and <origin>.

        voxel_size : tuple, default=(1, 1, 1)
            Voxel sizes in mm in order (x, y, z) to use if <source> is a numpy
            array or path to a numpy file and <affine> is not provided.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm in order (x, y, z) to use if <source> is a
            numpy array or path to a numpy file and <affine> is not provided.

        nifti_array : bool, default=False
            If True and <source> is a numpy array or numpy file, the array
            will be treated as a nifti-style array, i.e. (x, y, z) in
            (row, column, slice), as opposed to dicom style.

        downsample : int/list, default=None
            Amount by which to downsample the image. Can be a single value for
            all axes, or a list containing downsampling amounts in order
            (x, y, z).

        dtype : type, default=None
            Type to which loaded data should be cast.

        auto_timestamp : bool default=False
            If true and no valid timestamp is found within the path string,
            timestamp generated from current date and time.

        default_intensity : tuple,None default=(-200, 300)
            Default intensity range for image display.  This can
            be specified as a two-element tuple, giving minimum and maximum,
            or if set to None then intensity range used is from the
            minimum of zero and the image minimum, to the image maximum.
            If WindowCenter and WindowWidth are defined in a
            DICOM source file, these values will be used instead to
            define the default intensity range.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        """

        # Clone from another Image object
        if issubclass(type(path), Image):
            path.clone_attrs(self)
            if title is not None:
                self.title = title
            return

        # Otherwise, load from source
        self.data = None
        self.title = title
        self.source = (skrt.core.fullpath(path)
                       if isinstance(path, (str, pathlib.Path)) else path)
        if isinstance(self.source, str):
            self.source = sorted(glob.glob(self.source),
                    key=skrt.core.alphanumeric)
            if 1 == len(self.source):
                self.source = self.source[0]
            elif not self.source:
                self.source = ""
        elif isinstance(self.source, (list, set, tuple)):
            paths = list(self.source)
            self.source = []
            for path in paths:
                self.source.extend(glob.glob(skrt.core.fullpath(path)))

        self.source_type = None
        self.dicom_dataset = None
        self.voxel_size = list(voxel_size) if voxel_size is not None else None
        self.origin = list(origin) if origin is not None else None
        self.affine = affine
        self.downsampling = downsample
        self.nifti_array = nifti_array
        self.structure_sets = []
        self.doses = []
        self.plans = []
        self._custom_dtype = dtype
        self.sinogram = None
        # Default image plotting settings
        self._default_colorbar_label = "Radiodensity (HU)"
        self._default_cmap = "gray"
        self.default_intensity = default_intensity

        # Set up event logging.
        self.log_level = (skrt.core.Defaults().log_level if log_level is None
                else log_level)
        self.logger = skrt.core.get_logger(
                name=type(self).__name__, log_level=self.log_level)

        # Set parameters for converting from rgb to grey level.
        self.rgb_weights = rgb_weights
        self.rgb_rescale_slope = rgb_rescale_slope
        self.rgb_rescale_intercept = rgb_rescale_intercept

        path = self.source if isinstance(self.source, str) else ""
        # If self.source is a list of paths, all pointing to the same directory,
        # set image path to the first path in the list.
        if isinstance(self.source, list):
            path = pathlib.Path(self.source[0]).parent
            for file_path in self.source:
                if pathlib.Path(file_path).parent != path:
                    path = ""
                    break
            path = str(path)

        skrt.core.Archive.__init__(self, path, auto_timestamp)

        if load and (not isinstance(self.source, str) or self.source):
            self.load()

    def __add__(self, other):
        '''
        Define image addition.

        The result of the addition of self and other is an Image object
        that has the same affine matrix as self, and has a data array
        obtained by summing the data arrays of self and other.
        '''
        if not self.has_same_geometry(other):
            raise RuntimeError("Objects for addition must have same geometry")
        result = self.__class__(path=(self.get_data() + other.get_data()),
                affine=self.get_affine())
        return result

    def __iadd__(self, other):
        '''
        Define image addition in place.

        The result of the addition of self and other is an Image object
        that has the same affine matrix as self, and has a data array
        obtained by summing the data arrays of self and other.
        '''
        if not self.has_same_geometry(other):
            raise RuntimeError("Objects for addition must have same geometry")
        return self + other

    def __mul__(self, other):
        '''
        Define image multiplication by a scalar.

        The result of the multiplication of self and a scalar (other) is an
        Image object that has the same affine matrix as self, and has
        a data array obtained by multiplying the data array of self by
        the scalar.
        '''
        if not isinstance(other, numbers.Number):
            raise RuntimeError(
                    f"{type(self)} can only be multiplied by a scalar")
        result = self.__class__(path=(other * self.get_data()),
                affine=self.get_affine())
        return result

    def __rmul__(self, other):
        '''
        Define image multiplication by a scalar.

        The result of the multiplication of self and a scalar (other) is an
        Image object that has the same affine matrix as self, and has
        a data array obtained by multiplying the data array of self by
        the scalar.
        '''
        return (self * other)

    def __imul__(self, other):
        '''
        Define in-place multiplication of image by a scalar.

        The result of the multiplication of self and a scalar (other) is an
        Image object that has the same affine matrix as self, and has
        a data array obtained by multiplying the data array of self by
        the scalar.
        '''
        return self * other

    def __truediv__(self, other):
        '''
        Define image division by a scalar.

        The result of the division of self by a scalar (other) is an
        Image object that has the same affine matrix as self, and has
        a data array obtained by dividing the data array of self by
        the scalar.
        '''
        if not isinstance(other, numbers.Number):
            raise RuntimeError(
                    f"{type(self)} can only be divided by a scalar")
        result = self.__class__(path=(self.get_data() / other),
                affine=self.get_affine())
        return result

    def __itruediv__(self, other):
        '''
        Define in-place division of image by a scalar.

        The result of the division of self by a scalar (other) is an
        Image object that has the same affine matrix as self, and has
        a data array obtained by dividing the data array of self by
        the scalar.
        '''
        return self / other

    def __neg__(self):
        '''
        Define unary negative for image.

        The result of the unary negative is an Image object that has the
        same affine matrix as self, and has a data array obtained by
        taking the negative of each element of the data array of self.
        '''
        result = self.__class__(path=(-self.get_data()),
                affine=self.get_affine())
        return result

    def __pos__(self):
        '''
        Define unary positive for image.

        The result of the unitary positive is an Image object that has the
        same affine matrix as self, and has a data array that is the
        same as the data array of self.
        '''
        result = self.__class__(path=(self.get_data()),
                affine=self.get_affine())
        return result

    def __sub__(self, other):
        '''
        Define image subtraction.

        The result of the subtraction of other from self is an Image object
        that has the same affine matrix as self, and has a data array
        obtained by subtracting the data array of other from the data
        array of other.
        '''
        if not self.has_same_geometry(other):
            raise RuntimeError("Objects for addition must have same geometry")
        result = self.__class__(path=(self.get_data() - other.get_data()),
                affine=self.get_affine())
        return result

    def __isub__(self, other):
        '''
        Define image subtraction in place.

        The result of the subtraction of other from self is an Image object
        that has the same affine matrix as self, and has a data array
        obtained by subtracting the data array of other from the data
        array of self.
        '''
        if not self.has_same_geometry(other):
            raise RuntimeError("Objects for addition must have same geometry")
        return self - other

    def astype(self, itype):
        '''
        Return image object with requested type of representation.

        Image objects loaded from a DICOM source and Image objects
        loaded from a NIfTI source have different representations
        for two reasons:

        - indices for an image slice have the order
          [row][column] in pydicom vs [column][row] in nibabel;
        - axis definitions follow radiology convention
          in pydicom vs neurology convention in nibabel; for discussion
          of the conventions, see:
          https://nipy.org/nibabel/neuro_radio_conventions.html

        This function returns the requested representation,
        independently of the original source.

        **Parameter:**

        itype : str
            Identifier of the representation type required.  Allowed
            values are 'dcm' and 'dicom' for a pydicom/DICOM
            representation; 'nii' and 'nifti' for a nibabel/NIfTI
            representation.  For any other value, None is returned.
        '''

        # Check if the requested type is recognised.
        nii_type = itype in ['nii', 'nifti']
        dcm_type = itype in ['dcm', 'dicom']

        if nii_type or dcm_type:
            # Ensure that image is loaded, and create clone.
            self.load()
            im = self.__class__(self)

            # Modify image data if source_type isn't the requested type.
            if ((nii_type and 'nifti' not in self.source_type)
                    or (dcm_type and 'nifti' in self.source_type)):
                affine = self.affine.copy()
                # Convert to nibabel/NIfTI representation.
                if nii_type:
                    affine[0, :] = -affine[0, :]
                    affine[1, 3] = -(affine[1, 3] +
                            (self.get_data().shape[0] - 1)
                            * self.get_voxel_size()[1])
                    data = self.get_data().transpose(1, 0, 2)[:, ::-1, :]
                    im.source_type = 'nifti array'
                # Convert to pydicom/DICOM representation.
                else:
                    affine[0, :] = -affine[0, :]
                    affine[1, 3] = -(affine[1, 3] +
                            (self.get_data().shape[1] - 1)
                            * self.get_voxel_size()[1])
                    data = self.get_data().transpose(1, 0, 2)[::-1, :, :]
                    im.source_type = 'array'
                # Reset parameters and geometry based on
                # updated data and affine.
                im.source = data
                im.data = data
                im.affine = affine
                im.dicom_dataset = None
                im.voxel_size = None
                im.origin = None
                im.set_geometry()

        else:
            # Deal with case where requested type is unrecognised.
            im = None

        return im

    def clone_with_structure_set(self, structure_set=None, roi_names=None,
            image_structure_set_index=-1, structure_set_name=None):
        """
        Clone current image, and associate to clone a filtered structure set.

        **Parameters:**

        structure_set : skrt.structures.StructureSet, default=None
            Structure set to be filtered and associated to image clone.
            Disregarded if a null value.

        roi_names : dict, default=None
            Dictionary for renaming and filtering ROIs, where the
            keys are names for ROIs to be kept, and values are lists of
            alternative names with which these ROIs may have been labelled.
            The alternative names can contain wildcards with the '*' symbol.
            If a value of None is given, all ROIs in the structure set
            are kept, with no renaming.

        image_structure_set_index, int, default=-1
            Integer specifying index in current image's list of structure
            sets of structure set to be associated with clone.  This
            parameter is considered only if structure_set has a null value.

        structure_set_name, str, default=None
            Name to be assigned to structure set.  If None, existing name
            is kept.
        """
        # Clone the current image.
        im = self.clone()

        # Define structure set to be associated with clone.
        if structure_set:
            ss = structure_set
        else:
            try:
                ss = im.structure_sets[image_structure_set_index]
            except (IndexError, TypeError):
                ss = None

        # Filter structure set.  If result is non null, associate
        # structure set to image, and associate image to structure set.
        im.clear_structure_sets()
        if ss:
            ss = ss.filtered_copy(names=roi_names, keep_renamed_only=True,
                    copy_roi_data=False)
            if ss.get_roi_names():
                if isinstance(structure_set_name, str):
                    ss.name = structure_set_name
                ss.set_image(im)
                im.assign_structure_set(ss)

        return im

    def get_data(self, standardise=False, force_standardise=True):
        """Return 3D image array.

        **Parameters:**
        
        standardise : bool, default=False
            If False, the data array will be returned in the orientation in 
            which it was loaded; otherwise, it will be returned in standard
            dicom-style orientation such that [column, row, slice] corresponds
            to the [x, y, z] axes.
        """

        if self.data is None:
            self.load()
        if standardise:
            return self.get_standardised_data(force=force_standardise)
        return self.data

    def get_dicom_filepath(self, sl=None, idx=None, pos=None):
        """Return path to the dicom dataset corresponding to a specific 
        slice.

        **Parameters:**
        
        sl : int, default=None
            Slice number; used if not None.

        idx : int, default=None
            Slice array index; used if not None and <sl> is None.

        pos : float, default=None
            Slice position in mm; used if not None and <sl> and <idx> are both
            None.
        """

        if sl is None and idx is None and pos is None:
            print("Must provide a slice number, array index, or position in "
                  "mm!")
            return 

        idx = self.get_idx("x-y", sl=sl, idx=idx, pos=pos)
        paths = {
            self.pos_to_idx(z, "z"): path for z, path in self._z_paths.items()
        }
        return skrt.core.fullpath(paths[idx])

    def get_dicom_dataset(self, sl=None, idx=None, pos=None):
        """Return pydicom.dataset.FileDataset object associated with this Image
        if it was loaded from dicom; otherwise, return None.

        If any of <sl>, <idx> or <pos> are provided, the dataset corresponding
        to that specific slice will be returned; otherwise, the last loaded
        dataset will be returned.

        **Parameters:**
        
        sl : int, default=None
            Slice number; used if not None.

        idx : int, default=None
            Slice array index; used if not None and <sl> is None.

        pos : float, default=None
            Slice position in mm; used if not None and <sl> and <idx> are both
            None.

        """

        self.load()

        # If no specific slice is requested, just return the last loaded dataset
        if sl is None and idx is None and pos is None:
            return self.dicom_dataset

        if not getattr(self, "_z_paths", None):
            return

        # Otherwise, load the dataset for that slice
        return pydicom.dcmread(
            self.get_dicom_filepath(sl=sl, idx=idx, pos=pos), force=True
        )

    def get_voxel_size(self, standardise=False, force_standardise=True):

        """
        Return voxel sizes in mm in order [x, y, z].

        **Parameters:**
        
        standardise : bool, default=False
            If False, the voxel size will be returned for the image as loaded;
            otherwise, it will be returned for the image in standard
            dicom-style orientation, such that [column, row, slice] corresponds
            to the [x, y, z] axes.

        force_standardise : bool, default=True
            If True, the standardised image will be recomputed from self.data 
            even if it has previously been computed.
        """

        self.load()
        if not standardise:
            return self.voxel_size
        else:
            return self.get_standardised_voxel_size(force=force_standardise)

    def get_size(self):

        """Return image sizes in mm in order [x, y, z]."""

        self.load()
        return self.image_size

    def get_origin(self, standardise=False, force_standardise=True):
        """
        Return origin position in mm in order [x, y, z].

        **Parameters:**
        
        standardise : bool, default=False
            If False, the origin will be returned for the image as loaded;
            otherwise, it will be returned for the image in standard
            dicom-style orientation, such that [column, row, slice] corresponds
            to the [x, y, z] axes.

        force_standardise : bool, default=True
            If True, the standardised image will be recomputed from self.data 
            even if it has previously been computed.
        """

        self.load()
        if not standardise:
            return self.origin
        else:
            return self.get_standardised_origin(force=force_standardise)

    def get_n_voxels(self):
        """Return number of voxels in order [x, y, z]."""

        self.load()
        return self.n_voxels

    def get_volume(self, units="mm"):
        """Get image volume in specified units.

        **Parameter:**
        
        units : str, default="mm"
            Units of volume. Can be any of:
                - "mm": return volume in millimetres cubed.
                - "ml": return volume in millilitres.
                - "voxels": return volume in number of voxels.
        """

        # Obtain volume as number of voxels.
        volume = np.prod(self.get_n_voxels())

        # Convert to volume in cubic millimetres.
        if units != "voxels":
            volume *= np.prod(self.get_voxel_size())

        # Convert to volume in millilitres.
        if units == "ml":
            volume /= 1000

        return volume

    def get_extents(self):
        """
        Get minimum and maximum extent of the image in mm along all three axes,
        returned in order [x, y, z].
        """

        self.load()
        return self.image_extent

    def get_affine(self, standardise=False, force_standardise=True):
        """Return affine matrix.

        **Parameters:**
        
        standardise : bool, default=False
            If False, the affine matrix will be returned in the orientation in 
            which it was loaded; otherwise, it will be returned in standard
            dicom-style orientation such that [column, row, slice] corresponds
            to the [x, y, z] axes.
        """

        self.load()
        if not standardise:
            return self.affine
        else:
            return self.get_standardised_affine(force=force_standardise)

    def get_structure_sets(self):
        """Return list of StructureSet objects associated with this Image."""

        return self.structure_sets

    def get_rois(self, name):
        """
        Get all instances of ROIs with specified name in image structure sets.

        **Parameter:**
        name : str
            Name for which ROI instances are to be returned.
        """
        rois = []
        for ss in self.get_structure_sets():
            if name in ss.get_roi_names():
                rois.append(ss.get_roi(name))
        return rois

    def get_doses(self):
        """Return list of Dose objects associated with this Image."""

        return self.doses

    def get_plans(self):
        """Return list of Plan objects associated with this Image."""
        if hasattr(self, "plans"):
            plans = self.plans
        elif hasattr(self, "plan"):
            plans = [self.plan]
        else:
            plans = []

        return plans

    def get_alignment_translation(self, other, alignment=None):
        """
        Determine translation for aligning <self> to <other>.

        This method calls the function of the same name,
        with <self> and <other> as <im1> and <im2> respectively.

        For explanation of parameters, see documentation of
        skrt.image.get_alignment_translation().
        """
        return get_alignment_translation(self, other, alignment)

    def get_translation_to_align(self, other, alignments=None,
            default_alignment=2, threshold=None):
        """
        Determine translation for aligning <self> to <other>.

        This method calls the function of the same name,
        with <self> and <other> as <im1> and <im2> respectively.

        For explanation of parameters, see documentation of
        skrt.image.get_translation_to_align().
        """
        return get_translation_to_align(self, other, alignments,
                default_alignment, threshold)

    def get_translation_to_align_image_rois(self, other, roi_name1, roi_name2,
            z_fraction1=None, z_fraction2=None):
        """
        Determine translation for aligning ROI of <self> to ROI of <other>.

        This method calls the function of the same name,
        with <self> and <other> as <im1> and <im2> respectively.

        For explanation of parameters, see documentation of
        skrt.image.get_translation_to_align_image_rois().
    """
        return self.get_rois(roi_name1)[0].get_translation_to_align(
                other.get_rois(roi_name2)[0], z_fraction1, z_fraction2)

    def load(self, force=False):
        """Load pixel array from image source. If already loaded and <force> 
        is False, nothing will happen.

        **Parameters:**
        
        force : bool, default=True
            If True, the pixel array will be reloaded from source even if it 
            has previously been loaded.

        Data loading takes input from self.source and uses this to assign
        self.data (as well as geometric properties, where relevant). The 
        parameter self.source_type is set to a string indicating the type 
        of source, which can be any of:

            "array": 
                Data loaded from a numpy array in dicom-style orientation.

            "nifti array": 
                Data loaded from a numpy array in nifti-style orientation.

            "nifti": 
                Data loaded from a nifti file.

            "dicom": 
                Data loaded from one or more dicom file(s).

        The loading sequence is as follows: 

            1. If self.source is a numpy array, self.data will be set to the
            contents of self.source. If <nifti_array> was set to True when 
            __init__() was called, self.source_type is set to "nifti array";
            otherwise, self.source_type is set to "array".

            2. If self.source is a string, this string is passed to
            glob.glob().  The result is assigned to self.source, and 
            is treated as a list of filepaths.  If the list contains a
            single element, attempt to load a nifti file from this path
            using the function load_nifti(). If the path points to a
            valid nifti file, this will return a pixel array and affine
            matrix, which are assigned to self.data and self.affine,
            respectively. Set self.source_type to "nifti".
           
            3. If no data were loaded in step 2 (i.e. self.data is still None),
            and self.source contains a single path, attempt to load from
            a numpy binary file at this path using the function load_npy(). If
            the path points to a valid numpy binary file, this will return
            a pixel array, which is assigned to self.data. Set source_type
            to either "nifti array" or "array", depending on whether
            <nifti_array> was set to True or False, respectively,
            when __init__() was called.

            4. If no data were loaded in step 4 (i.e. self.data is still None),
            attempt to load from dicom file(s) or directory at the path(s) in
            self.source using the function load_dicom().  If successful, this 
            returns a pixel array, affine matrix, default greyscale window
            centre and width, the last loaded pydicom.dataset.FileDataset 
            object, and a dictionary mapping z positions to paths to the
            dicom file for that slice. These outputs are used to assign 
            self.data, self.affine, self.dicom_dataset, self._z_paths,
            and self._z_instance_numbers; self.source_type is set to "dicom".

            5. If no data were loaded in step 5 (i.e. self.data is still None),
            raise a RuntimeError.

            6. If self.data contains a 2D array, convert this to 3D by adding
            an extra axis.

            7. Apply any downsampling as specificied in __init__().

            8. Run self.set_geometry() in order to compute geometric quantities
            for this Image.

            9. If a default window width and window centre were loaded from 
            dicom, use these to set self.default_window to a greyscale window 
            range.

            10. If self.title is None and self.source is a filepath, infer
            a title from the basename of this path.

        """

        if self.data is not None and not force:
            return

        window_width = None
        window_centre = None

        # Load image array from source
        # Numpy array
        if isinstance(self.source, np.ndarray):
            self.data = self.source
            self.source_type = "nifti array" if self.nifti_array else "array"

        # Try loading from nifti file
        elif isinstance(self.source, str):
            if os.path.exists(self.source):
                if os.path.isfile(self.source):
                    self.data, affine = load_nifti(self.source)
                    self.source_type = "nifti"
                if self.data is not None:
                    self.affine = affine
            elif not hasattr(self, "dicom_paths"):
                raise RuntimeError(
                    f"Image input {self.source} does not exist!")

            # Try loading from numpy file
            if self.data is None:
                self.data = load_npy(self.source)
                self.source_type = ("nifti array" if self.nifti_array
                        else "array")

        # Try loading from dicom file
        if self.data is None and isinstance(self.source, (str, list)):
            if hasattr(self, "dicom_paths") and not self.source:
                paths = self.dicom_paths
            else:
                paths = self.source
            try:
                self.data, affine, window_centre, window_width, ds,\
                        self._z_paths, self._z_instance_numbers = \
                        load_dicom(paths)
                dicom_loaded = True
            except ValueError:
                dicom_loaded = False

            if dicom_loaded:
                self.source_type = "dicom"
                if self.data is not None:
                    self.dicom_dataset = ds
                    self.affine = affine

        # Try reading as a format known to Python Imaging Library.
        if self.data is None and isinstance(self.source, str):
            self.data = load_rgb(
                    self.source, self.rgb_weights,
                    self.rgb_rescale_slope, self.rgb_rescale_intercept)
            window_centre = self.rgb_rescale_slope / 2
            window_width = self.rgb_rescale_slope

        # If still None, raise exception
        if self.data is None:
            raise RuntimeError(f"{self.source} not a valid image source!")

        # Cast to custom type
        if (self._custom_dtype is not None
            and self.data.dtype != self._custom_dtype):
            self.data = self.data.astype(self._custom_dtype)

        # Ensure array is 3D
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]

        # Apply downsampling
        if self.downsampling:
            self.downsample(self.downsampling)
        else:
            self.set_geometry()

        # Set default grayscale range
        if window_width and window_centre:
            self._default_vmin = window_centre - window_width / 2
            self._default_vmax = window_centre + window_width / 2
        else:
            if self.default_intensity is None:
                self._default_vmin = min(self.data.min(), 0)
                self._default_vmax = self.data.max()
            else:
                self._default_vmin = self.default_intensity[0]
                self._default_vmax = self.default_intensity[1]

        # Set title from filename
        if self.title is None:
            if isinstance(self.source, str) and os.path.exists(self.source):
                self.title = os.path.basename(self.source)

    def get_standardised_data(self, force=True):
        """Return array in standard dicom orientation, where 
        [column, row, slice] corresponds to the [x, y, z] axes.
        standardised image array. 

        **Parameters:**
        
        force : bool, default=True
            If True, the standardised array will be recomputed from self.data 
            even if it has previously been computed.
        """

        if not hasattr(self, "_sdata") or force:
            self.standardise_data()
        return self._sdata

    def get_standardised_affine(self, force=True):
        """Return affine matrix in standard dicom orientation, where 
        [column, row, slice] corresponds to the [x, y, z] axes.

        **Parameters:**
        
        force : bool, default=True
            If True, the standardised array will be recomputed from self.data 
            even if it has previously been computed.
        """

        if not hasattr(self, "_saffine") or force:
            self.standardise_data()
        return self._saffine

    def get_standardised_origin(self, force=True):
        """Return origin for image in standard dicom orientation, where 
        [column, row, slice] corresponds to the [x, y, z] axes.

        **Parameters:**
        
        force : bool, default=True
            If True, the standardised array will be recomputed from self.data 
            even if it has previously been computed.
        """

        if not hasattr(self, "_sorigin") or force:
            self.standardise_data()
        return self._sorigin

    def get_standardised_voxel_size(self, force=True):
        """Return voxel size for image in standard dicom orientation, where 
        [column, row, slice] corresponds to the [x, y, z] axes.

        **Parameters:**
        
        force : bool, default=True
            If True, the standardised array will be recomputed from self.data 
            even if it has previously been computed.
        """

        if not hasattr(self, "_svoxel_size") or force:
            self.standardise_data()
        return self._svoxel_size

    def standardise_data(self):
        """Manipulate data array and affine matrix into standard dicom
        orientation, where [column, row, slice] corresponds to the [x, y, z] 
        axes; assign results to self._sdata and self._saffine, respectively.
        Standardised voxel sizes (self._svoxel_size) and origin position
        (self._sorigin) will also be inferred from self._affine.
        """

        # Adjust dicom
        if self.source_type == "dicom":

            data = self.get_data()
            affine = self.get_affine()

            # Transform array to be in order (row, col, slice) = (x, y, z)
            orient = np.array(self.get_orientation_vector()).reshape(2, 3)
            axes_colrow = self.get_axes(col_first=True)
            axes = self.get_axes(col_first=False)
            transpose = [axes.index(i) for i in (1, 0, 2)]
            data = np.transpose(self.data, transpose).copy()

            # Adjust affine matrix
            affine = self.affine.copy()
            for i in range(3):

                # Voxel sizes
                if i != axes_colrow.index(i):
                    voxel_size = affine[i, axes_colrow.index(i)].copy()
                    affine[i, i] = voxel_size
                    affine[i, axes_colrow.index(i)] = 0

                # Invert axis direction if negative
                if axes_colrow.index(i) < 2 and orient[axes_colrow.index(i), i] < 0:
                    affine[i, i] *= -1
                    to_flip = [1, 0, 2][i]
                    data = np.flip(data, axis=to_flip)
                    n_voxels = data.shape[to_flip]
                    affine[i, 3] = affine[i, 3] - (n_voxels - 1) * affine[i, i]

        # Adjust nifti
        elif "nifti" in self.source_type:

            # Load and cache canonical data array
            if not hasattr(self, "_data_canonical"):
                init_dtype = self.get_data().dtype
                nii = nibabel.as_closest_canonical(
                    nibabel.Nifti1Image(self.data.astype(np.float64), self.affine)
                )
                setattr(self, "_data_canonical", nii.get_fdata().astype(init_dtype))
                setattr(self, "_affine_canonical", nii.affine)

            data = self._data_canonical.copy()
            transpose = pad_transpose([1, 0, 2], data.ndim)
            data = data.transpose(*transpose)
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            affine = self._affine_canonical.copy()

            # Reverse x and y directions
            affine[0, 3] = -(affine[0, 3] + (data.shape[1] - 1) * affine[0, 0])
            affine[1, 3] = -(affine[1, 3] + (data.shape[0] - 1) * affine[1, 1])

        else:
            data = self.get_data()
            affine = self.get_affine()

        # Assign standardised image array and affine matrix
        self._sdata = data
        self._saffine = affine

        # Get standard voxel sizes and origin
        self._svoxel_size = list(np.diag(self._saffine))[:-1]
        self._sorigin = list(self._saffine[:-1, -1])

    def print_geometry(self, float_format='.4f'):
        self.load()
        print(f'Shape: {self.get_data().shape}')
        voxel_size = tuple([f'{dxyz:{float_format}}'
                for dxyz in self.get_voxel_size()])
        print(f'Voxel size: {voxel_size}')
        origin = tuple([f'{xyz:{float_format}}' for xyz in self.get_origin()])
        print(f'Origin: {origin}')
        image_extent = tuple([f'({v1:{float_format}}, {v2:{float_format}})'
            for v1, v2 in list(self.image_extent)])
        print(f'Extent: {image_extent}')
        image_size = tuple([f'{abs(v2 - v1):{float_format}}'
            for v1, v2 in list(self.image_extent)])
        print(f'Image size: {image_size}')

    def resample(self, voxel_size=(1, 1, 1), order=1):
        '''
        Resample image to have particular voxel sizes.

        **Parameters:**
        
        voxel_size: int/float/tuple/list, default =(1, 1, 1)
            Voxel size to which image is to be resampled.  If voxel_size is
            a tuple or list, then it's taken to specify voxel sizes
            in mm in the order x, y, z.  If voxel_size is an int or float,
            then it's taken to specify the voxel size in mm along all axes.

        order: int, default = 1
            Order of the b-spline used in interpolating voxel intensity values.
        '''
        if not voxel_size:
            return

        self.load()
        if not (isinstance(voxel_size, list) or isinstance(voxel_size, tuple)):
            voxel_size = 3 * [voxel_size]

        # Fill in None values with own voxel size
        voxel_size2 = []
        for i, v in enumerate(voxel_size):
            if v is not None:
                voxel_size2.append(v)
            else:
                voxel_size2.append(self.voxel_size[i])
        voxel_size = voxel_size2

        # Exit if no change to voxel size.
        if list(voxel_size) == list(self.voxel_size):
            return

        # Define scale factors to obtain requested voxel size
        scale = [self.voxel_size[i] / voxel_size[i] for i in range(3)]

        self.data = scipy.ndimage.zoom(self.data, scale, order=order,
                mode='nearest')

        # Reset properties
        self.origin = [
            self.origin[i] - self.voxel_size[i] / 2 + voxel_size[i] / 2
            for i in range(3)
        ]

        ny, nx, nz = self.data.shape
        self.n_voxels = [ny, nx, nz]
        self.voxel_size = voxel_size
        self.affine = None
        self.set_geometry()

    def get_coordinate_arrays(self, image_size, origin, voxel_size):
        '''
        Obtain (x, y, z) arrays of coordinates of voxel centres.

        Arrays are useful for image resizing.

        **Parameters:**
        
        image_size : tuple
            Image size in voxels, in order (x,y,z).

        origin : tuple
            Origin position in mm in order (x, y, z).

        voxel_size : tuple
            Voxel sizes in mm in order (x, y, z).
        '''

        self.load()

        # Extract parameters for determining coordinates of voxel centres.
        nx, ny, nz = image_size
        x, y, z = origin
        dx, dy, dz = voxel_size
  
        # Obtain coordinate arrays.
        try:
            x_array = np.linspace(x, x + (nx - 1) * dx, nx)
        except TypeError:
            x_array = None
        try:
            y_array = np.linspace(y, y + (ny - 1) * dy, ny)
        except TypeError:
            y_array = None
        try:
            z_array = np.linspace(z, z + (nz - 1) * dz, nz)
        except TypeError:
            z_array = None

        return (x_array, y_array, z_array)

    def get_foreground_box_mask(self, dx=0, dy=0, threshold=None):
        '''
        Slice by slice, create rectangular mask enclosing foreground mask.

        dx : int, default=0
            Margin along columns to be added on each side of mask bounding box.

        dy : int, default=0
            Margin along rows to be added on each side of mask bounding box.

        threshold : int/float, default=None
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
        '''

        foreground_mask = self.get_foreground_mask(threshold)
        foreground_box_mask = get_box_mask_from_mask(foreground_mask, dx, dy)

        return foreground_box_mask

    def get_foreground_bbox(self, threshold=None, convex_hull=False,
            fill_holes=True, dxy=0):
        """
        Obtain bounding box of image foreground.

        The bounding box is returned as
        [(xmin, xmax), (ymin, ymax), (zmin, zmax)], with values in mm.

        Method parameters are passed to skrt.image.Image.get_foreground_mask()
        to obtain a mask defining the image foreground.  For parameter
        explanations, see skrt.image.Image.get_foreground_mask() documentation.
        """
        return (get_mask_bbox(
            self.get_foreground_mask(threshold, convex_hull, fill_holes, dxy)))

    def get_foreground_bbox_centre_and_widths(self,
            threshold=None, convex_hull=False, fill_holes=True, dxy=0):
        """
        Get centre and widths in mm along all three axes of a
        bounding box enclosing the image foreground.  Centre
        and widths are returned as a tuple ([x, y, z], [dx, dy, dz]).

        Method parameters are passed to skrt.image.Image.get_foreground_mask()
        to obtain a mask defining the image foreground.  For parameter
        explanations, see skrt.image.Image.get_foreground_mask() documentation.
        """
        extents = self.get_foreground_bbox()
        centre = [0.5 * (extent[0] + extent[1]) for extent in extents]
        widths = [(extent[1] - extent[0]) for extent in extents]
        return (centre, widths)

    def get_foreground_comparison(
            self, other, name=None, threshold=None, convex_hull=False,
            fill_holes=True, dxy=0, voxel_size=None, **kwargs):
        """
        Return a pandas DataFrame comparing the foregrounds of
        this image and another.

        ROIs obtaining the image foregrounds are obtained, then
        these are compared using skrt.structures.ROI.get_comparison().

        **Parameters:**

        other: skrt.image.Image
            Image with which this image is to be compared.

        name: str, default=None
            Name to be assigned to the ROI representing the foreground
            of this image, and by default used as row index in DataFrame.
            If null, the name used is the image title, or if this is null
            then f"{skrt.core.Defaults().foreground_name}_1" is used.

        threshold : int/float, default=None
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
    
        convex_hull : bool, default=False
            If True, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=False
            If True, fill holes in the slice foreground masks initially
            obtained.

        dxy : int, default=0
            Margin, in pixel units, to be added to each slice foreground mask.

        voxel_size : tuple, default=None
            Voxel size (dx, dy, dz) in mm to be used for foreground
            masks in comparisons.  If None, the mask voxel size of
            <other> is used if not None; otherwise the default voxel
            size for dummy images, namely (1, 1, 1), is used.  If an
            element of the tuple specifying voxel size is None, the
            value for the corresponding element of the mask voxel size
            of <other> is used.

        kwargs: dict
            Keyword arguments, in addition to voxel_size, passed to
            skrt.structures.ROI.get_comparison().
        """
        name = (name or self.title
                or f"{skrt.core.Defaults().foreground_name}_1")
        roi1 = self.get_foreground_roi(
                threshold=threshold, convex_hull=convex_hull,
                fill_holes=fill_holes, dxy=dxy, name=name)
        roi2 = other.get_foreground_roi(
                threshold=threshold, convex_hull=convex_hull,
                fill_holes=fill_holes, dxy=dxy)

        return roi1.get_comparison(roi2, voxel_size=None, **kwargs)

    def get_foreground_roi(self, threshold=None, convex_hull=False,
            fill_holes=True, dxy=0, **kwargs):
        '''
        Create ROI represening image foreground.

        Slice by slice, the foreground is taken to correspond to the
        largest region of contiguous pixels above a threshold value.
        A binary mask representing the foreground is created, and
        is used as source for creating an ROI.

        **Parameters:**

        threshold : int/float, default=None
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
    
        convex_hull : bool, default=False
            If True, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=False
            If True, fill holes in the slice foreground masks initially
            obtained.

        dxy : int, default=0
            Margin, in pixel units, to be added to each slice foreground mask.

        **kwargs
            Keyword arguments passed to ROI constructor.
        '''
        from skrt.structures import ROI
        if not "name" in kwargs:
            kwargs["name"] = skrt.core.Defaults().foreground_name
        return ROI(self.get_foreground_mask(
            threshold, convex_hull, fill_holes, dxy), **kwargs)

    def get_foreground_mask(self, threshold=None, convex_hull=False,
            fill_holes=True, dxy=0):
        '''
        Create foreground mask.

        Slice by slice, the foreground is taken to correspond to the
        largest region of contiguous pixels above a threshold value.

        **Parameters:**

        threshold : int/float, default=None
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
    
        convex_hull : bool, default=False
            If True, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=False
            If True, fill holes in the slice foreground masks initially
            obtained.

        dxy : int, default=0
            Margin, in pixel units, to be added to each slice foreground mask.
        '''
        self.load()

        # Initialise mask array from foreground mask of first slice.
        out_array = np.array(self.get_slice_foreground(
            0, threshold, convex_hull, fill_holes, dxy))

        # Stack slice foregrounds masks.
        for idx in range(1, self.get_n_voxels()[2]):
            out_array = np.dstack((out_array, self.get_slice_foreground(
                idx, threshold, convex_hull, fill_holes, dxy)))

        # Clone the current image, and replace its data with
        # the foreground mask.
        out_image = Image(self)
        out_image.data = out_array

        return out_image

    def get_slice_foreground(self, idx=0, threshold=None,
            convex_hull=False, fill_holes=False, dxy=0):
        '''
        Create foreground mask for image slice.

        The foreground is taken to correspond to the largest region
        of contiguous pixels above a threshold value.

        **Parameters:**

        idx : int, default=0
            Index of slice for which foreground mask is to be obtained.

        threshold : int/float, default=None
            Intensity value above which pixels are assigned to regions
            for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
    
        convex_hull : bool, default=False
            If True, return the convex hull of the foreground mask
            initially obtained.

        fill_holes : bool, default=False
            If False, fill holes in the foreground mask initially
            obtained.
        '''

        if not _has_mahotas:
            print('WARNING: Module mahotas unavailable')
            print('WARNING: Unable to execute function '\
                    + 'skrt.image.Image.get_slice_foreground()')

        # Extract slice data.
        image_slice = self.get_data()[:, :, idx]

        # Obtain intensity values relative to the minimum.
        test_slice = image_slice - image_slice.min()
        # Make this a 2D array.
        test_slice = np.squeeze(test_slice)

        # Fallback to default threshold.
        if threshold is None:
            threshold = skrt.core.Defaults().foreground_threshold

        # Calculate Otsu threshold, or rescale threshold value provided.
        if threshold is None:
            rescaled_threshold = mahotas.thresholding.otsu(
                    test_slice.astype(np.uint32))
        else:
            rescaled_threshold = threshold - image_slice.min()

        # Handle case where slice contains intensity values above threshold,
        # and intensities are not the same.
        if ((test_slice.max() > rescaled_threshold) and
                (test_slice.max() - test_slice.min() > 0)):

            # Label regions of contiguous pixels of above-threshold intensity.
            label_array1, n_object = mahotas.label(
                    test_slice > rescaled_threshold)
            # Identify largest labelled region, and use this as foreground.
            foreground = np.argsort(np.bincount(label_array1.ravel()))[-2]
            label_array2 = np.int8(label_array1 == foreground)

            if fill_holes:
                # Fill holes using different structuring elements.
                label_array2 = scipy.ndimage.binary_fill_holes(label_array2)
                for i in range(5, 3, -1):
                    structure = np.ones((i, i))
                    label_array2 = scipy.ndimage.binary_fill_holes(
                        label_array2, structure)
                label_array2 = scipy.ndimage.binary_fill_holes(label_array2)

            if dxy > 0:
                # Add a margin to the mask.
                structure = np.ones((dxy, dxy))
                label_array2 = scipy.ndimage.binary_dilation(
                        label_array2, structure)

            if convex_hull:
                # Take the convex hull of the mask.
                label_array2 = skimage.morphology.convex_hull_image(
                        label_array2)

        # Handle the cases where all intensities are the same.
        else:
            if test_slice.max() > rescaled_threshold:
                label_array2 = np.ones(image_slice.shape, dtype=bool)
            else:
                label_array2 = np.zeros(image_slice.shape, dtype=bool)

        return label_array2

    def select_foreground(self, threshold=None, convex_hull=False,
            fill_holes=True, dxy=0, background=None):
        '''
        Modify image to show intensity information only for foreground.

        Slice by slice, the foreground is taken to correspond to the
        largest region of contiguous pixels above a threshold value.
        Voxels outside the foreground region are all assigned the
        same same (background) intensity value.

        **Parameters:**

        threshold : int/float, default=None
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.  If None, use value
            of Defaults().foreground_threshold.  If still None, use
            Otsu threshold.
    
        convex_hull : bool, default=False
            If True, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=False
            If False, fill holes in the slice foreground masks initially
            obtained.

        background : int/float, default=None
            Intensity value to be assigned to background voxels.  If
            None, the image's minimum value is used.
        '''
        self.load()

        # Obtain foreground mask.
        mask = self.get_foreground_mask(threshold=threshold,
                convex_hull=convex_hull, fill_holes=fill_holes, dxy=dxy)

        # Set background intensity if not specified.
        if background is None:
            background = self.get_data().min()

        # Set voxels outside foreground to background intensity.
        self.data[mask.get_data() == False] = background

    def get_intensity_mask(self, vmin=None, vmax=None, convex_hull=False,
                           fill_holes=True, dxy=0):
        '''
        Create intensity mask.

        Slice by slice, the mask corresponds to the largest region of
        contiguous pixels with intensity values inside a given interval.

        **Parameters:**

        vmin : int/float, default=None
            Minimum intensity value for pixel to be included in mask.
            If None, no constraint on minimum intensity is applied.

        vmax : int/float, default=None
            Maximum intensity value for pixel to be included in mask.
            If None, no constraint on maximum intensity is applied.

        convex_hull : bool, default=False
            If True, create mask from the convex hulls of the
            slice foreground masks initially obtained.
    
        fill_holes : bool, default=False
            If True, fill holes in the slice foreground masks initially
            obtained.

        dxy : int, default=0
            Margin, in pixel units, to be added to each slice foreground mask.
        '''
        self.load()

        if vmin is None and vmax is None:
            return Image(np.ones(self.data.shape, dtype=bool),
                         affine = self.get_affine())

        masks = [self.get_foreground_mask(
            threshold=val, convex_hull=convex_hull,
            fill_holes=fill_holes, dxy=dxy) if val is not None
                 else None for val in [vmin, vmax]]

        if masks[0] is None:
            masks[1].data = 1 - masks[1].data
            return masks[1]

        if masks[1] is not None:
            masks[0].data[masks[1].data > 0] = 0

        return masks[0]

    def rescale(self, v_min=0.0, v_max=1.0, constant=0.5):
        '''
        Linearly rescale image greyscale values,
        so that they span a specified range.

        **Parameters:**

        v_min: float, default=0.0
            Minimum greyscale value after rescaling.

        v_max: float, default=1.0
            Maximum greyscale value after rescaling.

        constant: float, default=0.5
            Greyscale value to assign after rescaling if all values
            in the original image are the same.  If None,
            original value is kept.
        '''
        # Perform rescaling.
        u_min = self.get_min(force=True)
        u_max = self.get_max(force=True)
        du = u_max - u_min
        dv = v_max - v_min
        if du:
            self.data = v_min + ((self.data.astype(np.float32) - u_min)
                                 * (dv / du))
        elif constant is not None:
            self.data.fill(constant)

    def resize(self, image_size=None, origin=None, voxel_size=None,
            fill_value=None, image_size_unit=None, centre=None,
            keep_centre=False, method='linear'):
        '''
        Resize image to specified image size, voxel size and origin.

        **Parameters:**
        
        image_size : tuple/list/None, default=None
            Image sizes in order (x,y,z) to which image is to be resized.
            If None, the image's existing size in mm is kept.  If a value
            in the tuple/list is None, the relevant existing value is
            kept.  The unit of measurement ('voxel' or 'mm') is specified
            via image_size_unit.  If the size is in mm, and isn't an
            integer multiple of voxel_size, resizing won't be exact.

        origin : tuple/list/None, default=None
            Origin position in mm in order (x, y, z).  If None, the image's
            existing origin is kept.  If a value in the tuple/list is None,
            the relevant existing value is kept.  Disregarded if centre
            isn't None, or if keep_centre is True.

        voxel_size : tuple/list/None, default=None
            Voxel sizes in mm in order (x, y, z).  If None, the image's
            existing voxel size is kept.  If a value in the tuple/list is None,
            the relevant existing value is kept.

        fill_value: float/None, default = None
            Intensity value to be assigned to any voxels in the resized
            image that are outside the original image.  If set to None,
            the minimum intensity value of the original image is used.

        image_size_unit: str, default=None
            Unit of measurement ('voxel' or 'mm') for image_size.  If None,
            use 'voxel'.

        centre : tuple/list/None, default=None
            Position (x, y, z) in mm in the original image to be set as
            centre for the resized image.  Disregarded if None.  Otherwise
            takes priority over origin and keep_centre.  If a value in
            the tuple/list is None, the relevant value from the original
            image centre is kept.

        keep_centre: bool, default=False
            If True, make the centre of the initial image the centre of
            the resized image, disregarding the value passed to origin.
            Disregarded if the value of the centre parameter isn't None.

        method: str, default='linear'
            Interpolation method to use.  Valid values are 'linear' and
            'nearest'
        '''
        # Return if no resizing requested.
        if image_size is None and voxel_size is None and origin is None:
            return

        # Ensure that data are loaded.
        self.load()

        # Ensure that resizing values are defined.
        allowed_unit = ['mm', 'voxel']
        if image_size is None:
            image_size_unit = "mm"
            image_size = self.get_size()
        elif image_size_unit is None or image_size_unit not in allowed_unit:
            image_size_unit = 'voxel'

        if origin is None:
            origin = self.get_origin()

        if voxel_size is None:
            voxel_size = self.get_voxel_size()

        if centre is None:
            centre = self.get_centre()
        else:
            keep_centre = True

        # Allow for two-dimensional images
        if 2 == len(self.get_data().shape):
            ny, nx = self.get_data().shape
            self.data = self.get_data().reshape(ny, nx, 1)

        # Ensure that values are in lists, rather than tuples,
        # to simplify value replacement.
        image_size = list(image_size)
        origin = list(origin)
        voxel_size = list(voxel_size)
        centre = list(centre)

        # Replace any None values among resizing parameters
        for i in range(3):
            if image_size[i] is None:
                image_size[i] = self.get_n_voxels()[i]
            if origin[i] is None:
                origin[i] = self.get_origin()[i]
            if voxel_size[i] is None:
                voxel_size[i] = self.get_voxel_size()[i]
            if centre[i] is None:
                centre[i] = self.get_centre()[i]

        # Convert to voxel units
        if 'mm' == image_size_unit:
            for i in range(3):
                image_size[i] = max(
                        1, math.floor(image_size[i] / voxel_size[i]))

        # Redefine origin to fix centre position.
        if keep_centre:
            origin = [centre[i] - (0.5 * image_size[i] - 0.5)
                    * voxel_size[i] for i in range(3)]

        # Check whether image is already the requested size
        nx, ny, nz = image_size
        match = (self.get_data().shape == [ny, nx, nz]
            and (self.get_origin() == origin) \
            and (self.get_voxel_size() == voxel_size))

        if not match:

            # If slice thickness not known, set to the requested value
            if self.get_voxel_size()[2] is None:
                self.voxel_size[2] = image.voxel_size[2]

            # Set fill value
            if fill_value is None:
                fill_value = self.get_data().min()

            #print(f"interpolation start time: {time.strftime('%c')}")
            x1_array, y1_array, z1_array = self.get_coordinate_arrays(
                    self.get_n_voxels(), self.get_origin(), self.get_voxel_size())
            if not (x1_array is None or y1_array is None or z1_array is None):
                # Define how intensity values are to be interpolated
                # for the original image
                interpolant = scipy.interpolate.RegularGridInterpolator(
                        (y1_array, x1_array, z1_array),
                        self.get_data(),
                        method=method,
                        bounds_error=False,
                        fill_value=fill_value)

                # Define grid of voxel centres for the resized image
                x2_array, y2_array, z2_array = self.get_coordinate_arrays(
                        image_size, origin, voxel_size)
                nx, ny, nz = image_size
                meshgrid = np.meshgrid(
                        y2_array, x2_array, z2_array, indexing="ij")
                vstack = np.vstack(meshgrid)
                point_array = vstack.reshape(3, -1).T.reshape(ny, nx, nz, 3)

                # Perform resizing, ensuring that original data type is kept.
                dtype = self.data.dtype
                self.data = interpolant(point_array)
                if dtype != self.data.dtype:
                    self.data = self.data.astype(dtype)

                # Reset geometry
                self.voxel_size = voxel_size
                self.origin= origin
                self.n_voxels = image_size
                self.affine = None
                self.set_geometry()

            #print(f"interpolation end time: {time.strftime('%c')}")

        return None

    def match_size(self, image=None, fill_value=None, method='linear'):

        '''
        Match image size to that of a reference image.

        After matching, the image voxels are in one-to-one correspondence
        with those of the reference.

        **Parameters:**
        
        image: skrt.image.Image/None, default=None
            Reference image, with which size is to be matched.

        fill_value: float/None, default = None
            Intensity value to be assigned to any voxels in the resized
            image that are outside the original image.  If set to None,
            the minimum intensity value of the original image is used.

        method: str, default='linear'
            Interpolation method to use.  Valid values are 'linear' and
            'nearest'
        '''

        if not self.has_same_geometry(image):
            image.load()
            self.resize(image.get_n_voxels(), image.get_origin(),
                    image.get_voxel_size(), fill_value, method=method)

    def match_voxel_size(self, image, method="self"):
        """Resample to match z-axis voxel size with that of another Image
        object.

        Note: This method matches voxel size along z-axis only.
              To match voxel sizes in all dimensions, use the function
              skrt.images.match_image_voxel_sizes().

        **Parameters:**
        
        image : Image
            Other image to which z-axis voxel size should be matched.

        method : str, default="self"
            String specifying the matching method. Can be any of:
                - "self": 
                    Match own z voxel size to that of <image>.
                - "coarse": 
                    Resample the image with the smaller z voxels to match
                    that of the image with larger z voxels.
                - "fine": 
                    Resample the image with the larger z voxels to match
                    that of the image with smaller z voxels.
        """

        # Determine which image should be resampled
        if method == "self":
            to_resample = self
            match_to = image
        else:
            own_vz = self.get_voxel_size()[2]
            other_vz = image.get_voxel_size()[2]
            if own_vz == other_vz:
                print("Voxel sizes already match! No resampling applied.")
                return
            if method == "coarse":
                to_resample = self if own_vz < other_vz else image
                match_to = self if own_vz > other_vz else image
            elif method == "fine":
                to_resample = self if own_vz > other_vz else image
                match_to = self if own_vz < other_vz else image
            else:
                raise RuntimeError(f"Unrecognised resampling option: {method}")

        # Perform resampling
        voxel_size = [None, None, match_to.get_voxel_size()[2]]
        init_vz = to_resample.get_voxel_size()[2]
        init_nz = int(to_resample.n_voxels[2])
        to_resample.resample(voxel_size)
        print(
            f"Resampled z axis from {init_nz} x {init_vz:.3f} mm -> "
            f"{int(to_resample.n_voxels[2])} x {to_resample.voxel_size[2]:.3f}"
            "mm"
        )

    def get_min(self, force=False):
        """Get minimum greyscale value of data array."""

        if not force and hasattr(self, "_min"):
            return self._min
        self.load()
        self._min = self.data.min()
        return self._min

    def get_max(self, force=False):
        """Get maximum greyscale value of data array."""

        if not force and hasattr(self, "_max"):
            return self._max
        self.load()
        self._max = self.data.max()
        return self._max

    def get_centroid_idx(self, view="x-y", fraction=1):
        """
        Get array index of slice containing centroid of above-threshold voxels.

        The centroid coordinate along a given axis is calculated as the
        unweighted mean of the coordinates of voxels with an intensity at
        least a given fraction of the maximum.

        **Parameters:**

        view : str, default="x-y"
            Orientation; can be "x-y", "y-z", or "x-z".

        fraction : float, default=1
            Minimum fraction of the maximum intensity that a voxel must record
            to be considered in the centroid calculation.
        """
        self.load()

        # Obtain indices of voxels with above-threshold intensity.
        indices = np.where(self.get_standardised_data()
                >= (self.get_standardised_data().min()
                    + (self.get_standardised_data().max()
                        - self.get_standardised_data().min()) * fraction))

        # Calculate means of indicies of voxels with above-threshold intensity.
        iy, ix, iz = [round(indices[idx].mean()) for idx in range(3)]

        return [ix, iy, iz][_slice_axes[view]]

    def get_centroid_pos(self, view="x-y", fraction=1):
        """
        Get position of slice containing centroid of above-threshold voxels.

        The centroid coordinate along a given axis is calculated as the
        unweighted mean of the coordinates of voxels with an intensity at
        least a given fraction of the maximum.

        **Parameters:**

        view : str, default="x-y"
            Orientation; can be "x-y", "y-z", or "x-z".

        fraction : float, default=1
            Minimum fraction of the maximum intensity that a voxel must record
            to be considered in the centroid calculation.
        """
        return self.idx_to_pos(
                self.get_centroid_idx(view, fraction), _slice_axes[view])

    def get_centroid_slice(self, view="x-y", fraction=1):
        """
        Get number of slice containing centroid of above-threshold voxels.

        The centroid coordinate along a given axis is calculated as the
        unweighted mean of the coordinates of voxels with an intensity at
        least a given fraction of the maximum.

        **Parameters:**

        view : str, default="x-y"
            Orientation; can be "x-y", "y-z", or "x-z".

        fraction : float, default=1
            Minimum fraction of the maximum intensity that a voxel must record
            to be considered in the centroid calculation.
        """
        return self.idx_to_slice(
                self.get_centroid_idx(view, fraction), _slice_axes[view])

    def get_orientation_codes(self, affine=None, source_type=None):
        """Get image orientation codes in order [row, column, slice] if 
        image was loaded in dicom-style orientation, or [column, row, slice] 
        if image was loaded in nifti-style orientation.

        This returns a list of code strings. Possible codes:
            "L" = Left (x axis)
            "R" = Right (x axis)
            "P" = Posterior (y axis)
            "A" = Anterior (y axis)
            "I" = Inferior (z axis)
            "S" = Superior (z axis)

        **Parameters:**
        
        affine : np.ndarray, default=None
            Custom affine matrix to use when determining orientation codes.
            If None, self.affine will be used.

        source_type : str, default=None
            Assumed source type to use when determining orientation codes. If 
            None, self.source_type will be used.
        """

        if affine is None:
            self.load()
            affine = self.affine
        codes = list(nibabel.aff2axcodes(affine))

        if source_type is None:
            source_type = self.source_type

        # Reverse codes for row and column of a dicom
        pairs = [("L", "R"), ("P", "A"), ("I", "S")]
        if "nifti" not in source_type:
            for i in range(2):
                switched = False
                for p in pairs:
                    for j in range(2):
                        if codes[i] == p[j] and not switched:
                            codes[i] = p[1 - j]
                            switched = True

        return codes

    def get_orientation_vector(self, affine=None, source_type=None):
        """Get image orientation as a row and column vector.

        **Parameters:**
        
        affine : np.ndarray, default=None
            Custom affine matrix to use when determining orientation vector.
            If None, self.affine will be used.

        source_type : str, default=None
            Assumed source type to use when determining orientation vector. If 
            None, self.source_type will be used.
        """

        self.load()

        if source_type is None:
            source_type = self.source_type
        if affine is None:
            affine = self.affine

        codes = self.get_orientation_codes(affine, source_type)
        if "nifti" in source_type:
            codes = [codes[1], codes[0], codes[2]]
        vecs = {
            "L": [1, 0, 0],
            "R": [-1, 0, 0],
            "P": [0, 1, 0],
            "A": [0, -1, 0],
            "S": [0, 0, 1],
            "I": [0, 0, -1],
        }
        return vecs[codes[0]] + vecs[codes[1]]

    def get_orientation_view(self):
        '''
        Determine view corresponding to the image's orientation.
        '''
        self.load()
        orient = self.get_orientation_vector()
        axis1 = ''.join([ax * abs(v) for ax, v in zip(_axes, orient[:3])])
        axis2 = ''.join([ax * abs(v) for ax, v in zip(_axes, orient[3:])])
        view = f'{axis1}-{axis2}'
        if view not in _plot_axes:
            view = f'{axis2}-{axis1}'
        if view not in _plot_axes:
            view = None

        return view

    def get_axes(self, col_first=False):
        """Return list of axis numbers in order [column, row, slice] if
        col_first is True, otherwise in order [row, column, slice]. The axis
        numbers 0, 1, and 2 correspond to x, y, and z, respectively.

        **Parameters:**
        
        col_first : bool, default=True
            If True, return axis numbers in order [column, row, slice] instead
            of [row, column, slice].
        """

        orient = np.array(self.get_orientation_vector()).reshape(2, 3)
        axes = [sum([abs(int(orient[i, j] * j)) for j in range(3)]) for i in
                range(2)]
        axes.append(3 - sum(axes))
        if col_first:
            return axes
        else:
            return [axes[1], axes[0], axes[2]]

    def get_machine(self, stations=_default_stations):

        machine = None
        if self.files:
            ds = pydicom.dcmread(self.files[0].path, force=True)
            try:
                station = ds.StationName
            except BaseException:
                station = None
            if station in stations:
                machine = stations[station]
        return machine

    def set_geometry(self):
        """Set geometric properties for this image. Should be called once image
        data has been loaded. Sets the following properties:

            Affine matrix (self.affine): 
                4x4 affine matrix. If initially None, this is computed 
                from self.voxel_size and self.origin.

            Voxel sizes (self.voxel_size):
                List of [x, y, z] voxel sizes in mm. If initially None, this 
                is computed from self.affine.

            Origin position (self.origin):
                List of [x, y, z] origin coordinates in mm. If initiall None,
                this is computed from self.affine.

            Number of voxels (self.n_voxels):
                List of number of voxels in the [x, y, z] directions. Computed 
                from self.data.shape.

            Limits (self.lims):
                List of minimum and maximum array positions in the [x, y, z]
                directions (corresponding the centres of the first and last 
                voxels). Calculated from standardised origin, voxel sizes, 
                and image shape.

            Image extent (self.image_extent):
                List of minimum and maximum positions of the edges of the array
                in the [x, y, z] directions. Calculated from self.lims +/-
                half a voxel size.

            Plot extent (self.plot_extent):
                Dict of plot extents for each orientation. Given in the form
                of a list [x1, x2, y1, y2], which is the format needed for 
                the <extent> argument for matplotlib plotting.

            Image sizes (self.image_size):
                List of [x, y, z] image sizes in mm.
        """

        # Ensure either affine or voxel sizes + origin are set
        if self.affine is None:
            if self.voxel_size is None:
                self.voxel_size = [1, 1, 1]
            if self.origin is None:
                self.origin = [0, 0, 0]

        # Set affine matrix, voxel sizes, and origin
        self.affine, self.voxel_size, self.origin = \
                get_geometry(
                    self.affine, self.voxel_size, self.origin, 
                    is_nifti=("nifti" in self.source_type),
                    shape=self.data.shape
                )

        # Set number of voxels in [x, y, z] directions
        for attribute in [
                "_affine_canonical", "_data_canonical",
                "_saffine", "_sdata", "_sorigin"]:
            if hasattr(self, attribute):
                delattr(self, attribute)
        self.standardise_data()
        self.n_voxels = [self._sdata.shape[1], self._sdata.shape[0],
                         self._sdata.shape[2]]

        # Set axis limits for standardised plotting
        self.lims = [
            (
                self._sorigin[i],
                self._sorigin[i] + (self.n_voxels[i] - 1) * self._svoxel_size[i],
            )
            for i in range(3)
        ]
        self.image_extent = [
            (
                self.lims[i][0] - self._svoxel_size[i] / 2,
                self.lims[i][1] + self._svoxel_size[i] / 2,
            )
            for i in range(3)
        ]
        self.plot_extent = {
            view: self.image_extent[x_ax] + self.image_extent[y_ax][::-1]
            for view, (x_ax, y_ax) in _plot_axes.items()
        }
        self.image_size = [
                self.n_voxels[i] * self._svoxel_size[i]
                for i in range(3)
                ]

    def get_length(self, ax):
        """Get image length along a given axis.

        **Parameters:**
        
        ax : str/int
            Axis along which to get length. Can be either "x", "y", "z" or
            0, 1, 2.
        """

        self.load()
        if not isinstance(ax, int):
            ax = _axes.index(ax)
        return abs(self.lims[ax][1] - self.lims[ax][0])

    def get_idx(self, view, sl=None, idx=None, pos=None):
        """Get an array index from either a slice number, index, or
        position. If <sl>, <idx>, and <pos> are all None, the index of the 
        central slice of the image in the orienation specified in <view> will 
        be returned.

        **Parameters:**
        
        view : str
            Orientation in which to compute the index. Can be "x-y", "y-z", or
            "x-z".

        sl : int, default=None
            Slice number. If given, this number will be converted to an index 
            and returned.

        idx : int, default=None
            Slice array index. If given and <sl> is None, this index will be 
            returned.

        pos : float, default=None
            Slice position in mm. If given and <sl> and <idx> are both None,
            this position will be converted to an index and returned.
        """

        if sl is not None:
            idx = self.slice_to_idx(sl, _slice_axes[view])
        elif idx is None:
            if pos is not None:
                idx = self.pos_to_idx(pos, _slice_axes[view])
            else:
                centre_pos = self.get_centre()[_slice_axes[view]]
                idx = self.pos_to_idx(centre_pos, _slice_axes[view])
        return idx

    def get_slice(
        self, view="x-y", sl=None, idx=None, pos=None, flatten=False, 
        force=True, shift=[None, None, None], force_standardise=True, **kwargs
    ):
        """Get a slice of the data in the correct orientation for plotting. 
        If <sl>, <pos>, and <idx> are all None, the central slice of the image
        in the orientation specified in <view> will be returned.

        **Parameters:**
        
        view : str
            Orientation; can be "x-y", "y-z", or "x-z".

        sl : int, default=None
            Slice number; used if not None.

        idx : int, default=None
            Slice array index; used if not None and <sl> is None.

        pos : float, default=None
            Slice position in mm; used if not None and <sl> and <idx> are both
            None.

        flatten : bool, default=False
            If True, the image will be summed across all slices in the 
            orientation specified in <view>; <sl>/<idx>/<pos> will be ignored.

        shift : list, default=[None, None, None]
            Translational shift in order [dx, dy, dz] to apply before returning
            slice.
        """

        # Get index
        idx = self.get_idx(view, sl=sl, idx=idx, pos=pos)

        # Invert x shift for nifti images
        if "nifti" in self.source_type:
            if shift[0] is not None:
                shift[0] = -shift[0]

        # Apply slice shift if requested
        z_ax = _slice_axes[view]
        if shift[z_ax] is not None:
            idx += round(shift[z_ax] / self.voxel_size[z_ax])

        # Check whether index and view match cached slice
        if hasattr(self, "_current_slice") and not force and not flatten:
            if self._current_idx == idx and self._current_view == view:
                return self._current_slice

        # Create slice
        transposes = {"x-y": [0, 1, 2], "y-z": [0, 2, 1], "x-z": [1, 2, 0]}
        transpose = pad_transpose(transposes[view], self.data.ndim)
        list(_plot_axes[view]) + [_slice_axes[view]]
        data = np.transpose(self.get_standardised_data(force=force_standardise), 
                            transpose)

        # Apply shifts in plane if requested
        x_ax, y_ax = _plot_axes[view]
        if shift[x_ax]:
            shift_x = round(shift[x_ax] / self.voxel_size[x_ax])
            data = np.roll(data, shift_x, axis=1)
            if shift_x > 0:
                data[:, :shift_x] = self.get_min()
            else:
                data[:, shift_x:] = self.get_min()
        if shift[y_ax]:
            shift_y = round(shift[y_ax] / self.voxel_size[y_ax])
            data = np.roll(data, shift_y, axis=0)
            if shift_y > 0:
                data[:shift_y, :] = self.get_min()
            else:
                data[shift_y:, :] = self.get_min()

        if flatten:
            return np.sum(data, axis=2)
        else:
            # Cache the slice
            self._current_idx = int(idx)
            self._current_view = view
            self._current_slice = data[:, :, int(idx)]
            return self._current_slice

    def assign_structure_set(self, structure_set):
        """
        Assign a structure set to this image.

        This does not affect the image associated with the structure set.
        Any previously assigned structure sets are cleared.

        **Parameters:**
        
        structure_set : skrt.structures.StructureSet
            A StructureSet object to assign to this image.
        """

        self.structure_sets = [structure_set]

    def add_structure_set(self, structure_set):
        """Add a structure set to be associated with this image. This does
        not affect the image associated with the structure set.

        **Parameters:**
        
        structure_set : skrt.structures.StructureSet
            A StructureSet object to assign to this image.
        """

        self.structure_sets.append(structure_set)

    def clear_structure_sets(self):
        """Clear all structure sets associated with this image."""

        self.structure_sets = []

    def add_dose(self, dose):
        """Add a Dose object to be associated with this image. This does not
        affect the image associated with the Dose object.

        **Parameters:**

        dose : skrt.dose.Dose
            A Dose object to assign to this image.
        """

        self.doses.append(dose)
        self.doses.sort()

    def clear_doses(self):
        """Clear all dose maps associated with this image."""

        self.doses = []

    def add_plan(self, plan):
        """Add a Plan object to be associated with this image. This does not
        affect the image associated with the Plan object.

        **Parameters:**

        plan : skrt.dose.Plan
            A Plan object to assign to this image.
        """

        self.plans.append(plan)
        self.plans.sort()

    def clear_plans(self):
        """Clear all plan maps associated with this image."""

        self.plans = []

    def get_mpl_kwargs(self, view, mpl_kwargs=None, scale_in_mm=True):
        """Get a dict of kwargs for plotting this image in matplotlib. This
        will create a default dict, which is updated to contain any kwargs
        contained in <mpl_kwargs>.

        The default parameters are:
            - "aspect":
                Aspect ratio determined from image geometry.
            - "extent": 
                Plot extent determined from image geometry.
            - "cmap":
                Colormap, self._default_cmap by default.
            - "vmin"/"vmax"
                Greyscale range to use; taken from self._default_vmin and
                self._default_vmax by default.

        For information on matplotlib colour maps, see:
            https://matplotlib.org/stable/gallery/color/colormap_reference.html

        **Parameters:**
        
        view : str
            Orientation (any of "x-y", "x-z", "y-z"); needed to compute
            correct aspect ratio and plot extent.

        mpl_kwargs : dict, default=None
            Dict of kwargs with which to overwrite default kwargs.

        scale_in_mm : bool, default=True
            If True, indicates that image will be plotted with axis scales in 
            mm; needed to compute correct aspect ratio and plot extent.
        """

        if mpl_kwargs is None:
            mpl_kwargs = {}

        # Set colour range
        for name in ["vmin", "vmax", "cmap"]:
            if name not in mpl_kwargs:
                mpl_kwargs[name] = getattr(self, f"_default_{name}")
        
        # Set image extent and aspect ratio
        extent = self.plot_extent[view]
        mpl_kwargs["aspect"] = 1
        x_ax, y_ax = _plot_axes[view]
        if not scale_in_mm:
            extent = [
                self.pos_to_slice(extent[0], x_ax, False),
                self.pos_to_slice(extent[1], x_ax, False),
                self.pos_to_slice(extent[2], y_ax, False),
                self.pos_to_slice(extent[3], y_ax, False),
            ]
            mpl_kwargs["aspect"] = abs(self.voxel_size[y_ax]
                                       / self.voxel_size[x_ax])
        mpl_kwargs["extent"] = extent

        return mpl_kwargs

    def view(self, images=None, **kwargs):
        """View self with BetterViewer along with any additional images in 
        <images>. Any ``**kwargs`` will be passed to BetterViewer
        initialisation.
        """

        from skrt.better_viewer import BetterViewer
        ims = [self]
        if images is not None:
            if isinstance(images, Image):
                ims.append(images)
            else:
                ims.extend(images)

        return BetterViewer(ims, **kwargs)

    def plot(
        self,
        view=None,
        sl=None,
        idx=None,
        pos=None,
        scale_in_mm=True,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        save_as=None,
        zoom=None,
        zoom_centre=None,
        intensity=None,
        mpl_kwargs=None,
        show=True,
        colorbar=False,
        colorbar_label=None,
        clb_kwargs=None,
        clb_label_kwargs=None,
        title=None,
        no_xlabel=False,
        no_ylabel=False,
        no_xticks=False,
        no_yticks=False,
        no_xtick_labels=False,
        no_ytick_labels=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        no_axis_labels=False,
        rois=None,
        roi_plot_type="contour",
        roi_opacity=None,
        roi_linewidth=None,
        consensus_type=None,
        exclude_from_consensus=None,
        consensus_color="blue",
        consensus_linewidth=None,
        legend=False,
        roi_kwargs=None,
        centre_on_roi=None,
        legend_bbox_to_anchor=None,
        legend_loc="lower left",
        dose=None,
        dose_opacity=0.5,
        dose_kwargs=None,
        grid=None,
        grid_opacity=1.0,
        grid_kwargs=None,
        flatten=False,
        xlim=None,
        ylim=None,
        zlim=None,
        shift=[None, None, None],
        mask=None,
        mask_threshold=0.5,
        masked=True,
        invert_mask=False,
        mask_color="black",
        jacobian=None,
        jacobian_opacity=0.8,
        jacobian_range=None,
        jacobian_kwargs=None,
        df=None,
        df_plot_type="quiver",
        df_spacing=30,
        df_opacity=None,
        df_kwargs=None,
    ):
        """Plot a 2D slice of the image.

        **Parameters:**
        
        view : str, default=None
            Orientation in which to plot the image. Can be any of 'x-y',
            'y-z', and 'x-z'.  If None, the initial view is chosen to match
            the image orienation.

        sl : int, default=None
            Slice number to plot. Takes precedence over <idx> and <pos> if not
            None. If all of <sl>, <idx>, and <pos> are None, the central
            slice will be plotted.

        idx : int, default=None
            Index of the slice in the array to plot. Takes precendence over
            <pos>.

        pos : float, default=None
            Position in mm of the slice to plot. Will be rounded to the nearest
            slice. Only used if <sl> and <idx> are both None.

        standardised : bool, default=True
            If True, a standardised version of the image array will be plotted
            such that the axis labels are correct.

        scale_in_mm : bool, default=True
            If True, axis labels will be in mm; otherwise, they will be slice
            numbers.

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot. If None, new axes will be created.

        gs : matplotlib.gridspec.GridSpec, default=None
            If not None and <ax> is None, new axes will be created on the
            current matplotlib figure with this gridspec.

        figsize : float, default=None
            Figure height in inches; only used if <ax> and <gs> are None.

        zoom : int/float/tuple, default=None
            Factor by which to zoom in. If a single int or float is given,
            the same zoom factor will be applied in all directions. If a tuple
            of three values is given, these will be used as the zoom factors
            in each direction in the order (x, y, z). If None, the image will
            not be zoomed in.

        zoom_centre : tuple, default=None
            Position around which zooming is applied. If None, the centre of
            the image will be used.

        colorbar : int/bool, default=False
            Indicate whether to display colour bar(s):
            - 1 or True: colour bar for main image;
            - 2: colour bars for main image and for any associated image
            or overlay;
            - 0 or False: no colour bar.

        colorbar_label : str, default='HU'
            Label for the colorbar, if drawn.

        clb_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to pyplot.colorbar().

        clb_label_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to colorbar.set_label().

        intensity : list, default=None
            Two-item list containing min and max intensity for plotting. 
            Supercedes 'vmin' and 'vmax' in <mpl_kwargs>.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        show : bool, default=True
            If True, the plotted figure will be shown via
            matplotlib.pyplot.show().

        title : str, default=None
            Custom title for the plot. If None, a title inferred from the image
            filename will be used. If False or '', no title will be added.

        no_xlabel : bool, default=False
            If True, the x axis will not be labelled.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        no_xticks : bool, default=False
            If True, ticks (and their labels) on the x axis will not be shown.

        no_yticks : bool, default=False
            If True, ticks (and their labels) on the y axis will not be shown.

        no_xtick_labels : bool, default=False
            If True, ticks on the x axis will not be labelled.

        no_ytick_labels : bool, default=False
            If True, ticks on the y axis will not be labelled.

        no_axis_label : bool, default=False
            If True, axis labels and axis values aren't shown.

        annotate_slice : bool/str/dict/list, default=False
            Specification of slice annotations:

            - bool: annotate with slice position (scale_in_mm True)
            or number (scale_in_mm False), in default colour (white).

            - str: annotate with slice position or number in colour
            specified by string.

            - dict: annotation dictionary, containing keyword-value pairs
            to be passed to annotate() method of figure axes.  The
            following defaults are defined:
                
                'text': slice position or number;
                'xy': (0.05, 0.93)
                'xycoords': 'axes fraction'
                'color': 'white'
                'fontsize': 'large'

            - list: list of annotation dictionaries

            For information on all parameters that can be passed to
            annotate() method, see:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
            
        major_ticks : float, default=None
            If not None, this value will be used as the interval between major
            tick marks. Otherwise, automatic matplotlib axis tick spacing will
            be used.

        minor_ticks : int, default=None
            If None, no minor ticks will be plotted. Otherwise, this value will
            be the number of minor tick divisions per major tick interval.

        ticks_all_sides : bool, default=False
            If True, major (and minor if using) tick marks will be shown above
            and to the right hand side of the plot as well as below and to the
            left. The top/right ticks will not be labelled.

        rois : int/str, default=None
            Option for which structure set should be plotted (if the Image
            owns any structure sets). Can be:

                - None: no structure sets will be plotted.
                - The index in self.structure_sets of the structure set
                  (e.g. to plot the newest structure set, use rois=-1)
                - 'all': all structure sets will be plotted.

        roi_plot_type : str, default='contour'
            ROI plotting type (see ROI.plot() for options).

        roi_opacity : float, default=None
            Opacity to use if plotting ROI as mask (i.e. roi_plot_type
            "mask", "filled", or "filled centroid"). If None, opacity
            will be 1 by default for solid mask plots and 0.3 by default
            for filled plots.

        roi_linewidth : float, default=None
            Width of ROI contour lines. If None, the matplotlib default setting 
            will be used.

        consensus_type : str, default=None
            If not None, the consensus of all ROIs will be plotting rather than
            plotting ROIs individually. Requires <rois> to be a single
            StructureSet. Options are "majority", "sum", "overlap", "staple".

        exclude_from_consensus : str, default=None
            If set to the name of an ROI and consensus_type is a valid 
            consensus type, this ROI will be excluded from the consensus 
            calculation and plotted separately on top of the consensus ROI.

        consensus_color : matplotlib color, default="white"
            Color in which to plot consensus contour.

        consensus_linewidth : float, default=None
            Linewidth of consensus contour. If None, the default matplotlib
            linewidth + 1 will be used (such that consensus contours are 
            thicker than standard contours).

        legend : bool, default=False
            If True, a legend will be drawn containing ROI names.

        roi_kwargs : dict, default=None
            Extra arguments to provide to ROI plotting via the ROI.plot()
            method.

        centre_on_roi : str, default=None
            Name of ROI on which to centre, if no idx/sl/pos is given.
            If <zoom> is given but no <zoom_centre>, the zoom will also centre
            on this ROI.

        legend_bbox_to_anchor : tuple, default=None
            Specify placement of ROI legend.
            - If a four-element tuple, the elements specify
              (x, y, width, height) of the legend bounding box.
            - If a two-element tuple, the elements specify (x, y) of
              the part of the legend bounding box specified by legend_loc.

        legend_loc : str, default='lower left'
            Legend location for ROI legend.

        dose : skrt.dose.Dose / int, default=None
            Dose field to overlay on the image. Can be either a skrt.dose.Dose
            object or an integer referring to the 
            dose at a given index in self.doses (e.g. to plot the last dose 
            assigned to this Image, set dose=-1).

        dose_opacity : float, default=0.5
            Opacity of overlaid dose field, if <dose> is not None.

        dose_kwargs : dict, default=None
            Extra arguments to provide to the mpl_kwargs argument in the dose 
            plotting method.

        xlim, ylim : tuples, default=None
            Custom limits on the x and y axes of the plot.

        mask : Image/list/ROI/str/StructureSet, default=None
            Image object representing a mask or a source from which
            an Image object can be initialised.  In addition to the
            sources accepted by the Image constructor, the source
            may be an ROI, a list of ROIs or a StructureSet.  In the
            latter cases, the mask image is derived from the ROI mask(s).

        mask_threshold : float, default=0.5
            Threshold for mask data.  Values above and below this value are
            set to True and False respectively.  Taken into account only
            if the mask image has non-boolean data.

        masked : bool, default=True
            If True and a mask is specified, the image is masked.

        invert_mask : bool, default=False
            If True and a mask is applied, the mask will be inverted.

        mask_color : matplotlib color, default="black"
            color in which to plot masked areas.

        jacobian : skrt.registration.Jacobian, default=None
            Jacobian determinannt to be overlaid on plot.  This parameter
            is ignored if a non-null value is specified for dose.

        jacobian_opacity : float, default=0.8
            Initial opacity of the overlaid jacobian determinant. Can later
            be changed interactively.

        jacobian_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the jacobian determinant. For options, see:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

            Some useful keywords are:

            - 'cmap': colormap (default='jacobian' - custom colour map).
            - 'interpolation': interpolation method (default='antialiased')

        df_plot_type : str, default='quiver'
            Option for initial plotting of deformation field. Can be 'quiver',
            'grid', 'x-displacement', 'y-displacement',
            'z-displacement', '3d-displacement', or 'none'.
            All quantities relate to the mapping of points from
            fixed image to moving image in image registration.

        df_spacing : int/tuple, default=30
            Spacing between arrows on the quiver plot/gridlines on the grid
            plot of a deformation field. Can be a single value for spacing in
            all directions, or a tuple with values for (x, y, z). Dimensions
            are mm if <scale_in_mm> is True, or voxels if <scale_in_mm> is
            False.

        df_opacity : float, default=0.5
            Initial opacity of the overlaid deformation field.

        df : str/skrt.registration.DeformationField/list, default=None
            Source(s) of deformation field(s) to overlay on each plot.

        df_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib when plotting
            the deformation field.

            For grid plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            Some useful keywords are:

            - 'linewidth': default=2
            - 'color': default='green'

            For quiver plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html.

        roi_plot_type : str, default='contour'
            Option for initial plot of ROIs. Can be 'contour', 'mask',
            'filled', or 'none'.
        grid : string/nifti/array/list, default=None
            Source(s) of grid array(s) to overlay on image
            (see valid image sources for <images>).

        grid_opacity : float, default=1.0
            Opacity of the overlaid grid.

        grid_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the grid. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
        """

        self.load()

        if not view:
            view = self.get_orientation_view()

        # Set up axes
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        self.load()

        # Get dose object to plot
        if isinstance(dose, int):
            dose_to_plot = self.doses[dose]
        else:
            dose_to_plot = dose

        # Get list of input ROI sources
        if rois is None:
            roi_input = []
        elif isinstance(rois, str) and rois == "all":
            roi_input = self.structure_sets
        elif isinstance(rois, int):
            roi_input = [self.structure_sets[rois]]
        elif not skrt.core.is_list(rois):
            roi_input = [rois]
        else:
            roi_input = rois

        # Get list of ROI objects to plot
        rois_to_plot = []
        n_structure_sets = 0
        if consensus_type is None:
            for roi in roi_input:
                if type(roi).__name__ == "ROI":
                    rois_to_plot.append(roi)
                elif type(roi).__name__ == "StructureSet":
                    rois_to_plot.extend(roi.get_rois())
                    n_structure_sets += 1
                elif isinstance(roi, int):
                    try:
                        rois_to_plot.extend(self.structure_sets[roi].get_rois())
                        n_structure_sets += 1
                    except IndexError:
                        raise IndexError(f"Index {roi} not found in Image.structure_sets!")
        else:
            if len(roi_input) != 1 \
               or type(roi_input[0]).__name__ != "StructureSet":
                raise TypeError("Consensus plots require a single StructureSet.")
            rois_to_plot = roi_input[0].get_rois()

        # If centering on an ROI, find index of its central slice
        roi_names = [roi.name for roi in rois_to_plot]
        if centre_on_roi:
            central_roi = rois_to_plot[roi_names.index(centre_on_roi)]
            idx = central_roi.get_mid_idx(view)
            if zoom and zoom_centre is None:
                zoom_centre = central_roi.get_zoom_centre(view)

        # Get image slice
        idx = self.get_idx(view, sl=sl, idx=idx, pos=pos)
        pos = self.idx_to_pos(idx, ax=_slice_axes[view])
        image_slice = self.get_slice(view, idx=idx, flatten=flatten, 
                                     shift=shift)

        # If required, apply mask to image slice
        if mask and masked:
            mask = get_mask(mask, mask_threshold, self)
            mask_slice = mask.get_slice(view, idx=idx, flatten=flatten,
                    shift=shift)
            if invert_mask:
                image_slice = np.ma.masked_where(mask_slice, image_slice)
            else:
                image_slice = np.ma.masked_where(~mask_slice, image_slice)

        # Initialise kwargs dicts
        if mpl_kwargs is None:
            mpl_kwargs = {}
        if roi_kwargs is None:
            roi_kwargs = {}
        roi_kwargs['no_xlabel'] = no_xlabel
        roi_kwargs['no_ylabel'] = no_ylabel
        roi_kwargs['no_xticks'] = no_xticks
        roi_kwargs['no_yticks'] = no_yticks
        roi_kwargs['no_xtick_labels'] = no_xtick_labels
        roi_kwargs['no_ytick_labels'] = no_ytick_labels
        if roi_linewidth is None:
            roi_linewidth = roi_kwargs.get("linewidth",
                    mpl.defaultParams["lines.linewidth"][0])
        roi_kwargs["linewidth"] = roi_linewidth
        if roi_opacity is None:
            roi_opacity = roi_kwargs.get("opacity",
                    0.3 if "filled" in roi_plot_type else 1)
        roi_kwargs["opacity"] = roi_opacity
        roi_kwargs["title"] = roi_kwargs.get("title", title) or ""

        if dose_kwargs is None:
            dose_kwargs = {}
        if clb_kwargs is None:
            clb_kwargs = {}
        if clb_label_kwargs is None:
            clb_label_kwargs = {}

        jacobian_kwargs = jacobian_kwargs or {}
        df_kwargs = df_kwargs or {}
        grid_kwargs = grid_kwargs or {}

        # Set defaults for clb_kwargs and clb_label_kwargs
        clb_kwargs['pad'] = clb_kwargs.get('pad', 0.06)
        clb_label_kwargs['labelpad'] = clb_label_kwargs.get('labelpad', 7)

        # Apply intensity window if given
        if 'auto' == intensity:
            mpl_kwargs["vmin"] = self.get_data().min()
            mpl_kwargs["vmax"] = self.get_data().max()
        elif intensity is not None:
            mpl_kwargs["vmin"] = intensity[0]
            mpl_kwargs["vmax"] = intensity[1]
        else:
            mpl_kwargs["vmin"] = mpl_kwargs.get("vmin", self._default_vmin)
            mpl_kwargs["vmax"] = mpl_kwargs.get("vmax", self._default_vmax)

        # Ensure colour map is defined, and set mask colour
        cmap = mpl_kwargs.get("cmap", self._default_cmap)
        if isinstance(cmap, str):
            cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
        cmap.set_bad(color=mask_color)
        mpl_kwargs["cmap"] = cmap

        # Plot the slice
        mesh = self.ax.imshow(
            image_slice, **self.get_mpl_kwargs(view, mpl_kwargs, scale_in_mm)
        )

        # If colour bar isn't to be plotted for current image,
        # use kwargs values relating to colour bar for overlay image.
        clb_kwargs2 = clb_kwargs if colorbar == -1 else {}
        clb_label_kwargs2 = clb_label_kwargs if colorbar == -1 else {}

        # Plot the dose 
        if dose is not None:
            dose_to_plot.plot(
                view=view,
                idx=idx,
                ax=self.ax,
                show=False,
                colorbar= max((colorbar - 1), -colorbar, 0),
                include_image=False, 
                opacity=dose_opacity, 
                title="",
                no_xlabel=no_xlabel,
                no_ylabel=no_ylabel,
                no_xticks=no_xticks,
                no_yticks=no_yticks,
                no_xtick_labels=no_xtick_labels,
                no_ytick_labels=no_ytick_labels,
                mpl_kwargs=dose_kwargs,
                mask=mask,
                mask_threshold=mask_threshold,
                masked=masked,
                invert_mask=invert_mask,
                mask_color=mask_color,
                clb_kwargs=clb_kwargs2,
                clb_label_kwargs=clb_label_kwargs2,
            )

        # Plot the deformation field.
        if df:
            df.plot(
                    view=view,
                    idx=idx,
                    ax=self.ax,
                    show=False,
                    colorbar= max((colorbar - 1), -colorbar, 0),
                    include_image=False, 
                    df_opacity=df_opacity, 
                    title="",
                    no_xlabel=no_xlabel,
                    no_ylabel=no_ylabel,
                    no_xticks=no_xticks,
                    no_yticks=no_yticks,
                    no_xtick_labels=no_xtick_labels,
                    no_ytick_labels=no_ytick_labels,
                    mask=mask,
                    mask_threshold=mask_threshold,
                    masked=masked,
                    invert_mask=invert_mask,
                    mask_color=mask_color,
                    df_plot_type=df_plot_type,
                    df_spacing=df_spacing,
                    clb_kwargs=clb_kwargs2,
                    clb_label_kwargs=clb_label_kwargs2,
                    **df_kwargs,
                    )

        # Plot the Jacobian determinant.
        if jacobian and not dose_to_plot:
            if jacobian_range is not None:
                jacobian_intensity = jacobian_range
            elif "vmin" in jacobian_kwargs and "vmax" in jacobian_kwargs:
                jacobian_intensity = (
                        jacobian_kwargs["vmin"], jacobian_kwargs["vmax"])
            else:
                jacobian_intensity = None
            jacobian.plot(
                view=view,
                idx=idx,
                ax=self.ax,
                show=False,
                colorbar=(max((colorbar - 1), -colorbar, 0)),
                include_image=False, 
                opacity=jacobian_opacity, 
                intensity=jacobian_intensity,
                title="",
                no_xlabel=no_xlabel,
                no_ylabel=no_ylabel,
                no_xticks=no_xticks,
                no_yticks=no_yticks,
                no_xtick_labels=no_xtick_labels,
                no_ytick_labels=no_ytick_labels,
                mpl_kwargs=jacobian_kwargs,
                mask=mask,
                mask_threshold=mask_threshold,
                masked=masked,
                invert_mask=invert_mask,
                mask_color=mask_color,
                clb_kwargs=clb_kwargs2,
                clb_label_kwargs=clb_label_kwargs2,
            )

        # Plot the grid array.
        if grid and not dose_to_plot:
            grid.plot(
                view=view,
                idx=idx,
                ax=self.ax,
                show=False,
                include_image=False, 
                opacity=grid_opacity, 
                title="",
                no_xlabel=no_xlabel,
                no_ylabel=no_ylabel,
                no_xticks=no_xticks,
                no_yticks=no_yticks,
                no_xtick_labels=no_xtick_labels,
                no_ytick_labels=no_ytick_labels,
                mpl_kwargs=grid_kwargs,
                mask=mask,
                mask_threshold=mask_threshold,
                masked=masked,
                invert_mask=invert_mask,
                mask_color=mask_color,
                clb_kwargs=clb_kwargs2,
                clb_label_kwargs=clb_label_kwargs2,
            )

        # Plot ROIs
        plotted_rois = []
        if consensus_type is None:
            for roi in rois_to_plot:
                if roi.on_slice(view, pos=pos):
                    roi.plot(
                        view,
                        pos=pos,
                        ax=self.ax,
                        plot_type=roi_plot_type,
                        show=False,
                        include_image=False,
                        no_invert=True,
                        color=roi.get_color_from_kwargs(roi_kwargs),
                        **roi_kwargs
                    )
                    plotted_rois.append(roi)

        # Consensus plot
        else:
            roi_input[0].plot(
                view,
                pos=pos,
                ax=self.ax,
                plot_type=roi_plot_type,
                show=False,
                include_image=False,
                no_invert=True,
                consensus_type=consensus_type,
                exclude_from_consensus=exclude_from_consensus,
                legend=legend,
                consensus_color=consensus_color,
                consensus_linewidth=consensus_linewidth,
                **roi_kwargs
            )

        # Label axes
        self.label_ax(
            view,
            idx,
            scale_in_mm,
            title,
            no_xlabel,
            no_ylabel,
            no_xticks,
            no_yticks,
            no_xtick_labels,
            no_ytick_labels,
            annotate_slice,
            major_ticks,
            minor_ticks,
            ticks_all_sides,
            no_axis_labels
        )
        self.zoom_ax(view, zoom, zoom_centre)

        # Set custom x and y limits
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        # Add ROI legend
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_ax, y_ax = _plot_axes[view] 
        roi_handles = []
        if legend and consensus_type is None:
            for roi in plotted_rois:

                # Check whether this ROI is currently visible
                roi_xlim = roi.get_extent(view=view, ax=x_ax, 
                                          single_slice=True, pos=pos)
                if max(roi_xlim) < min(xlim) or min(roi_xlim) > max(xlim):
                    continue
                roi_ylim = roi.get_extent(view=view, ax=y_ax, single_slice=True, pos=pos)
                if max(roi_ylim) < min(ylim) or min(roi_ylim) > max(ylim):
                    continue

                # Define ROI handle.
                roi_color = roi.get_color_from_kwargs(roi_kwargs)
                roi_handle = roi.get_patch(
                        roi_plot_type, roi_color, roi_opacity, roi_linewidth)
                if roi_handle:
                    roi_handles.append(roi_handle)

            # Draw ROI legend
            if legend and len(roi_handles):
                self.ax.legend(
                    handles=roi_handles, bbox_to_anchor=legend_bbox_to_anchor,
                    loc=legend_loc, facecolor="white",
                    framealpha=1
                )

        # Add colorbar
        clb_label = colorbar_label if colorbar_label is not None \
                else self._default_colorbar_label
        if colorbar > 0 and mpl_kwargs.get("alpha", 1) > 0:
            if "jacobian" == cmap.name:
                scalar_mappable = matplotlib.cm.ScalarMappable(
                        norm=matplotlib.colors.Normalize(-1, 2), cmap=cmap)
            else:
                scalar_mappable = mesh
            clb = self.fig.colorbar(scalar_mappable, ax=self.ax, **clb_kwargs)
            clb.set_label(clb_label, **clb_label_kwargs)
            clb.solids.set_edgecolor("face")

        # Display image
        plt.tight_layout()
        if show:
            plt.show()

        # Save to file
        if save_as:
            self.fig.savefig(save_as, bbox_inches="tight", pad_inches=0.03)
            plt.close()

    def label_ax(
        self,
        view,
        idx,
        scale_in_mm=True,
        title=None,
        no_xlabel=False,
        no_ylabel=False,
        no_xticks=False,
        no_yticks=False,
        no_xtick_labels=False,
        no_ytick_labels=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        no_axis_labels=False,
        **kwargs,
    ):

        x_ax, y_ax = _plot_axes[view]

        # Set title
        if title is None:
            title = self.title
        if title:
            # Previously passed: pad=8
            self.ax.set_title(title)

        # Set axis labels
        units = " (mm)" if scale_in_mm else ""
        # Previously passed: labelpad=0
        if not no_xlabel:
            self.ax.set_xlabel(f"${_axes[x_ax]}${units}")
        if no_xticks:
            self.ax.set_xticks([], [])
        elif no_xtick_labels:
            self.ax.set_xticklabels([])
        if not no_ylabel:
            self.ax.set_ylabel(f"${_axes[y_ax]}${units}")
        if no_yticks:
            self.ax.set_yticks([], [])
        elif no_ytick_labels:
            self.ax.set_yticklabels([])

        #else:

        # Add annotation(s).
        if annotate_slice:
            z_ax = _axes[_slice_axes[view]]
            pos = self.idx_to_pos(idx, z_ax)
            im_slice = self.idx_to_slice(idx, z_ax)
            if hasattr(self, "get_n_voxels"):
                n_slice = self.get_n_voxels()[_axes.index(z_ax)]
            else:
                n_slice = self.image.get_n_voxels()[_axes.index(z_ax)]

            # Set default string, indicating slice index or position.
            if scale_in_mm:
                z_str = f"${z_ax}$ = {pos:.1f} mm"
            else:
                z_str = f"${z_ax}$ = {im_slice} of {n_slice}"

            # Set default font colour.
            if matplotlib.colors.is_color_like(annotate_slice):
                color = annotate_slice
            else:
                color = "white"

            # Map from value of annotate_slice
            # to a list of annotation dictionaries.
            if isinstance(annotate_slice, (list, tuple)):
                annotations = annotate_slice
            elif isinstance(annotate_slice, dict):
                annotations = [annotate_slice]
            else:
                annotations = [{'text': z_str, 'color': color}]

            # Annotate slice, with defaults set for some annotate() parameters.
            for annotation in annotations:
                annotation_now = copy.deepcopy(annotation)
                annotation_now['text'] = annotation.get('text', z_str) or z_str
                # By default, multiple annotations written
                # one on top of the other...
                annotation_now['xy'] = annotation.get('xy', (0.04, 0.91))
                annotation_now['xycoords'] = annotation.get(
                    'xycoords', 'axes fraction')
                annotation_now['color'] = annotation.get('color', color)
                annotation_now['fontsize'] = annotation.get('fontsize','large')
                self.ax.annotate(**annotation_now)

        # Adjust tick marks
        if not no_axis_labels:
            if major_ticks:
                self.ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
                self.ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
            if minor_ticks:
                self.ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
                self.ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
            if ticks_all_sides:
                self.ax.tick_params(bottom=True, top=True, left=True, right=True)
                if minor_ticks:
                    self.ax.tick_params(
                        which="minor", bottom=True, top=True, left=True, right=True
                    )

        # Remove axis labels if needed
        if no_axis_labels:
            plt.axis("off")

    def zoom_ax(self, view, zoom=None, zoom_centre=None):
        """Zoom in on axes if needed."""

        if not zoom or isinstance(zoom, str):
            return
        zoom = skrt.core.to_list(zoom)
        x_ax, y_ax = _plot_axes[view]
        if zoom_centre is None:
            im_centre = self.get_centre()
            mid_x = im_centre[x_ax]
            mid_y = im_centre[y_ax]
        else:
            if len(zoom_centre) == 2:
                mid_x, mid_y = zoom_centre
            else:
                mid_x, mid_y = zoom_centre[x_ax], zoom_centre[y_ax]

        # Calculate new axis limits
        init_xlim = self.ax.get_xlim()
        init_ylim = self.ax.get_ylim()
        xlim = [
            mid_x - (init_xlim[1] - init_xlim[0]) / (2 * zoom[x_ax]),
            mid_x + (init_xlim[1] - init_xlim[0]) / (2 * zoom[x_ax]),
        ]
        ylim = [
            mid_y - (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax]),
            mid_y + (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax]),
        ]

        # Set axis limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def idx_to_pos(self, idx, ax, standardise=True):
        """Convert an array index to a position in mm along a given axis."""

        self.load()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if standardise:
            origin = self._sorigin
            voxel_size = self._svoxel_size
        else:
            origin = self.origin
            voxel_size = self.voxel_size
        return origin[i_ax] + idx * voxel_size[i_ax]

    def pos_to_idx(self, pos, ax, return_int=True, standardise=True):
        """Convert a position in mm to an array index along a given axis."""

        self.load()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if standardise:
            origin = self._sorigin
            voxel_size = self._svoxel_size
        else:
            origin = self.origin
            voxel_size = self.voxel_size
        idx = (pos - origin[i_ax]) / voxel_size[i_ax]
        if return_int:
            return round(idx)
        else:
            return idx

    def idx_to_slice(self, idx, ax):
        """Convert an array index to a slice number along a given axis."""

        self.load()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if i_ax == 2:
            return self.n_voxels[i_ax] - idx
        else:
            return idx + 1

    def slice_to_idx(self, sl, ax):
        """Convert a slice number to an array index along a given axis."""

        self.load()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if i_ax == 2:
            return self.n_voxels[i_ax] - sl
        else:
            return sl - 1

    def pos_to_slice(self, pos, ax, return_int=True, standardise=True):
        """Convert a position in mm to a slice number along a given axis."""

        sl = self.idx_to_slice(self.pos_to_idx(
            pos, ax, return_int, standardise=standardise), ax)
        if return_int:
            return round(sl)
        else:
            return sl

    def slice_to_pos(self, sl, ax, standardise=True):
        """Convert a slice number to a position in mm along a given axis."""

        return self.idx_to_pos(self.slice_to_idx(sl, ax), ax, standardise)

    def get_centre(self):
        """Get position in mm of the centre of the image."""

        self.load()
        return np.array([np.mean(self.lims[i]) for i in range(3)])

    def get_range(self, ax="z"):
        """Get range of the image in mm along a given axis."""

        i_ax = _axes.index(ax) if ax in _axes else ax
        origin = self.get_origin()[i_ax]
        return [origin, origin + (self.n_voxels[i_ax] - 1)
                * self.voxel_size[i_ax]]

    def get_length(self, ax="z"):
        """Get total length of image."""

        i_ax = _axes.index(ax) if ax in _axes else ax
        return abs(self.n_voxels[i_ax] * self.voxel_size[i_ax])

    def get_voxel_coords(self):
        """Get arrays of voxel coordinates in each direction."""

        return

    def set_ax(
        self,
        view=None,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        colorbar=False
    ):
        """Set up axes for this Image."""

        if figsize is None:
            figsize = _default_figsize

        skrt.image.set_ax(
            self, 
            view=view,
            ax=ax,
            gs=gs,
            figsize=figsize,
            zoom=zoom,
            colorbar=colorbar,
            aspect_getter=self.get_plot_aspect_ratio,
        )

    def get_plot_aspect_ratio(
        self, view, zoom=None, n_colorbars=0, figsize=None
    ):
        """Estimate the ideal width/height ratio for a plot of this image
        in a given orientation.

        view : str
            Orientation ('x-y', 'y-z', or 'x-z')

        zoom : float/list, default=None
            Zoom factors; either a single value for all axes, or three values
            in order (x, y, z).

        n_colorbars : int, default=0
            Number of colorbars to make space for.
        """

        if figsize is None:
            figsize = _default_figsize

        # Get length of the image in the plot axis directions
        self.load()
        x_ax, y_ax = _plot_axes[view]
        x_len = abs(self.lims[x_ax][1] - self.lims[x_ax][0])
        y_len = abs(self.lims[y_ax][1] - self.lims[y_ax][0])

        # Add padding for axis labels and title
        font = mpl.rcParams["font.size"] / 72
        y_pad = 2 * font
        if self.title:
            y_pad += 1.5 * font
        max_y_digits = np.floor(np.log10(max([abs(lim) for lim in
                                              self.lims[y_ax]])))
        minus_sign = any([lim < 0 for lim in self.lims[y_ax]])
        x_pad = (0.7 * max_y_digits + 1.2 * minus_sign + 1) * font

        # Account for zoom
        if zoom:
            zoom = skrt.core.to_list(zoom)
            x_len /= zoom[x_ax]
            y_len /= zoom[y_ax]

        # Add padding for colorbar(s).
        # Numerator of 10 here is fairly arbitrary...
        x_len *= (1 + (n_colorbars * 10) / figsize)

        # Return estimated width ratio
        total_y = figsize + y_pad
        total_x = figsize * x_len / y_len + x_pad
        return total_x / total_y

    def downsample(self, downsampling):
        """Apply downsampling to the image array. Can be either a single
        value (to downsampling equally in all directions) or a list of 3
        values."""

        # Get downsampling in each direction
        if skrt.core.is_list(downsampling):
            if len(downsampling) != 3:
                raise TypeError("<downsample> must contain 3 elements!")
            dx, dy, dz = downsampling
        else:
            dx = dy = dz = downsampling

        # Apply to image array
        self.data = downsample(self.data, dx, dy, dz)

        # Adjust voxel sizes
        self.voxel_size = [v * d for v, d in zip(self.voxel_size,
                                                 [dx, dy, dz])]
        self.affine = None

        # Reset geometric properties of this image
        self.set_geometry()

    def get_nifti_array_and_affine(self, standardise=False):
        """Get image array and affine matrix in canonical nifti
        configuration."""

        # Convert dicom-style array to nifti
        if "nifti" not in self.source_type:
            data = self.get_data().transpose(1, 0, 2)[:, ::-1, :]
            affine = self.affine.copy()
            affine[0, :] = -affine[0, :]
            affine[1, 3] = -(affine[1, 3] + (data.shape[1] - 1)
                             * self.voxel_size[1])

        # Use existing nifti array
        else:
            data = self.get_data()
            affine = self.affine

        # Convert to canonical if requested
        if standardise:
            nii = nibabel.as_closest_canonical(nibabel.Nifti1Image(data,
                                                                   affine))
            return nii.get_fdata(), nii.affine
        else:
            return data, affine

    def get_dicom_array_and_affine(self, standardise=False):
        """Get image array and affine matrix in dicom configuration."""

        # Return standardised dicom array
        if standardise:
            self.standardise_data()
            return self._sdata, self._saffine

        # Convert nifti-style array to dicom
        if "nifti" in self.source_type:
            data = self.get_data().transpose(1, 0, 2)[::-1, :, :]
            affine = self.affine.copy()
            affine[0, :] = -affine[0, :]
            affine[1, 3] = -(affine[1, 3] + (data.shape[0] - 1)
                             * self.voxel_size[1])
            if standardise:
                nii = nibabel.as_closest_canonical(
                    nibabel.Nifti1Image(data, affine))
                return nii.get_fdata(), nii.affine
            else:
                return data, affine

        # Otherwise, return array as-is
        return self.get_data(), self.affine

    def write(
        self,
        outname,
        standardise=False,
        write_geometry=True,
        nifti_array=False,
        overwrite=True,
        header_source=None,
        patient_id=None,
        modality=None,
        root_uid=None,
        verbose=True,
        header_extras={},
    ):
        """Write image data to a file. The filetype will automatically be
        set based on the extension of <outname>:

            (a) ``*``.nii or ``*``.nii.gz: Will write to a nifti file with
            canonical nifti array and affine matrix.

            (b) ``*``.npy: Will write the dicom-style numpy array to a binary
            filem unless <nifti_array> is True, in which case the canonical
            nifti-style array will be written. If <write_geometry> is True,
            a text file containing the voxel sizes and origin position will
            also be written in the same directory.

            (c) ``*``.dcm: Will write to dicom file(s) (1 file per x-y slice)
            in the directory of the filename given, named by slice number.

            (d) No extension: Will create a directory at <outname> and write
            to dicom file(s) in that directory (1 file per x-y slice), named
            by slice number.

        If (c) or (d) (i.e. writing to dicom), the header data will be set in
        one of three ways:

            * If the input source was not a dicom, <dicom_for_header> is None,
              a brand new dicom with freshly generated UIDs will be created.
            * If <dicom_for_header> is set to the path to a dicom file, that
              dicom file will be used as the header.
            * Otherwise, if the input source was a dicom or directory
              containing dicoms, the header information will be taken from the
              input dicom file.

        In addition, when writing to dicom, the header obtained as outlined
        above will be updated via attribute-value pairs passed
        via the dictionary header_extras.
        """

        outname = os.path.expanduser(outname)
        self.load()

        # Write to nifti file
        if outname.endswith(".nii") or outname.endswith(".nii.gz"):
            data, affine = self.get_nifti_array_and_affine(standardise)
            if data.dtype == bool:
                data = data.copy().astype(np.int8)
            write_nifti(outname, data, affine)
            if verbose:
                print("Wrote to NIfTI file:", outname)

        # Write to numpy file
        elif outname.endswith(".npy"):
            if nifti_array:
                data, affine = self.get_nifti_array_and_affine(standardise)
            else:
                data, affine = self.get_dicom_array_and_affine(standardise)
            if not write_geometry:
                affine = None
            write_npy(outname, data, affine)
            if verbose:
                print("Wrote to numpy file:", outname)

        # Write to dicom
        else:

            # Get name of dicom directory
            if outname.endswith(".dcm"):
                outdir = os.path.abspath(os.path.dirname(outname))
                filename = os.path.basename(outname)
            else:
                outdir = outname
                filename = None

            # Get header source
            if header_source is None and self.source_type == "dicom":
                header_source = self.source
            data, affine = self.get_dicom_array_and_affine(standardise)
            orientation = self.get_orientation_vector(affine, "dicom")
            dicom_writer = DicomWriter(
                outdir,
                data,
                affine,
                overwrite,
                header_source,
                orientation,
                patient_id,
                modality,
                root_uid,
                header_extras,
                type(self).__name__,
                filename
            )
            self.dicom_dataset = dicom_writer.write()
            if verbose:
                print("Wrote dicom file(s) to directory:", outdir)

    def copy_dicom(self, outdir="image_dicom", overwrite=True, sort=True,
                   *args, **kwargs):
        """
        Copy source dicom files.

        **Parameters:**

        outdir : pathlib.Path/str, default="image_dicom"
            Path to directory to which source files are to be copied.

        overwrite : bool, default=True
            If True, delete and recreate <outdir> before copying
            files.  If False and <outdir> exists already, a file
            is copied only if this doesn't mean overwriting an
            existing file.

        sort : bool, default=True
            If True, copied dicom files will be named by instance number
            if all files have a different instance number, or else
            will be numbered sequentially from 1, in order of increasing
            z-coordinate.  If False, files are copied to the output directory
            with their names unaltered.

        args : list
            Arguments to be ignored.

        kwargs : dict
            Keyword arguments to be ignored.
        """
        self.load()

        # Check that image has dicom files to be copied.
        z_paths = getattr(self, "_z_paths", {})
        if not z_paths:
            if 1 == len(self.files):
                z_paths = {0: self.files[0].path}
            else:
                return

        # Define the output directory.
        outdir = skrt.core.make_dir(outdir, overwrite)

        # If sorting, obtain dictionary of file indices.
        # Use instance numbers as indices of all files have a different,
        # non-null instance number.  Otherwise, use sequential integers.
        if sort:
            z_instance_numbers = getattr(self, "_z_instance_numbers", {}) or {}
            instance_numbers = list(z_instance_numbers.values())
            if (not instance_numbers or (None in instance_numbers) or
                    (len(instance_numbers) != len(set(instance_numbers)))):
                z_instance_numbers = {z: idx + 1
                        for idx, z in enumerate(sorted(z_paths.keys()))}

        # Loop over image files.
        for z, path in sorted(z_paths.items()): 
            if sort:
                outpath = outdir / f"{z_instance_numbers[z]}.dcm"
            else:
                outpath = outdir / Path(path).name
            if overwrite or not outpath.exists():
                shutil.copy2(path, outpath)

    def get_coords(self):
        """Get grids of x, y, and z coordinates for each voxel in the image."""

        # Make coordinates
        coords_1d = [
            np.arange(
                self.origin[i],
                self.origin[i] + self.n_voxels[i] * self.voxel_size[i],
                self.voxel_size[i],
            )
            for i in range(3)
        ]
        return np.meshgrid(*coords_1d)

    def transform(self, scale=1, translation=[0, 0, 0], rotation=[0, 0, 0],
            centre=[0, 0, 0], resample="fine", restore=True, order=1,
            fill_value=None):
        """Apply three-dimensional similarity transform using scikit-image.

        The image is first translated, then is scaled and rotated
        about the centre coordinates


        **Parameters:**
        
        scale : float, default=1
            Scaling factor.

        translation : list, default=[0, 0, 0]
            Translation in mm in the [x, y, z] directions.

        rotation : float, default=0
            Euler angles in degrees by which to rotate the image.
            Angles are in the order pitch (rotation about x-axis),
            yaw (rotation about y-axis), roll (rotation about z-axis).

        centre : list, default=[0, 0, 0]
            Coordinates in mm in [x, y, z] about which to perform rotation
            and scaling of translated image.

        resample: float/string, default='coarse'
            Resampling to be performed before image transformation.
            If resample is a float, then the image is resampled to that
            this is the voxel size in mm along all axes.  If the
            transformation involves scaling or rotation in an image
            projection where voxels are non-square:
            if resample is 'fine' then voxels are resampled to have
            their smallest size along all axes;
            if resample is 'coarse' then voxels are resampled to have
            their largest size along all axes.

        restore: bool, default=True
            In case that image has been resampled:
            if True, restore original voxel size for transformed iamge;
            if False, keep resampled voxel size for transformed image.

        order: int, default = 1
            Order of the b-spline used in interpolating voxel intensity values.

        fill_value: float/None, default = None
            Intensity value to be assigned to any voxels in the resized
            image that are outside the original image.  If set to None,
            the minimum intensity value of the original image is used.
        """

        self.load()

        # Decide whether to perform resampling.
        # The scipy.ndimage function affine_transform() assumes
        # that voxel sizes are the same # in all directions, so resampling
        # is necessary if the transformation # affects a projection in which
        # voxels are non-square.  In other cases resampling is optional.
        small_number = 0.1
        voxel_size = tuple(self.voxel_size)
        voxel_size_min = min(voxel_size)
        voxel_size_max = max(voxel_size)
        if isinstance(resample, int) or isinstance(resample, float):
            image_resample = True
        else:
            image_resample = False
        if not image_resample:
            if abs(voxel_size_max - voxel_size_min) > 0.1:
                if (scale - 1.) > small_number:
                    image_resample = True
                pitch, yaw, roll = rotation
                dx, dy, dz = voxel_size
                if abs(pitch) > small_number and abs(dy - dz) > small_number:
                    image_resample = True
                if abs(yaw) > small_number and abs(dz - dx) > small_number:
                    image_resample = True
                if abs(roll) > small_number and abs(dx - dy) > small_number:
                    image_resample = True
            if image_resample and resample not in ['coarse', 'fine']:
                resample = 'fine'

        if image_resample:
            if 'fine' == resample:
                resample_size = voxel_size_min
            elif 'coarse' == resample:
                resample_size = voxel_size_max
            else:
                resample_size = resample
            self.resample(voxel_size=resample_size, order=order)

        # Obtain rotation in radians
        pitch, yaw, roll = [math.radians(x) for x in rotation]

        # Obtain translation in pixel units
        idx, idy, idz = [translation[i] / self.voxel_size[i] for i in range(3)]

        # Obtain centre coordinates in pixel units
        xc, yc, zc = centre
        ixc = self.pos_to_idx(xc, "x", False)
        iyc = self.pos_to_idx(yc, "y", False)
        izc = self.pos_to_idx(zc, "z", False)

        # Overall transformation matrix composed from
        # individual transformations, following suggestion at:
        # https://stackoverflow.com/questions/25895587/
        #     python-skimage-transform-affinetransform-rotation-center
        # This gives control over the order in which
        # the individaul transformation are performed
        # (translation before rotation), and allows definition
        # of point about which rotation and scaling are performed.
        tf_translation = skimage.transform.SimilarityTransform(
                translation=[-idy, -idx, -idz], dimensionality=3)
        tf_centre_shift = skimage.transform.SimilarityTransform(
                translation=[-iyc, -ixc, -izc], dimensionality=3)
        tf_rotation = skimage.transform.SimilarityTransform(
                rotation=[yaw, pitch, roll], dimensionality=3)
        tf_scale = skimage.transform.SimilarityTransform(
                scale=1. / scale, dimensionality=3)
        tf_centre_shift_inv = skimage.transform.SimilarityTransform(
                translation=[iyc, ixc, izc], dimensionality=3)

        matrix = tf_translation + tf_centre_shift + tf_rotation + tf_scale \
                 + tf_centre_shift_inv

        # Set fill value
        if fill_value is None:
            fill_value = self.data.min()

        # Apply transform
        self.data = scipy.ndimage.affine_transform(self.data, matrix,
                order=order, cval=fill_value)

        # Revert to original voxel size
        if image_resample and restore:
            self.resample(voxel_size=voxel_size, order=order)

        # Remove prior standardised data
        if hasattr(self, "_sdata"):
            del self._sdata
            del self._saffine

        return None

    def translate_origin(self, translation=[0, 0, 0]):
        """Translate origin, effectively changing image position.

        **Parameter:**

        translation : list, default=[0, 0, 0]
            Translation in mm in the [x, y, z] directions.
        """
        self.load()
        self.origin = [self.get_origin()[i] + translation[i] for i in range(3)]
        self.affine = None
        self.set_geometry()

    def has_same_data(self, im, max_diff=0.005):
        """Check whether this Image has the same data as
        another Image <im> (i.e. same array shape and same data values),
        with tolerance <max_diff> on agreement of data values."""

        same = self.get_data().shape == im.get_data().shape
        if same:
            same *= np.all(abs(self.get_data() - im.get_data()) < max_diff)

        return same

    def has_same_geometry(self, im, max_diff=0.005, standardise=False,
                          force_standardise=True):
        """
        Check whether this Image has the same geometric properties as
        another Image (i.e. same origins and voxel sizes within tolerance,
        same shapes).

        **Parameters:**
        im : skrt.image.Image
            Image with which to compare geometry.

        max_diff : float, default=0.005
            Maximum difference accepted between components of origins
            and voxel sizes.

        standardise : bool, default=False
            If False, geometry is compared for the images as loaded;
            otherwise, geometry is compared for the images in standard
            dicom-style orientation, such that [column, row, slice] corresponds
            to the [x, y, z] axes.

        force_standardise : bool, default=True
            If True, the standardised image will be recomputed from self.data 
            even if it has previously been computed.
        """
        
        shape = self.get_data(standardise, force_standardise).shape
        origin = self.get_origin(standardise, force_standardise)
        voxel_size = self.get_voxel_size(standardise, force_standardise)
        im_shape = im.get_data(standardise, force_standardise).shape
        im_origin = im.get_origin(standardise, force_standardise)
        im_voxel_size = im.get_voxel_size(standardise, force_standardise)

        same = (shape == im_shape)
        same *= np.all([abs(origin[i] - im_origin[i]) < max_diff
                            for i in range(3)])
        same *= np.all([abs(voxel_size[i] - im_voxel_size[i]) < max_diff
                            for i in range(3)])
    
        return same

    def crop(self, xlim=None, ylim=None, zlim=None):
        """
        Crop the image to a given x, y, z range in mm. If any are None, the 
        image will not be cropped in that direction.
        """

        lims = [xlim, ylim, zlim]
        if all([lim is None for lim in lims]):
            return

        # Ensure DICOM representation for cropping.
        self.load()
        if "nifti" in self.source_type:
            im = self.astype("dicom")
        else:
            im = self

        for i_ax, lim in enumerate(lims):

            if lim is None:
                continue

            # Find array indices at which to crop
            i1 = im.pos_to_idx(lims[i_ax][0], ax=i_ax, return_int=False)
            i2 = im.pos_to_idx(lims[i_ax][1], ax=i_ax, return_int=False)
            i_big, i_small = i2, i1
            if i1 > i2:
                i_big, i_small = i_small, i_big
            i_small = int(np.floor(round(i_small, 3) + 0.5))
            i_big = int(np.floor(round(i_big, 3) + 0.5))

            # Ensure indices are within image range
            if i_small < 0:
                i_small = 0
            if i_big > im.n_voxels[i_ax]:
                i_big = im.n_voxels[i_ax]

            # Crop the data array
            ax_to_slice = im.get_axes().index(i_ax)
            im.data = im.data.take(indices=range(i_small, i_big),
                                       axis=ax_to_slice)

            # Reset origin position
            if im.image_extent[i_ax][1] > im.image_extent[i_ax][0]:
                im.origin[i_ax] = im.idx_to_pos(i_small, ax=i_ax)
            else:
                im.origin[i_ax] = im.idx_to_pos(i_big, ax=i_ax)

        # Reset image geometry
        im.affine = get_geometry(None, im.voxel_size, im.origin)[0]

        # Revert to original representation.
        if "nifti" in self.source_type:
            im = im.astype("nifti")
            self.data = im.data.copy()
            self.affine, self.voxel_size, self.origin = get_geometry(
                    im.affine, im.voxel_size, im.origin)

        self.set_geometry()

    def crop_about_point(self, point=None, xlim=None, ylim=None, zlim=None):
        """
        Crop the image to a given x, y, z range in mm about a point.

        If any range is None, the image will not be cropped in that direction.

        **Parameters:**

        point : tuple, default=None
            (x, y, z) coordinates about which cropping is to be performed.
            If None, the point (0, 0, 0) is used, and the result will be
            the same as when calling the Image.crop() method.

        xlim : tuple, default=None
            Lower and upper bounds relative to reference point of cropping
            along x-axis.  If None, cropping is not performed along this axis.

        ylim : tuple, default=None
            Lower and upper bounds relative to reference point of cropping
            along y-axis.  If None, cropping is not performed along this axis.

        zlim : tuple, default=None
            Lower and upper bounds relative to reference point of cropping
            along z-axis.  If None, cropping is not performed along this axis.
        """
        if point is None:
            point = (0, 0, 0)

        lims = []
        for idx, lim in enumerate([xlim, ylim, zlim]):
            lims.append(None if lim is None else
                    [lim[0] + point[idx], lim[1] + point[idx]])

        self.crop(*lims)

    def crop_by_amounts(self, dx=None, dy=None, dz=None):
        """
        Crop image by the amounts dx, dy, dz in mm.

        This method calls the function skrt.image.crop_by_amounts(), with
        self passed as object for cropping.

        The amount of cropping along each direction should be one of:
        - float : the image is cropped by this amount on both sides;
        - two-element tuple: the image is cropped on the sides of lower
          and higher values by the amounts specified;
        - None : no cropping is performed.

        For more details, see documentation of skrt.image.crop_by_amounts().
        """
        crop_by_amounts(self, dx, dy, dz)

    def crop_to_roi(self, roi, crop_margins=None, crop_about_centre=False,
                    method=None):
        """
        Crop image to region defined by an ROI or StructureSet, plus margins.

        **Parameters:**

        rois : skrt.structures.ROI/skirt.structures.StructureSet
            ROI or StructureSet to which image will be cropped.

        crop_margins : float/tuple, default=None
            Float or three-element tuple specifying the margins, in mm,
            to be added to extents or centre point of ROI or StructureSet.  If
            a float, minus and plus the value specified are added to lower
            and upper extents respectively along each axis.  If a
            three-element tuple, elements are taken to specify margins in
            the order (x, y, z).  Elements can be either floats (minus and
            plus the value added respectively to lower and upper extents)
            or two-element tuples (elements 0 and 1 added respectively
            to lower and upper extents).

        crop_about_centre : bool, default=False
            If True, image is cropped to the centre point of ROI or
            StructureSet plus margins.  If False, image is cropped to 
            the extents of ROI or StructureSet plus margins.

        method : str, default=None
            Method to use for calculating extent of <roi> region. Can be: 

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the 
                  binary mask.
                * None: use the method set in self.default_geom_method.
        """
        if crop_about_centre:
            self.crop_about_point(roi.get_centre(method=method),
                                  *checked_crop_limits(crop_margins))
        else:
            self.crop(*roi.get_crop_limits(
                crop_margins=crop_margins, method=method))

    def crop_to_image(self, image, alignment=None):
        """
        Crop to extents of another image, optionally after image alignment.

        **Parameters:**

        image : skrt.image.Image
            Image to whose extents to perform cropping.

        alignment : tuple/dict/str, default=None
            Strategy to be used for image alignment prior to cropping.
            For further details, see documentation of
            skrt.image.get_alignment_translation().
        """
        # Calculate any translation to be applied prior to cropping.
        translation = self.get_alignment_translation(image, alignment)

        # Apply translation to image origin.
        if translation is not None:
            translation = np.array(translation)
            self.translate_origin(translation)

        # Perform cropping.
        self.crop(*image.get_extents())

        # Apply reverse translation to origin of cropped image.
        if translation is not None:
            self.translate_origin(-translation)

    def map_hu(self, mapping='kv_to_mv'):
        '''
        Map radiodensities in Hounsfield units according to mapping function.

        **Parameter:**

        mapping - str, default='kv_to_mv'
            Identifier of mapping function to be used.  Currently only
            the VoxTox mapping from kV CT scan to MV CT scan is implemented.
        '''
        if 'kv_to_mv' == mapping:
            self.data = (np.vectorize(kv_to_mv, otypes=[np.float64])
                    (self.get_data()))
        else:
            print(f'Mapping function {mapping} not known.')

    def get_sinogram(self, force=False, verbose=False):
        '''
        Retrieve image where each slice corresponds to a sinogram.

        A slice sinogram is obtain through application of a Radon
        transform to the original image slice.

        **Parameters:**

        force : bool, default=False
            The first time that sinogram image is created, it is
            stored as self.sinogram.  It is recreated only if force
            is set to True.

        verbose : bool, default=False
            Print information on progress in sinogram creation.
        '''
        self.load()

        if (self.sinogram is None) or force:
            if verbose:
                print('Creating sinogram for each image slice:')

            # Value to subtract to have only positive intensities.
            vmin = min(self.get_data().min(), 0)

            sinogram_stack = []
            nz = self.get_n_voxels()[2]
            # Apply Radon transform slice by slice
            for iz in range(nz):
                if verbose:
                    print(f'    ...slice {iz + 1:3d} of {nz:3d}')
                im_slice = self.get_data()[:, :, iz] - vmin
                sinogram_slice = skimage.transform.radon(im_slice, circle=False)
                sinogram_stack.append(sinogram_slice)
            self.sinogram = Image(np.stack(sinogram_stack, axis=-1))

        return self.sinogram
            
    def add_sinogram_noise(self, phi0=60000, eta=185000, verbose=False):

        '''
        Add Poisson fluctuations at level of sinogram.

        For a kV CT scan mapped to the radiodensity scale of an MV CT scan
        (self.map_hu('kv_to_mv'), this function adds intensity fluctuations
        to reproduce better the fluctuations in an MV CT scan.

        The procedure for adding fluctuations was originally devised and
        implemented by M.Z. Wilson.

        **Parameters:**

        phi0 : int/float, default=60000
            Notional photon flux used in image creation.

        eta : int/float, default=185000
            Notional constant of proportionality linking radiodensity
            and line integrals measuring attenuation along photon paths.

        verbose : bool, default=False
            Print information on progress in noise addition
        '''
        self.load()

        self.get_sinogram(verbose=verbose)

        if verbose:
            print('Adding sinogram-level fluctuations for each image slice:')

        # Value to subtract to have only positive intensities.
        vmin = min(self.get_data().min(), 0)

        nz = self.get_n_voxels()[2]
        
        # Loop over slices
        for iz in range(nz):
            if verbose:
                print(f'    ...slice {iz + 1:3d} of {nz:3d}')
            sinogram_slice = self.get_sinogram().get_data()[:, :, iz].copy()
            for idx, value in np.ndenumerate(sinogram_slice):
                if sinogram_slice[idx] > 0:
                    # Sample notional photon flux from a poisson distribution
                    flux = np.random.poisson(
                            phi0 * np.exp(-sinogram_slice[idx] / eta))
                    if flux > 0:
                        # Recalculate the sinogram value from the new flux value
                        sinogram_slice[idx] = -eta * np.log(flux / phi0)

            # Apply inverse Radon transform through filtered back propagation
            self.data[:, :, iz] = skimage.transform.iradon(sinogram_slice,
                    circle=False, filter_name='hann') + vmin

    def assign_intensity_to_rois(self, rois=None, intensity=0):
        '''
        Assign intensity value to image regions corresponding to ROIs.

        **Parameters:**

        rois : list of skrt.structures.ROI objects, default=None
            ROIs for which voxels are to be assigned an intensity value.

        intensity : int/float, default=0
            Intensity value to be assigned to ROI voxels.
        '''

        self.load()

        for roi in rois:
            self.data[roi.get_mask()] = intensity

    def remove_bolus(self, structure_set=None, bolus_names=None,
            intensity=None):
        '''
        Attempt to remove bolus from image.

        In treatment planning, ROIs labelled as bolus may be defined on
        the skin, and are assigned a radiodensity of water (zero) to help guide
        the treatment optimiser.  The original image is approximately
        recovered by overwriting with the radiodensity of air.

        **Parameters:**
        structure_set : skrt.structures.StructureSet
            Structure set to search for ROIs labelled as bolus.  If None,
            the image's earliest associated structure set is used.

        bolus_names : list, default=None
            List of names, optionally including wildcards, with which
            bolus may be labelled.  If None, use
            skrt.image._default_bolus_names.

       intensity : int/float, default=None
            Intensity value to be assigned to voxels of ROI labelled as bolus.
            If None, use image's minimum value.
        '''

        if structure_set is None:
            if self.structure_sets:
                structure_set = self.structure_sets[0]
            else:
                return

        self.load()

        if bolus_names is None:
            bolus_names = _default_bolus_names

        if intensity is None:
            intensity = self.get_data().min()
        bolus_rois = []
        for bolus_name in bolus_names:
            bolus_rois.extend(structure_set.get_rois_wildcard(bolus_name))
        self.assign_intensity_to_rois(bolus_rois, intensity)

    def apply_banding(self, bands=None):
        '''
        Apply banding to image data.

        **Parameter:**

        bands - dict, default=None
            Dictionary of value bandings to be applied to image data.
            Keys specify band limits, and values indicte the values
            to be assigned.  For example:

            - bands{-100: -1024, 100: 0, 1e10: 1024}

            will result in the following banding:
 
            - value <= -100 => -1024;
            - -100 < value <= 100 => 0;
            - 100 < value <= 1e10 => 1024.
        '''

        if bands is None:
            return

        # Retreive data, and initialise banded values to lowest band.
        image_data = self.get_data()
        values = sorted(list(bands.keys()))
        banded_data = np.ones(image_data.shape) * bands[values[0]]

        # Set values between values[i-1] and values[i] to bands[values[i]],
        for i in range(1, len(values)):
            v1 = values[i - 1]
            v2 = values[i]
            v_band = bands[v2]
            banded_data[(image_data > v1) & (image_data <= v2)] = v_band

        self.data = banded_data

    def apply_selective_banding(self, bands=None):
        '''
        Apply banding to selected intensity ranges of image data.

        This is one of two skrt.image.Image methods to perform banding:

        - apply_banding():
          assign all intensity values to bands;
        - apply_selective_banding():
          assign only selected intensity values to bands.

        Both methods accept as input a dictionary specifying bands, but
        the dictionary format for the two methods is different.

        **Parameter:**

        bands - dict, default=None
            Dictionary of value bandings to be applied to image data.
            Keys are floats specifying intensities to be assigned, and
            values are two-element tuples indicating lower and upper
            band limits.  If the first element is None, no lower limit
            is applied; if the second element is None, no upper limit
            is applied.  For example:

            - bands{-1024: (None, -100), 1024: (100, None}

            will result in the following banding:
 
            - value <= -100 => -1024;
            - -100 < value <= 100 => original values retained;
            - 100 < value => 1024.
        '''

        if not bands:
            return

        # Retreive data, and create copy on which to apply banding.
        image_data = self.get_data()
        banded_data = image_data.copy()

        # Apply banding.
        for v_band, values in sorted(bands.items()):
            v1 = values[0] if values[0] is not None else image_data.min() - 1
            v2 = values[1] if values[1] is not None else image_data.max() + 1
            banded_data[(image_data > v1) & (image_data <= v2)] = v_band

        self.data = banded_data

    def get_masked_image(self, mask=None, mask_threshold=0.5,
            invert_mask=False):
        '''
        Return image after application of mask.

        mask : Image/list/ROI/str/StructureSet, default=None
            Image object representing a mask or a source from which
            an Image object can be initialised.  In addition to the
            sources accepted by the Image constructor, the source
            may be an ROI, a list of ROIs or a StructureSet.  In the
            latter cases, the mask image is derived from the ROI mask(s).

        mask_threshold : float, default=0.5
            Threshold for mask data.  Values above and below this value are
            set to True and False respectively.  Taken into account only
            if the mask image has non-boolean data.

        invert_mask : bool, default=False
            If True, the mask is inverted before being applied.
        '''

        # Try to ensure that mask is an Image object,
        # has boolean data, and matches the size of self.
        mask = get_mask(mask, mask_threshold, self)

        # Clone self and apply mask.
        self.get_data()
        masked_image = self.clone()
        if invert_mask:
            masked_image.data = np.ma.masked_where(mask.data, self.data)
        else:
            masked_image.data = np.ma.masked_where(~mask.data, self.data)

        return masked_image

    def get_comparison(
            self, other, metrics=None, name=None, name_as_index=True,
            nice_columns=False, decimal_places=None,
            base=2, bins=100, xyrange=None):
        """
        Return a pandas DataFrame comparing this image with another.

        If this image doesn't have the same shape and geometry as
        the other image, comparison metrics are evaluated for a clone
        of this image, resized to match the other.

        **Parameters:**
        
        other : skrt.image.Image
            Other image, with which to compare this image.

        metrics : list, default=None
            List of metrics to evaluate.  Available metrics:
                
            Calculated in Image.get_mutual_information():

                * "mutual_information";
                * "normalised_mutual_information";
                * "information_quality_ratio";
                * "rajski_distance".

            Calculated in Image.get_quality():

                * "relative_structural_content";
                * "fidelity";
                * "correlation_quality".

            If None, defaults to ["mutual_information"]

        name : str, default=None
            Name identifying comparison.  If null, the name is
            constructed from the titles of the two images compared,
            as "title1_vs_title2".

        name_as_index : bool, default=True
            If True, the index column of the pandas DataFrame will 
            contain the name of this comparison; otherwise, the name will
            appear in a column labelled "images".

        nice_columns : bool, default=False
            If False, column labels will be the same as the input metric names;
            if True, the names will be capitalized and underscores will be 
            replaced with spaces.

        decimal_places : int, default=None
            Number of decimal places to keep for each metric. If None, full
            precision will be used.

        base : int/None, default=2
            Base to use when taking logarithms, in calculations of
            mutual information and variants.  If None, use base e.

        bins : int/list, default=50
            Numbers of bins to use when histogramming grey-level joint
            probabilities for self and other, in calculations of
            mutual information and variants.  This is passed as
            the bins parameter of numpy.histogram2d:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

        xyrange : list, default=None
            Upper and lower of each axis when histogramming grey-level
            joint probabilities for self and image, in calculations of
            mutual informatio and variants.  This is passed as
            the range parameter of numpy.histogram2d:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
        """
        # Ensure that images for comparison have the same geometry.
        if self.has_same_geometry(other):
            im1 = self
        else:
            im1 = self.clone()
            im1.match_size(other)
        im2 = other

        # Ensure that comparison name is defined.
        if not name:
            title1 = im1.title or "image"
            title2 = im2.title or "other"
            name = f"{title1}_vs_{title2}"

        # Set default metric.
        metrics = metrics or ["mutual_information"]

        # Check that specified metrics are recognised,
        # and identify quality metrics.
        quality_metrics = []
        for metric in metrics:
            if metric not in get_image_comparison_metrics():
                raise RuntimeError(f"Metric {metric} not recognised by "
                                   "Image.get_comparison()")
            if metric in get_quality_metrics():
                quality_metrics.append(metric)

        # Compute metric scores.
        quality_scores = im1.get_quality(im2, quality_metrics)
        scores = {}
        for metric in metrics:
            # Store scores for quality metrics.
            if metric in get_quality_metrics():
                scores[metric] = quality_scores[metric]
            # Store scores for mutual information and variants.
            else:
                scores[metric] = im1.get_mutual_information(
                        im2, base=base, bins=bins, xyrange=xyrange,
                        variant=metric)
            if decimal_places is not None:
                scores[metric] = round(scores[metric], decimal_places)

        # Convert to pandas DataFrame.
        df = pd.DataFrame(scores, index=[name])

        # Turn name into a regular column if requested
        if not name_as_index:
            df = df.reset_index().rename({"index": "images"}, axis=1)

        # Capitalize column names and remove underscores if requested.
        if nice_columns:
            df.columns = [col.capitalize().replace("_", " ")
                          if len(col) > 3 else col.upper()
                          for col in df.columns]
        return df

    def get_mutual_information(self, image, base=2, bins=100, xyrange=None,
                               variant="mutual_information"):
        """
        For this and another image, calculate mutual information or a variant.

        The two images considered must have the same shape.

        **Parameters:**

        image : skrt.image.Image
            Image with respect to which mutual information is calculated.

        base : int/None, default=2
            Base to use when taking logarithms.  If None, use base e.

        bins : int/list, default=50
            Numbers of bins to use when histogramming grey-level joint
            probabilities for self and image.  This is passed as
            the bins parameter of numpy.histogram2d:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

        xyrange : list, default=None
            Upper and lower of each axis when histogramming grey-level
            joint probabilities for self and image.  This is passed as
            the range parameter of numpy.histogram2d:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

        variant : str, default=None
            Variant of mutual information to be returned.

            - "mi", "mutual_information":
              return mutual information, as introduced in:
              https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

            - "nmi", "normalised_mutual_information":
              return normalised mutual information (range 1 to 2) as defined in:
              https://doi.org/10.1117/12.310835

            - "iqr", "information_quality_ratio":
              return information quality ratio (range 0 to 1) as defined in:
              https://doi.org/10.1016/j.chemolab.2016.11.012
              => information_quality_ratio = normalised_mutual_information - 1

            - "rajski", "rajski_distance":
              return Rajski distance (range 1 to 0) as defined in:
              https://doi.org/10.1016/S0019-9958(61)80055-7
              => rajski_distance = 2 - normalised_mutual_information

            - Any other value, return None
        """
        # Check that requested variant is recognised.
        if not variant in get_mi_metrics():
            return

        # Check base for taking logarithms.
        if base is None:
            base = math.e

        # Create 2d histogram of voxel-by-voxel grey-level values,
        # comparing images.
        hist2d, xedges, yedges = np.histogram2d(
                self.get_data().ravel(), image.get_data().ravel(),
                bins, xyrange)

        # Convert numbers of entries to joint probabilities.
        pxy = hist2d / float(np.sum(hist2d))

        # Obtain marginal probabilities.
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Calculate entropies.
        small_number = 1.e-6
        hx = entropy(px, base)
        hy = entropy(py, base)
        hxy = entropy(np.reshape(pxy, -1), base)

        mi = hx + hy - hxy
        # Return mutual information or a variant.
        if variant in ["mi", "mutual_information"]:
            return mi
        if variant in ["nmi", "normalised_mutual_information"]:
            return 1 + mi / (hxy or small_number)
        elif variant in ["iqr", "information_quality_ratio"]:
            return mi / (hxy or small_number)
        if variant in ["rajski", "rajski_distance"]:
            return 1 - mi / (hxy or small_number)

    def get_quality(self, image, metrics=None):
        """
        Evaluate quality of this image relative to another.

        The metrics considered are:

        - relative structural content;
        - fidelity;
        - correlation quality.

        These are defined in:

        - https://doi.org/10.1364/JOSA.46.000740
        - https://doi.org/10.1080/713826248

        The three metrics should each have a value between 0 and 1,
        and are related by:

        correlation quality = 0.5 * (relative structural content + fidelity).

        Quality scores are returned in a dictionary, with a key corresponding
        to each metric: "relative_structural_content", "fidelity",
        "correlation_quality".  If a metric isn't evaluated, the value
        returned for it is None.

        **Parameters:**

        image : skrt.image.Image
            Image with respect to which quality metrics are to be calculated.

        metrics: list, default=None
            List of strings specifying quality metrics to be evaluated.
            If None, all defined quality metrics are evaluated.
        """
        # Define metrics to be evaluated.
        all_metrics = ["relative_structural_content",
                       "fidelity", "correlation_quality"]
        if metrics is None:
            metrics = all_metrics
        else:
            if not skrt.core.is_list(metrics):
                metrics = [metrics]
            metrics = [metric for metric in metrics if metric in all_metrics]

        # Initialise dictionary of metric values.
        quality = {metric: None for metric in all_metrics}

        if metrics:
            # Recale intensity values, so that the minimum is 0.
            v_min = min(self.get_min(), image.get_min())
            data1 = self.get_data() - v_min
            data2 = image.get_data() - v_min

            # If intensity values have non-zero sum, normalise to 1.
            if data1.sum():
                data1 = data1 / data1.sum()
            if data2.sum():
                data2 = data2 / data2.sum()

            # Evaluate quality metrics.
            denominator = (data2**2).sum()
            metric = "relative_structural_content"
            if metric in metrics:
                quality[metric] = (data1**2).sum() / denominator
            metric = "fidelity"
            if metric in metrics:
                quality[metric] = (
                        1 - (((data2 - data1)**2).sum() / denominator))
            metric = "correlation_quality"
            if metric in metrics:
                quality[metric] = (data2 * data1).sum() / denominator

        return quality

    def get_relative_structural_content(self, image):
        """
        Calculate structural content of this image relative to another.

        Uses method get_quality().

        **Parameter:**

        image : skrt.image.Image
            Image with respect to which relative structural content is
            to be evaluated
        """
        metric = "relative_structural_content"
        return self.get_quality(image, metrics=[metric])[metric]

    def get_fidelity(self, image):
        """
        Calculate fidelity with which this image matches another.

        Uses method get_quality().

        **Parameter:**

        image : skrt.image.Image
            Image with respect to which fidelity is to be calculated.
            to be evaluated
        """
        metric = "fidelity"
        return self.get_quality(image, metrics=[metric])[metric]

    def get_correlation_quality(self, image):
        """
        Calculate quality of correlation between this image and another.

        Uses method get_quality().

        **Parameter:**

        image : skrt.image.Image
            Image with respect to which correlation quality is to be calculated.
            to be evaluated
        """
        metric = "correlation_quality"
        return self.get_quality(image, metrics=[metric])[metric]


class ImageComparison(Image):
    """Plot comparisons of two images and calculate comparison metrics."""

    def __init__(self, im1, im2, plot_type="overlay", title=None, **kwargs):

        # Load images
        self.ims = []
        for im in [im1, im2]:
            if issubclass(type(im), Image):
                self.ims.append(im)
            else:
                self.ims.append(Image(im, **kwargs))

        if plot_type is not None:
            self.plot_type = plot_type
        else:
            self.plot_type = "overlay"

        self.override_title = title
        self.gs = None
        self.loaded = False

    def load(self, force=False):
        """
        Load associated images, and set array limits in [x, y, z] directions.

        **Parameter:**
        
        force : bool, default=True
            If True, associated images will be reloaded from source, even if
            previously loaded.
        """
        if (self.loaded and not force) or (not self.ims):
            return

        for im in self.ims:
            im.load(force=force)

        self.lims = self.ims[0].lims

        self.loaded = True

    def view(self, **kwargs):
        """View self with BetterViewer."""

        from skrt.better_viewer import BetterViewer
        kwargs.setdefault("comparison", True)

        BetterViewer(self.ims, **kwargs)

    def plot(
        self,
        view=None,
        sl=None,
        idx=None,
        pos=None,
        scale_in_mm=True,
        invert=False,
        ax=None,
        mpl_kwargs=None,
        show=True,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        plot_type=None,
        cb_splits=8,
        overlay_opacity=0.5,
        overlay_legend=False,
        overlay_legend_bbox_to_anchor=None,
        overlay_legend_loc=None,
        colorbar=False,
        colorbar_label=None,
        show_mse=False,
        dta_tolerance=None,
        dta_crit=None,
        diff_crit=None,
        use_cached_slices=False,
        **kwargs
    ):

        # Use default plot_type attribute if no type given
        if plot_type is None:
            plot_type = self.plot_type

        # If view not specified, set based on image orientation
        if view is None:
            view = self.get_orientation_view()

        # By default, use comparison type as title
        if self.override_title is None:
            self.title = plot_type[0].upper() + plot_type[1:]
        else:
            self.title = self.override_title

        # Set slice inputs to lists of two items
        sl = skrt.core.to_list(sl, 2, False)
        idx = skrt.core.to_list(idx, 2, False)
        pos = skrt.core.to_list(pos, 2, False)

        # Get image slices
        idx[0] = self.ims[0].get_idx(view, sl=sl[0], idx=idx[0], pos=pos[0])
        idx[1] = self.ims[1].get_idx(view, sl=sl[1], idx=idx[1], pos=pos[1])
        self.set_slices(view, idx=idx, use_cached_slices=use_cached_slices)

        # Set up axes
        self.set_ax(view, ax=ax, gs=self.gs, figsize=figsize, zoom=zoom)
        self.mpl_kwargs = self.ims[0].get_mpl_kwargs(view, mpl_kwargs)
        self.cmap = copy.copy(matplotlib.cm.get_cmap(self.mpl_kwargs.pop("cmap")))

        # Make plot
        if plot_type in ["chequerboard", "cb"]:
            mesh = self._plot_chequerboard(view, invert, cb_splits)
        elif plot_type == "overlay":
            mesh = self._plot_overlay(
                view, invert, overlay_opacity, overlay_legend, overlay_legend_bbox_to_anchor, overlay_legend_loc
            )
        elif plot_type in ["difference", "diff"]:
            mesh = self._plot_difference(invert)
        elif plot_type == "absolute difference":
            mesh = self._plot_difference(invert, ab=True)
        elif plot_type in ["distance to agreement", "dta", "DTA"]:
            mesh = self._plot_dta(view, idx[0], dta_tolerance)
        elif plot_type == "gamma index":
            mesh = self._plot_gamma(view, idx[0], invert, dta_crit, diff_crit)
        elif plot_type == "image 1":
            self.title = self.ims[0].title
            kwargs = self.ims[0].get_mpl_kwargs(view)
            kwargs["vmin"] = self.mpl_kwargs["vmin"]
            kwargs["vmax"] = self.mpl_kwargs["vmax"]
            mesh = self.ax.imshow(
                self.slices[0],
                cmap=self.cmap,
                **kwargs
            )
        elif plot_type == "image 2":
            self.title = self.ims[1].title
            kwargs = self.ims[1].get_mpl_kwargs(view)
            kwargs["vmin"] = self.mpl_kwargs["vmin"]
            kwargs["vmax"] = self.mpl_kwargs["vmax"]
            mesh = self.ax.imshow(
                self.slices[1],
                cmap=self.cmap,
                **kwargs
            )
        else:
            print("Unrecognised plotting option:", plot_type)
            return

        # Draw colorbar
        if colorbar:
            clb_label = colorbar_label if colorbar_label is not None \
                    else self.ims[0]._default_colorbar_label
            if plot_type in ["difference", "absolute difference"]:
                clb_label += " difference"
            elif plot_type == "distance to agreement":
                clb_label = "Distance (mm)"
            elif plot_type == "gamma index":
                clb_label = "Gamma index"

            clb = self.fig.colorbar(mesh, ax=self.ax, label=clb_label)
            clb.solids.set_edgecolor("face")

        # Adjust axes
        self.label_ax(view, idx, title=self.title, **kwargs)
        self.zoom_ax(view, zoom, zoom_centre)

        # Annotate with mean squared error
        if show_mse:
            mse = np.sqrt(((self.slices[1] - self.slices[0]) ** 2).mean())
            mse_str = f"Mean sq. error = {mse:.2f}"
            if matplotlib.colors.is_color_like(show_mse):
                col = show_mse
            else:
                col = "white"
            self.ax.annotate(
                mse_str,
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                color=col,
                fontsize="large",
            )

        if show:
            plt.show()

    def set_slices(self, view, sl=None, idx=None, pos=None, 
                   use_cached_slices=False):
        """Get slice of each image and set to self.slices."""

        sl = skrt.core.to_list(sl, 2, False)
        idx = skrt.core.to_list(idx, 2, False)
        pos = skrt.core.to_list(pos, 2, False)
        self.slices = [
            im.get_slice(view, sl=sl[i], pos=pos[i], idx=idx[i],
                         force=(not use_cached_slices))
            for i, im in enumerate(self.ims)]

    def _plot_chequerboard(
        self,
        view,
        invert=False,
        cb_splits=8,
    ):

        # Get masked image
        i1 = int(invert)
        i2 = 1 - i1
        size_x = int(np.ceil(self.slices[i2].shape[0] / cb_splits))
        size_y = int(np.ceil(self.slices[i2].shape[1] / cb_splits))
        cb_mask = np.kron(
            [[1, 0] * cb_splits, [0, 1] * cb_splits] * cb_splits, np.ones((size_x, size_y))
        )
        cb_mask = cb_mask[: self.slices[i2].shape[0], : self.slices[i2].shape[1]]
        to_show = {
            i1: self.slices[i1],
            i2: np.ma.masked_where(cb_mask < 0.5, self.slices[i2]),
        }

        # Ensure that overlay masked sections are transparent
        # (otherwise underlay image won't be visible).
        self.cmap.set_bad(alpha=0)

        # Plot
        for i in [i1, i2]:
            kwargs = self.ims[i].get_mpl_kwargs(view)
            kwargs["vmin"] = self.mpl_kwargs["vmin"]
            kwargs["vmax"] = self.mpl_kwargs["vmax"]
            kwargs["cmap"] = self.cmap
            mesh = self.ax.imshow(
                to_show[i],
                **kwargs
            )

        return mesh

    def _plot_overlay(
        self,
        view,
        invert=False,
        opacity=0.5,
        legend=False,
        legend_bbox_to_anchor=None,
        legend_loc="auto"
    ):

        order = [0, 1] if not invert else [1, 0]
        cmaps = ["Reds", "Blues"]
        alphas = [1, opacity]
        self.ax.set_facecolor("w")
        handles = []
        for n, i in enumerate(order):

            # Show image
            kwargs = self.ims[i].get_mpl_kwargs(view)
            kwargs["vmin"] = self.mpl_kwargs["vmin"]
            kwargs["vmax"] = self.mpl_kwargs["vmax"]
            kwargs["cmap"] = cmaps[n]
            kwargs["alpha"] = alphas[n]
            mesh = self.ax.imshow(
                self.slices[i],
                **kwargs
            )

            # Make handle for legend
            if legend:
                patch_color = cmaps[n].lower()[:-1]
                alpha = 1 - opacity if alphas[n] == 1 else opacity
                title = self.ims[i].title
                name = pathlib.Path(self.ims[i].path).name
                if title == name:
                    title = name.split(".")[0].capitalize()
                handles.append(
                    mpatches.Patch(
                        color=patch_color, alpha=alpha, label=title
                    )
                )

        # Draw legend
        if handles:
            self.ax.legend(
                    handles=handles, bbox_to_anchor=legend_bbox_to_anchor,
                    loc=legend_loc, facecolor="white",
                    framealpha=1
                )

    def get_difference(self, view=None, sl=None, idx=None, pos=None, 
                       invert=False, ab=False, reset_slices=True):
        """Get array containing difference between two Images."""

        # No view/position/index/slice given: use 3D arrays
        if reset_slices and (view is None and sl is None and pos is None and idx is None):
            diff = self.ims[1].get_data() - self.ims[0].get_data()
        else:
            if view is None:
                view = "x-y"
            if reset_slices:
                self.set_slices(view, sl, idx, pos)
            diff = (
                self.slices[1] - self.slices[0]
                if not invert
                else self.slices[0] - self.slices[1]
            )
        if ab:
            diff = np.absolute(diff)
        return diff

    def _plot_difference(self, invert=False, ab=False):
        """Produce a difference plot."""

        diff = self.get_difference(reset_slices=False, ab=ab, invert=invert)
        if ab:
            min_diff = np.min(diff)
            self.mpl_kwargs["vmin"] = 0
        else:
            self.mpl_kwargs["vmin"] = np.min(diff)
        self.mpl_kwargs["vmax"] = np.max(diff)
        return self.ax.imshow(
            diff,
            cmap=self.cmap,
            **self.mpl_kwargs,
        )

    def _plot_dta(self, view, idx, tolerance=5):
        """Produce a distance-to-agreement plot."""

        dta = self.get_dta(view, idx=idx, reset_slices=False, 
                           tolerance=tolerance)
        return self.ax.imshow(
            dta,
            cmap="viridis",
            interpolation=None,
            **self.mpl_kwargs,
        )

    def _plot_gamma(self, view, idx, invert=False, dta_crit=None, 
                    diff_crit=None):
        """Produce a distance-to-agreement plot."""

        gamma = self.get_gamma(view, idx=idx, invert=invert, dta_crit=dta_crit, 
                               diff_crit=diff_crit, reset_slices=False)
        return self.ax.imshow(
            gamma,
            cmap="viridis",
            interpolation=None,
            **self.mpl_kwargs,
        )

    def get_dta(self, view="x-y", sl=None, idx=None, pos=None, tolerance=None, 
                reset_slices=True):
        """Compute distance to agreement array on current slice."""

        if not hasattr(self, "dta"):
            self.dta = {}
        if view not in self.dta:
            self.dta[view] = {}
        idx = self.ims[0].get_idx(view, sl=sl, idx=idx, pos=pos)

        if sl not in self.dta[view]:

            x_ax, y_ax = _plot_axes[view]
            vx = abs(self.ims[0].get_voxel_size()[x_ax])
            vy = abs(self.ims[0].get_voxel_size()[y_ax])

            if reset_slices:
                self.set_slices(view, sl, idx, pos)
            im1, im2 = self.slices
            if tolerance is None:
                tolerance = 5
            abs_diff = np.absolute(im2 - im1)
            agree = np.transpose(np.where(abs_diff <= tolerance))
            disagree = np.transpose(np.where(abs_diff > tolerance))
            dta = np.zeros(abs_diff.shape)
            for coords in disagree:
                dta_vec = agree - coords
                dta_val = np.sqrt(
                    vy * dta_vec[:, 0] ** 2 + vx * dta_vec[:, 1] ** 2
                ).min()
                dta[coords[0], coords[1]] = dta_val

            self.dta[view][idx] = dta

        return self.dta[view][idx]

    def get_gamma(self, view="x-y", sl=None, pos=None, idx=None, invert=False, 
                  dta_crit=None, diff_crit=None, reset_slices=True):
        """Get gamma index on current slice."""

        if reset_slices:
            self.set_slices(view, sl, idx, pos)
        im1, im2 = self.slices
        if invert:
            im1, im2 = im2, im1

        if dta_crit is None:
            dta_crit = 1
        if diff_crit is None:
            diff_crit = 15

        diff = im2 - im1
        dta = self.get_dta(view, sl=sl, pos=pos, idx=idx, reset_slices=False)
        return np.sqrt((dta / dta_crit) ** 2 + (diff / diff_crit) ** 2)

    def get_plot_aspect_ratio(self, *args, **kwargs):
        """Get relative width of first image."""

        return self.ims[0].get_plot_aspect_ratio(*args, **kwargs)


def load_nifti(path):
    """Load an image from a nifti file."""

    try:
        nii = nibabel.load(path)
        data = nii.get_fdata()
        affine = nii.affine
        if data.ndim > 3:
            data = np.squeeze(data)
        return data, affine

    except FileNotFoundError:
        print(f"Warning: file {path} not found! Could not load nifti.")
        return None, None

    except nibabel.filebasedimages.ImageFileError:
        return None, None

def load_rgb(path, rgb_weights=(0.299, 0.587, 0.114),
             rgb_rescale_slope=100, rgb_rescale_intercept=0):
    """
    Load an rgb image with the Python Imaging Library (PIL),
    and convert to grey levels.

    **Parameter:**

    rgb_weights : tuple, default=(0.299, 0.587, 0.114)
        Three-element tuple specifying the weights to red (R),
        green (G), blue (B) channels to arrive at a grey level:

        L = R * rgb[0] + G * rgb[1] + B * rgb[2]

        The default is the same as the PIL default:
        https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

    rgb_rescale_slope: float, default=100
        Factor by which to multiply grey level in rescaling.

    rgb_rescale_intercept: float, default=0
        Constant to add to multiplied grey level in rescaling.
    """
    try:
        img_rgb = plt.imread(path)
    except:
        return

    return (np.dot(img_rgb[...,:3], rgb_weights)
            * rgb_rescale_slope + rgb_rescale_intercept)

def get_dicom_paths(path):
    """Get list of dicom files correpsonding to a single dicom image. 

    **Parameters**:

    path : str
        Path to either a single dicom file, or a directory containing multiple
        dicom files.

        If path is a directory, a list of dicom files within that directory will
        be returned.

        If path is a single file, that file will be opened and its 
        ImagesInAcquisition property will be checked. 

            - If ImagesInAcquisition == 1, a list containing the single input 
            path will be returned. 
            - Otherwise, a list containing all the dicom files in the same 
            directory as the input file will be returned.

    **Returns**:

    paths : list of strs
        List of strings, each pointing to a single dicom file. If no valid
        dicom files are found, returns an empty list.
    """

    path = skrt.core.fullpath(path)
    paths = []

    # Case where path points to a single file
    if os.path.isfile(path):
        try:
            ds = pydicom.dcmread(path, force=True)
        except pydicom.errors.InvalidDicomError:
            return paths

        # Return empty list if this is not a valid dicom file
        if not hasattr(ds, "SOPClassUID"):
            return paths

        # Assign TransferSyntaxUID if missing
        if not hasattr(ds, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Check whether there are multiple files for this image
        if (ds.get("ImagesInAcquisition", None) == 1
                or hasattr(ds, "NumberOfFrames")):
            paths = [path]

    # Case where there are multiple dicom files for this image
    if not paths:
        if os.path.isdir(path):
            dirname = path
        else:
            dirname = os.path.dirname(path)

        for filename in sorted(os.listdir(dirname)):
            p = os.path.join(dirname, filename)
            if not os.path.isdir(p):
                ds = pydicom.dcmread(p, force=True)
                if hasattr(ds, "SOPClassUID"):
                    paths.append(p)

        # Ensure user-specified file is loaded first
        if path in paths:
            paths.insert(0, paths.pop(paths.index(path)))

    return paths


def load_dicom(paths, debug=False):
    """Load a dicom image from one or more dicom files.

    **Parameters**:

    paths : str/list
        Path to a single dicom file, path to a directory containing multiple
        dicom files, or list of paths to dicom files. If path points to a
        single file that is found to be part of a series of multiple dicom
        files corresponding to one image,  the image will be loaded from
        all dicom files in the same directory as the first that match its
        StudyInstanceUID and SeriesNumber.

    **Returns**:

    data : np.ndarray
        Numpy array containing the voxel intensities for the image.

    affine : np.ndarray
        3x3 numpy array containing the affine matrix.

    window_centre : float
        Default intensity window centre for viewing this image.

    window_width : float
        Default intensity window width for viewing this image.

    ds : pydicom.FileDataset
        Dicom dataset corresponding to the last file loaded.

    z_paths : dict / None
        Dictionary mapping z slice positions in mm to the file corresponding to 
        that slice. If image was loaded from a single file, this will be None.

    z_instance_numbers : dict / None
        Dictionary mapping z slice positions in mm to the instance number of
        that slice. If image was loaded from a single file, this will be None.
    """

    # Get list of paths corresponding to this image
    paths = paths if isinstance(paths, list) else get_dicom_paths(paths)
    if not len(paths):
        return tuple([None] * 6)

    # Load image array and pydicom FileDataset object from file(s)
    if len(paths) > 1:
        data, affine, ds, z_paths, z_instance_numbers = load_dicom_many_files(
                paths)
    else:
        data, affine, ds = load_dicom_single_file(paths[0])
        z_paths = None
        z_instance_numbers = None

    # Load other properties
    window_centre, window_width = get_dicom_window(ds)

    # Rescale the data
    data = rescale_dicom_data(ds, data)

    # Take into account photometric interpretation.
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        vmin = data.min()
        vmax = data.max()
        data = -data + vmin + vmax
        if window_centre:
            window_centre = -window_centre + vmin + vmax

    return (data, affine, window_centre, window_width, ds,
            z_paths, z_instance_numbers)


def load_dicom_many_files(paths):
    """Load an image array from multiple dicom files and use the spacing 
    between slices to determine the slice thickness.

    **Parameters**:

    paths : list of strs
        List of paths to the dicom files from which the image should be loaded.

    **Returns**:

    image : np.ndarray
        Numpy array containing image data.

    affine : np.ndarray
        3x3 numpy array containing affine matrix.

    ds : pydicom.FileDataset
        Dataset object corresponding to the last file loaded.

    z_paths : dict
        Dict mapping z slice positions in mm  to the filepath 
        from which that slice of the image was read.

    z_instance_numbers : dict
        Dict mapping z slice positions in mm  to the instance
        number of that slice of the image.
    """

    # Set attributes to check for consistency between files
    attrs_to_check = [
        "StudyInstanceUID", 
        "SeriesNumber",
        "Modality",
        "ImageOrientationPatient"
    ]
    attr_vals = {name: None for name in attrs_to_check}

    # Load 2D image arrays from all files
    orientation = None
    axes = None
    data_slices = {}
    image_positions = {}
    z_paths = {}
    z_instance_numbers = {}
    for path in paths:
        
        # Load dataset from this file
        try:
            ds = pydicom.dcmread(path, force=True)
        except pydicom.errors.InvalidDicomError:
            continue

        # Try to ensure that dataset has attribute ImageOrientationPatient
        set_image_orientation_patient(ds)

        # Get orientation info from first file
        if orientation is None:
            orientation, axes = get_dicom_orientation(ds)

        # Check attributes are consistent with others
        attr_ok = True
        for attr in attrs_to_check:
            own_attr = getattr(ds, attr)
            if attr_vals[attr] is None:
                attr_vals[attr] = own_attr
            elif attr_vals[attr] != own_attr:
                attr_ok = False
                break

        if not attr_ok:
            continue

        # Fill empty TransferSyntaxUID 
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = \
                pydicom.uid.ImplicitVRLittleEndian

        # Get data
        pos = getattr(ds, "ImagePositionPatient", [0, 0, 0])
        z = pos[axes[2]]
        z_paths[z] = path
        data_slices[z] = ds.pixel_array
        image_positions[z] = pos
        z_instance_numbers[z] = getattr(ds, "InstanceNumber", None)

    # Stack the 2D arrays into one 3D array
    # Sort by slice position
    sorted_slices = sorted(list(data_slices.keys()))
    sorted_data = [data_slices[z] for z in sorted_slices]
    data = np.stack(sorted_data, axis=-1)
    z_paths = {z : z_paths[z] for z in sorted_slices}
    z_instance_numbers = {z : z_instance_numbers[z] for z in sorted_slices}
    # Get affine matrix
    affine = get_dicom_affine(ds, image_positions)

    return data, affine, ds, z_paths, z_instance_numbers


def load_dicom_single_file(path):
    """Load an image array from a single dicom file.

    **Parameters**:

    path : str
        Path to the dicom file from which the image should be loaded.

    **Returns**:

    image : np.ndarray
        Numpy array containing image data.

    affine : np.ndarray
        3x3 numpy array containing affine matrix.

    ds : pydicom.FileDataset
        Dataset object corresponding to the dicom file.
    """

    ds = pydicom.dcmread(path, force=True)

    # Fill empty TransferSyntaxUID 
    if not hasattr(ds, "TransferSyntaxUID"):
        ds.file_meta.TransferSyntaxUID = \
            pydicom.uid.ImplicitVRLittleEndian

    # Get data and transpose such that it's a 3D array with slice in last
    data = ds.pixel_array
    if data.ndim == 2:
        data = data[..., np.newaxis]
        image_positions = {}
    elif data.ndim == 3:
        # If DICOM dataset has GridFrameOffsetVector defined,
        # use this to determine slice positions.
        offsets = getattr(ds, "GridFrameOffsetVector", [])
        origin = getattr(ds, "ImagePositionPatient", [0, 0, 0])
        image_positions = {origin[2] + offset:
                (origin[0], origin[1], origin[2] + offset)
                for offset in offsets}
        data = data.transpose((1, 2, 0))
        # Invert z-axis if offsets are decreasing as slice index increases.
        if offsets and offsets[0] > offsets[-1]:
            data = data[:, :, ::-1]
    else:
        raise RuntimeError(f"Unrecognised number of image dimensions: {data.ndim}")

    # Try to ensure that dataset has attribute ImageOrientationPatient
    set_image_orientation_patient(ds)

    affine = get_dicom_affine(ds, image_positions)
    return data, affine, ds


def get_dicom_orientation(ds):
    """Extract and parse image orientation from a dicom file.

    **Parameters**:

    ds : pydicom.FileDataset
        Dicom dataset object from which the orientation vector should be read.

    **Returns**:

    orientation: np.ndarray
        Direction cosines of the first row and first column, reshaped to (2, 3)
        such that the top row of the array contains the row direction cosine,
        and the bottom row contains the column direction cosine.

    axes : list
        List of the axes (x = 0, y = 1, z = 2) corresponding to each image
        array axis in order [row, column, slice].
    """

    orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
    axes = [
        sum([abs(int(orientation[i, j] * j)) for j in range(3)])
        for i in range(2)
    ]
    axes.append(3 - sum(axes))
    return orientation, axes


def get_dicom_voxel_size(ds):
    """Get voxel sizes from a dicom file.

    **Parameters**:
    
    ds : pydicom.FileDataset
        Dicom dataset from which to load voxel sizes.

    **Returns**:

    voxel_size : list
        List of voxel sizes in order [row, column, slice].
    """
    # Get voxel spacings
    for attr in ["PixelSpacing", "ImagerPixelSpacing",
            "ImagePlanePixelSpacing"]:
        pixel_size = getattr(ds, attr, None)
        if pixel_size:
            break

    # If not possible to determine pixel size from DICOM data,
    # set to 1 mm x 1 mm
    if not pixel_size:
        pixel_size = [1, 1]

    # Get slice thickness
    slice_thickness = getattr(ds, "SliceThickness", 1)
    if not slice_thickness and hasattr(ds, 'GridFrameOffsetVector'):
        if (isinstance(ds.GridFrameOffsetVector, pydicom.multival.MultiValue)
            and len(ds.GridFrameOffsetVector) > 1):
            slice_thickness = abs(
                    ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])

    return pixel_size[0], pixel_size[1], slice_thickness


def get_dicom_affine(ds, image_positions=None):
    """Assemble affine matrix from a dicom file. Optionally infer slice 
    thickness from the positions of different slices and origin from the 
    minimum slice; otherwise, extract slice thickness and origin directly from 
    the dicom dataset.

    **Parameters**:
    
    ds : pydicom.FileDataset
        Dicom dataset from which to load voxel sizes.

    image_positions : dict, default=None
        Dict of 3D origins for each slice, where keys are slice positions
        and values are origins.

    **Returns**:
        
    affine : np.ndarray
        3x3 array containing the affine matrix for this image.
    """

    # Get voxel sizes and orientation
    voxel_size = get_dicom_voxel_size(ds)
    orientation, axes = get_dicom_orientation(ds)

    # Get slice-related matrix elements
    if image_positions:
        sorted_slices = sorted(list(image_positions.keys()))
        zmin = sorted_slices[0]
        zmax = sorted_slices[-1]
        n = len(sorted_slices)
        if n > 1:
            slice_elements = [
                (image_positions[zmax][i] - image_positions[zmin][i]) / (n - 1)
                for i in range(3)
            ]
        else:
            slice_elements = [0] * 3
            slice_elements[axes[2]] = voxel_size[2]
        origin = image_positions[zmin]
    else:
        slice_elements = [0] * 3
        slice_elements[axes[2]] = voxel_size[2]
        origin = getattr(ds, "ImagePositionPatient", [0, 0, 0])
        n_slices = getattr(ds, "NumberOfFrames", 1)
        origin[2] -= (n_slices - 1) * voxel_size[2]

    # Make affine matrix
    affine = np.array(
        [
            [
                orientation[0, 0] * voxel_size[0],
                orientation[1, 0] * voxel_size[1],
                slice_elements[0],
                origin[0]
            ],
            [
                orientation[0, 1] * voxel_size[0],
                orientation[1, 1] * voxel_size[1],
                slice_elements[1],
                origin[1]
            ],
            [
                orientation[0, 2] * voxel_size[0],
                orientation[1, 2] * voxel_size[1],
                slice_elements[2],
                origin[2]
            ],
            [0, 0, 0, 1],
        ]
    )
    return affine


def rescale_dicom_data(ds, data):
    """Rescale an array according to rescaling info in a dicom dataset.

    **Parameters**:

    ds : pydicom.FileDataset
        Dicom dataset from which to read rescaling info.

    data : np.ndarray
        Image array to be rescaled.

    **Returns**:

    data : np.ndarray
        Rescaled version of the input array
    """

    # Get rescale settings
    rescale_slope = getattr(ds, "RescaleSlope", None)
    if rescale_slope is None:
        rescale_slope = getattr(ds, "DoseGridScaling", 1.)
    rescale_intercept = getattr(ds, "RescaleIntercept", 0.)

    # Apply rescaling
    return (data * float(rescale_slope)
            + float(rescale_intercept)).astype(np.float32)

def get_dicom_window(ds):
    """Get intensity window defaults from a dicom file.

    **Parameters**:
    ds : pydicom.FileDataset
        Dicom dataset from which to read intensity window info.

    **Returns**:

    window_centre : float
        Default window centre.

    window_width : float
        Default window width.
    """

    window_centre = getattr(ds, "WindowCenter", None)
    if isinstance(window_centre, pydicom.multival.MultiValue):
        window_centre = window_centre[0]
    window_width = getattr(ds, "WindowWidth", None)
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]

    return window_centre, window_width


def load_npy(path):
    """Load a numpy array from a .npy file."""

    try:
        data = np.load(path)
        return data

    except (IOError, ValueError):
        return


def downsample(data, dx=None, dy=None, dz=None):
    """Downsample an array by the factors specified in <dx>, <dy>, and <dz>."""

    if dx is None:
        dx = 1
    if dy is None:
        dy = 1
    if dx is None:
        dz = 1

    return data[:: round(dy), :: round(dx), :: round(dz)]


def to_inches(size):
    """Convert a size string to a size in inches. If a float is given, it will
    be returned. If a string is given, the last two characters will be used to
    determine the units:

        - 'in': inches
        - 'cm': cm
        - 'mm': mm
        - 'px': pixels
    """

    if not isinstance(size, str):
        return size

    val = float(size[:-2])
    units = size[-2:]
    inches_per_cm = 0.394
    if units == "in":
        return val
    elif units == "cm":
        return inches_per_cm * val
    elif units == "mm":
        return inches_per_cm * val / 10
    elif units == "px":
        return val / mpl.rcParams["figure.dpi"]


def write_nifti(outname, data, affine):
    """Create a nifti file at <outname> containing <data> and <affine>."""

    nii = nibabel.Nifti1Image(data, affine)
    nii.to_filename(outname)


def write_npy(outname, data, affine=None):
    """Create numpy file containing data. If <affine> is not None, voxel
    sizes and origin will be written to a text file."""

    np.save(outname, data)
    if affine is not None:
        voxel_size = np.diag(affine)[:-1]
        origin = affine[:-1, -1]
        geom_file = outname.replace(".npy", ".txt")
        with open(geom_file, "w") as f:
            f.write("voxel_size")
            for vx in voxel_size:
                f.write(" " + str(vx))
            f.write("\norigin")
            for p in origin:
                f.write(" " + str(p))
            f.write("\n")


def default_aspect():
    return 1


def set_ax(
    obj,
    view=None,
    ax=None,
    gs=None,
    figsize=None,
    zoom=None,
    colorbar=False,
    aspect_getter=default_aspect,
    **kwargs,
):
    """Set up axes for plotting an object, either from a given exes or
    gridspec, or by creating new axes."""

    # Set up figure/axes
    if ax is None and gs is not None:
        ax = plt.gcf().add_subplot(gs)
    if ax is not None:
        obj.ax = ax
        obj.fig = ax.figure
    else:
        if figsize is None:
            figsize = _default_figsize
        if skrt.core.is_list(figsize):
            fig_tuple = figsize
        else:
            aspect = aspect_getter(view, zoom, colorbar, figsize)
            figsize = to_inches(figsize)
            fig_tuple = (figsize * aspect, figsize)
        obj.fig = plt.figure(figsize=fig_tuple)
        obj.ax = obj.fig.add_subplot()


def get_geometry(affine, voxel_size, origin, is_nifti=False, shape=None):
    """Get an affine matrix, voxel size list, and origin list from 
    a combination of these inputs."""

    # Get affine matrix from voxel size and origin
    if affine is None:
        
        if voxel_size is None or origin is None:
            return None, None, None

        voxel_size = list(voxel_size)
        origin = list(origin)

        affine = np.array(
            [
                [voxel_size[0], 0, 0, origin[0]],
                [0, voxel_size[1], 0, origin[1]],
                [0, 0, voxel_size[2], origin[2]],
                [0, 0, 0, 1],
            ]
        )
        if is_nifti:
            if shape is None:
                raise RuntimeError("Must provide data shape if converting "
                                   "affine matrix from nifti!")

            affine[0, :] = -affine[0, :]
            affine[1, 3] = -(
                affine[1, 3] + (shape[1] - 1) * voxel_size[1]
            )

    # Otherwise, get origin and voxel size from affine
    else:
        voxel_size = list(np.diag(affine))[:-1]
        origin = list(affine[:-1, -1])

    return affine, voxel_size, origin


def pad_transpose(transpose, ndim):
    """Pad a transpose vector to match a given number of dimensions."""

    nt = len(transpose)
    if ndim > nt:
        for i in range(ndim - nt):
            transpose.append(i + nt)
    return transpose

def get_box_mask_from_mask(image=None, dx=0, dy=0):
    '''
    Slice by slice, create box masks enclosing an arbitrarily shaped mask.

    **Parameters:**

    image : Image, default=None
        Image object representing arbitrarily shaped mask for which
        slice-by-slice box masks are to be determined.

    dx : int, default=0
        Margin along columns to be added on each side of mask bounding box.

    dy : int, default=0
        Margin along rows to be added on each side of mask bounding box.
    '''

    if not _has_mahotas:
        print('WARNING: Module mahotas unavailable')
        print('WARNING: Unable to execute function '\
                + 'skrt.image.Image.get_box_mask_from_mask()')

    # Retrieve image data and numbers of voxels.
    mask_array = image.get_data()
    nx, ny, nz = image.get_n_voxels()

    # Initialise output array.
    out_array = np.zeros((ny, nx, nz), np.int8)

    # Slice by slice, determine bounding box of mask, and fill with ones.
    for iz in range(nz):
        jmin, jmax, imin, imax = mahotas.bbox(mask_array[:, :, iz])
        jmin = min(ny - 1, max(0, jmin - dy))
        jmax = max(0, min(ny - 1, jmax + dy))
        imin = min(nx - 1, max(0, imin - dx))
        imax = max(0, min(nx - 1, imax + dx))
        out_array[jmin: jmax, imin: imax, iz] = 1

    # Create clone of input image, then assign box mask as its data.
    out_image = Image(image)
    out_image.data = out_array

    return out_image

def set_image_orientation_patient(ds):
    '''
    Try to ensure that image orientation patient is set for DICOM dataset.

    If the dataset doesn't have the attribute ImageOrientationPatient,
    the effective value is determined from PatientOrientation, if present.

    **Parameter:**

    ds : pydicom.FileDataset
        DICOM dataset.
    '''

    direction_cosines = {
            'P' : [0, -1, 0],
            'A' : [0, 1, 0],
            'L' : [-1, 0, 0],
            'R' : [1, 0, 0],
            'FR' : [1, 0, 0],
            'H' : [0, 0, 1],
            'F' : [0, 0, -1],
            }
    
    if not hasattr(ds, 'ImageOrientationPatient'):
        patient_orientation =  getattr(ds, 'PatientOrientation', [])
        unknown_orientations = (
            {"AL", "LA", "PR", "RP"}.intersection(patient_orientation))
        if patient_orientation and not unknown_orientations:
            ds.ImageOrientationPatient = (
                    direction_cosines[patient_orientation[1]] +
                    direction_cosines[patient_orientation[0]])
        else:
            if unknown_orientations:
                print('WARNING: PatientOrientation not understood: '
                      f'{patient_orientation}')
            else:
                print('WARNING: ImageOrientationPatient '
                      'and PatientOrientation undefined')
            patient_position = getattr(ds, 'PatientPosition', None)
            if patient_position:
                print(f'Patient position: {patient_position}')
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            print('WARNING: Guessing ImageOrientationPatient = '
                 f'\'{ds.ImageOrientationPatient}\'')

def kv_to_mv(hu):
    '''
    Map radiodensity in Hounsfield units from kV CT scan to MV CT scan.

    Function originally written by M.Z. Wilson.  Parameterisation
    derived from kV and MV CT scans collected in VoxTox study.

    **Parameter:**

    hu - int
        Radiodensity (Hounsfield units) for kV CT scan.
    '''

    # hu is the old kVCT HU
    # y is the new synthetic MVCT HU
    # x's are the relative electron densities

    # remove issues with dental implants
    if hu > 2200:
        hu = 2200

    if hu > 192:
        y = 945.8 *((hu + 1754.3) / 1769.5) - 944.91

    elif 75 < hu < 193:
        # fit a spline curve using the densities
        x0 = np.linspace(0, 2, 100)
        x1 = x0[:54]
        x2 = x0[57:]
        xnew = np.delete(x0, [54, 55, 56])
        # upper kvct line
        y2 = 1769.5 * x2 - 1754.3
        # lower kvct line
        y3 = 970.2 * x1 - 991.98
        ynew = np.concatenate((y3,y2), axis=0)

        # fit a spline between the two kvct calibration lines
        # for each hu value - shift the y axis such that
        # the density can be found by finding the roots of
        # the spline. This density can then be used in the
        # simple straight line fit of the MVCT calibration
        yreduced = np.array(ynew[45:65]) - hu
        freduced = scipy.interpolate.UnivariateSpline(
                xnew[45:65], yreduced, s=0)
        density = freduced.roots()[0]
 
        y = (945.8 * density) - 944.91

    elif hu < 76:
        y = 945.8 * ((hu + 991.98) / 970.2) - 944.91

    elif hu <= -1024:
        y = -1024

    return round(y)

def get_mask(mask=None, mask_threshold=0.5, image_to_match=None):
    """
    Return mask as Image object, with boolean data, resizing if required.

    **Parameters:**

    mask : Image/list/ROI/str/StructureSet, default=None
        Image object representing a mask or a source from which
        an Image object can be initialised.  In addition to the
        sources accepted by the Image constructor, the source
        may be an ROI, a list of ROIs or a StructureSet.  In the
        latter cases, the mask image is derived from the ROI mask(s).

    mask_threshold : float, default=0.5
        Threshold for mask data.  Values above and below this value are
        set to True and False respectively.  Taken into account only
        if the mask image has non-boolean data.

    image_to_match : Image/str, default=None
        Image object or a source from which an Image object can be
        initialised.  The mask will be resized if needed, to match
        this image.
    """

    if mask:
        # Try to ensure that mask is an Image object.
        # This includes workaround for using ROI and StructureSet,
        # while avoiding the cyclic imports that would result
        # from attempting to import the structures module.
        if hasattr(mask, "get_mask_image"):
            mask = mask.get_mask_image()
        elif (isinstance(mask, list)
                and "skrt.structures.ROI" in str(type(mask[0]))):
            roi_new = mask(0).clone()
            roi_new.name = "mask_from_roi_list"
            for i in range(1, len(mask)):
                roi_new.mask.data += self.get_roi(roi_names[i]).get_mask()
            roi_new.mask.data = roi_new.mask.dat > mask_threshold
            mask = roi_new.get_mask_image()
        elif not isinstance(mask, Image):
            mask = Image(mask)

        # Match mask to size of reference image.
        mask.get_data()
        if image_to_match is not None:
            if not isinstance(image_to_match, Image):
                image_to_match = Image(image_to_match)
            mask.match_size(image_to_match, 0, "nearest")

        # Ensure data array is boolean.
        if mask.data.dtype != bool:
            mask.data = mask.data > mask_threshold
        return mask

def remove_duplicate_images(images=None):
    '''
    Remove duplicates from a list of image objects.

    Image instance image1 is taken to be a duplicate of image2 if
    image1.has_same_data(image2) is True.

    **Parameter:**
    images: list, default=None
        List of image objects, from which duplicates are to be removed.
    '''
    images = images or []
    filtered_images = []

    # Add to list of filtered images each image that isn't a duplicate
    # of an image already in the list.
    while images:
        image1 = images.pop()
        duplicate_image = False
        for image2 in filtered_images:
            if image1.has_same_data(image2):
                duplicate_image = True
                break
        if not duplicate_image:
            filtered_images.append(image1)

    return filtered_images

def sum_images(images=None):
    '''
    Sum images of the same geometry.

    If not all images have the same same geometry (shape, origin,
    voxel size), None is returned.  Otherwise, an Image object is
    returned that has the same geometry as the input images, and
    where array values are the sums of the array values of
    the input images.

    **Parameter:**
    images: list, default=None
        List of Image objects to be summed.
    '''

    # Check that a list of at least two Image objects has been passed
    # as input.  If not, return something reasonable.
    if issubclass(type(images), Image):
        return images
    if not isinstance(images, (list, set, tuple)):
        return None
    images = [image for image in images if issubclass(type(image), Image)]
    if not len(images):
        return None
    if 1 == len(images):
        return images[0]

    # Sum images.
    image_sum = images[0].clone()
    for image in images[1: ]:
        if not image.has_same_geometry(image_sum):
            image_sum = None
            break
        image_sum += image

    return image_sum

def get_mask_bbox(mask):
    """
    Obtain bounding box of labelled area of label mask.

    The bounding box is returned as [(xmin, xmax), (ymin, ymax), (zmin, zmax)],
    with values in mm.

    **Parameter:**

    mask : skrt.image.Image
        Image object representing label mask for which bounding box is
        to be obtained.
    """
    if not _has_mahotas:
        print('WARNING: Module mahotas unavailable')
        print('WARNING: Unable to execute function '\
                + 'skrt.image.get_mask_bbox()')

    jmin, jmax, imin, imax, kmin, kmax = mahotas.bbox(mask.get_data())
    bbox = []
    for axis, idxs in enumerate([(imin, imax), (jmin, jmax), (kmin, kmax)]):
        bbox.append(tuple(sorted(
            [mask.idx_to_pos(idxs[0], axis), mask.idx_to_pos(idxs[1], axis)]
            )))

    return bbox

def get_translation_to_align(im1, im2, alignments=None, default_alignment=2,
        threshold=None):
    """
    Determine translation for aligning <im1> to <im2>.

    **Parameters:**

    im1 : skrt.image.Image
        Image that is to be translated to achieve the alignment.

    im2 : skrt.image.Image
        Image with which alignment is to be performed.

    alignments : dict, default=None
        Dictionary indicating how alignment is to be performed along each
        axis, where keys are axis identifiers ('x', 'y', 'z'), and values
        are the types of alignment.  The valid alignment values are:
        - 1: align on lowest coordinates (right, posterior, inferior);
        - 2: align on centre coodinates;
        - 3: align on highest coordinates (left, anterior, superior).
        If an axis isn't included in the dictionary, or is included with
        an invalid alignment type, the value of <default_alignment> is
        used for this axis.

    default_alignment : int, default=2
        Type of alignment to be applied along any axis not included in
        the <alignments> dictionary.

    threshold : int/float, default=None
        If None, alignment is with respect to the whole images.  If an
        integer or float, alignment is with respect to the masks
        returned for the images by skrt.image.Image.get_foreground_mask(),
        using value specified as the threshold parameter for mask creation.
    """
    # Create message logger for this function.
    logger = skrt.core.get_logger(identifier="funcName")

    # Initialise dictionaries of alignments.
    alignments = alignments or {}
    checked_alignments = {}

    # Ensure that the default_alignment is valid.
    valid_alignments = [1, 2, 3]
    fallback_alignment = 2
    if default_alignment not in valid_alignments:
        logger.warning(f"Invalid default_alignment: {default_alignment}")
        logger.warning(f"Valid alignment values are: {valid_alignments}")
        logger.warning(f"Setting default_alignment={fallback_alignment}")
        default_alignment = fallback_alignment

    # Check alignment values, and store the ones that are valid.
    for axis, alignment in sorted(alignments.items()):
        if isinstance(axis, str) and axis.lower() in _axes:
            if isinstance(alignment, int) and alignment in [1, 2, 3]:
                checked_alignments[axis.lower()] = alignment
            else:
                logger.warning(f"Axis {axis}, disregarding invalid "
                        f"alignment value: {alignment}")
                logger.warning("Valid alignment values are: 1, 2, 3")
        else:
            logger.warning(f"Disregarding invalid axis label: {axis}")
            logger.warning(f"Valid axis labels are: {_axes}")
    
    # Determine image bounding boxes.
    xyz_lims = {}
    for image in [im1, im2]:
        if threshold is not None:
            # Obtain bounding box for foreground mask.
            xyz_lims[image] = get_mask_bbox(
                    image.get_foreground_mask(threshold))
        else:
            # Obtain bounding box for whole image.
            xyz_lims[image] = image.get_extents()

    # Determine translation along each axis, for type of alignment specified.
    translation = []
    for idx, axis in enumerate(_axes):
        alignment = checked_alignments.get(axis, default_alignment)

        # Lower alignment.
        if 1 == alignment:
            dxyz = xyz_lims[im2][idx][0] - xyz_lims[im1][idx][0]
        # Centre alignment.
        elif 2 == alignment:
            dxyz = 0.5 * (sum(xyz_lims[im2][idx]) - sum(xyz_lims[im1][idx]))
        # Upper alignment.
        elif 3 == alignment:
            dxyz = xyz_lims[im2][idx][1] - xyz_lims[im1][idx][1]

        translation.append(dxyz)

    return tuple(translation)

def get_translation_to_align_image_rois(im1, im2, roi_name1, roi_name2,
        z_fraction1=None, z_fraction2=None):
    """
    Determine translation for aligning ROI of <im1> to ROI of <im2>.

    **Parameters:**

    im1 : skrt.image.Image
        Image with linked StructureSet containing ROI to be
        translated to achieve the alignment.

    im2 : skrt.image.Image
        Image with linked StructureSet containing ROI with
        which alignment is to be performed.

    roi_name1 : str
        Name of ROI contained in StructureSet linked to <im1>,
        and that it to be translated to achieve the alignment.

    roi_name2 : str
        Name of ROI contained in StructureSet linked to <im2>,
        and with which alignment is to be performed.

    z_fraction1 : float, default=None
        For ROI identified by <roi_name1>, position along z axis
        of iROI slice on which to align.  If None, alignment is
        to the centroid of the whole ROI volume.  Otherwise, alignment
        is to the centroid of the slice at the specified distance
        from the ROI's most-inferior point: 0 corresponds to
        the most-inferior point (lowest z); 1 corresponds to the
        most-superior point (highest z).  Values for z_fraction
        outside the interval [0, 1] result in a RuntimeError.

    z_fraction2 : float, default=None
        For ROI identified by <roi_name2>, position along z axis
        of ROI slice on which to align.  If None, alignment is
        to the centroid of the whole ROI volume.  Otherwise, alignment
        is to the centroid of the slice at the specified distance
        from the ROI's most-inferior point: 0 corresponds to
        the most-inferior point (lowest z); 1 corresponds to the
        most-superior point (highest z).  Values for z_fraction
        outside the interval [0, 1] result in a RuntimeError.
    """
    return im1.get_rois(roi_name1)[0].get_translation_to_align(
            im2.get_rois(roi_name2)[0], z_fraction1, z_fraction2)

def get_alignment_translation(im1, im2, alignment=None):
    """
    Determine translation for aligning <im1> to <im2>, based on <alignment>.

    **Parameters:**

    im1 : skrt.image.Image
        Image that is to be translated to achieve the alignment.

    im2 : skrt.image.Image
        Image with which alignment is to be performed.

    alignment : tuple/dict/str, default=None
        Strategy to be used for image alignment.  This can be
        translation-based, image-based or ROI-based.

    - Translation-based initial alignment:
      This is specified by:
      - A tuple indicating the amounts (dx, dy, dz) by which a
        point in <im1> must be translated to align with
        the corresponding point in <im2>;

    - Image-based alignment:
      This can be specified by:
      - A dictionary indicating how <im1> and <im2> are
        to be aligned along each axis, where keys are axis
        identifiers ('x', 'y', 'z'), and values are the types
        of alignment:
        - 1: align on lowest coordinates (right, posterior, inferior);
        - 2: align on centre coodinates;
        - 3: align on highest coordinates (left, anterior, superior).
        If an axis isn't included in the dictionary, or is included
        with an invalid alignment type, the alignment type defaults to 2.
      - One of the strings "_top_", "_centre_", "_bottom_",
        in which case <im1> and <im2> have their (x, y)
        centres aligned, and have z positions aligned at
        image top, centre or bottom.

    - ROI-based alignment:
      This defines a translation of <im1> so that an ROI associated
      with <im1> is aligned with an ROI associated with <im2>.  For
      each image, the alignment point is defined by an ROI name and an
      optional position.  If the optional position is omitted
      or is None, the alginment point is the centroid of the
      ROI as a whole.  Otherwise, the alignment point is the
      centroid of an (x, y) slice through the ROI at the
      specified relative position along the z-axis.  This
      relative position must be in the interval [0, 1], where
      0 corresponds to the most-inferior point (lowest z) and
      1 corresponds to the most-superior point (highest z).
          
      The full ROI-based alignment specification is a
      tuple of two two-element tuples:
      (("im1_roi_name", im1_roi_position),
      ("im2_roi_name", im2_roi_position)).
      If a position is omitted, it defaults to None, and the
      ROI name can be given either as a string or as a
      one-element tuple.  If information is given only for
      <im1>, the same information is used for <im2>.

      The following are examples of valid ROI-based alignment
      specifications, and how they're interpreted:

      - "roi":
        align "roi" of <im1> with "roi" of <im2>,
        aligning on volume centroids;

      - ("roi1", "roi2"):
        align "roi1" of <im1> with "roi2" of <im2>,
        aligning on volume centroids;
             
      - ("roi", 1):
        align "roi" of <im1> with "roi" of <im2>,
        aligning on centroids of top slices;

      - (("roi1", 0.75), "roi2"):
        align "roi1" of <im1> with "roi2" of <im2>,
        aligning centroid of slice three quarters of the way
        up "roi1" with the volume centroid of "roi2".

      Note that ROI-based alignment relies on the named
      ROIs being contained in structure sets associated with
      <im1> and <im2>.  Association of structure sets
      to images may be performed automatically when loading
      DICOM data to Patient objects, or may be performed
      manually using an Image object's set_structure_set() method.
   """
    # If alignment specified as a 3-element list-like object,
    # return this as the translation.
    if skrt.core.is_list(alignment) and len(alignment) == 3:
        return alignment

    # Parse alignment string.
    alignments = get_alignment_strategy(alignment)

    # Define translation for image-based alignment.
    if isinstance(alignments, dict):
        # Always use standardised data.
        return get_translation_to_align(
            Image(im1.get_standardised_data(),
                affine=im1.get_standardised_affine()),
            Image(im2.get_standardised_data(),
                affine=im2.get_standardised_affine()),
            alignments)

    # Return None for null alignment.
    if not alignments:
        return

    # Define translation for roi-based alignment.
    return get_translation_to_align_image_rois(
        im1, im2, alignments[0][0], alignments[1][0],
        alignments[0][1], alignments[1][1])

def get_alignment_strategy(alignment=None):
    """
    Extract information defining strategy for image alignment.

    alignment : tuple/dict/str, default=None
        Strategy to be used for image alignment.  For defailts,
        see documentation of skrt.image.get_alignment_translation().
    """
    # If alignment passed as a dictionary, return this.
    if isinstance(alignment, dict):
        return alignment

    # Return None for null alignment.
    if not alignment:
        return

    # If alignment passed as a position string,
    # create alignment dictionary based on this.
    image_alignments = ["_bottom_", "_centre_", "_top_"]
    if alignment in image_alignments:
        return {"x": 2, "y": 2, "z": 1 + image_alignments.index(alignment)}

    # Alignment passed as name only for a single ROI.
    roi_alignments = None
    if isinstance(alignment, str):
        roi_alignments = [(alignment, None), (alignment, None)]

    # Alignment passed as list-like object.
    elif skrt.core.is_list(alignment):
        # Name given for a single ROI.
        if len(alignment) == 1:
            roi_alignments = [(alignment[0], None), (alignment[0], None)]
        # Information for two ROIs,
        # or name and position for a single ROI.
        elif len(alignment) == 2:
            if skrt.core.is_list(alignment[0]):
                # Name and position passed for two ROIs
                if skrt.core.is_list(alignment[1]):
                    roi_alignments = list(alignment)
                # Name and position passed for first ROI,
                # Name only passed for second ROI.
                else:
                    roi_alignments = [alignment[0], (alignment[1], None)]
            else:
                # Name only passed for first ROI,
                # Name and position passed for second ROI.
                if skrt.core.is_list(alignment[1]):
                    roi_alignments = [(alignment[0], None), alignment[1]]
                else:
                    # Name passed for two ROIs.
                    if isinstance(alignment[1], str):
                        roi_alignments = [(alignment[0], None),
                                (alignment[1], None)]
                    # Name and position passed for one ROI.
                    else:
                        roi_alignments = [alignment, alignment]

            # Deal with ROI name passed as single-element tuples.
            for idx in range(len(roi_alignments)):
                if 1 == len(roi_alignments[idx]):
                    roi_alignments[idx] = [roi_alignments[idx][0], None]

    return roi_alignments

def match_images(im1, im2, ss1=None, ss2=None, ss1_index=-1, ss2_index=-1,
                 ss1_name=None, ss2_name=None, roi_names=None,
                 im2_crop_focus=None, im2_crop_margins=None,
                 im2_crop_about_centre=False, alignment=None,
                 voxel_size=None, bands=None):
    """
    Process pair of images, so that they match in one or more respects.

    Matching can be with respect to image size, voxel size,
    grey-level banding, and/or ROI naming in associated structure sets.
    The returned images are created from clones of the input images,
    which are left unchanged.

    The second ROI of the pair may be cropped about a focus (ROI or point).
    The first ROI of the pair may be cropped to the size of the first.

    **Parameters:**

    im1, im2 : skrt.image.Image
        Images to be matched with one another.

    ss1, ss2 : skrt.structures.StructureSet, default=None
        Structure sets to associate with images im1, im2.  If a value
        is None, and image matching involves ROI alignment, a structure
        set already associated with the relevant image is used
        (im1.structure_sets[ss1_index], im2.structure_sets[ss2_index]).

    ss1_index, ss2_index : int, default=-1
        Structure set indices to use for ROI alignment based on
        structure sets already associated with images.

    ss1_name, ss2_name : str, default=None
        Names to be assigned to structure sets associated with the
        returned image objects.  If a value is None, the original
        name is kept.

    roi_names : dict, default=None
        Dictionary for renaming and filtering ROIs for inclusion
        in the structure sets associated with the output images.
        Keys are names for ROIs to be kept, and values are lists of
        alternative names with which these ROIs may have been
        labelled in the input structure sets.  The alternative names
        can contain wildcards with the '*' symbol.  If a value of
        None is given, all ROIs in the input structure sets are kept,
        with no renaming.

    im2_crop_focus : str/tuple, default=None
        Name of an ROI, or (x, y, z) coordinates of a point, about
        which to perform cropping of im2.  If None, cropping is performed
        about the point (0, 0, 0) if im2_crop_margins is non-null, or
        otherwise no cropping is performed.

    im2_crop_margins, float/tuple, default=None
        Specification of margin around focus for cropping of im2.
        For information on specifying crop margins for the case where
        im2_crop_focus is the name of an ROI, see documentation for
        method skrt.image.Image.crop_to_roi(). For the case where
        im2_crop_focus is a point coordinate, margins should be
        sepecified by a tuple (xlim, ylim, zlim).  Here, xlim, ylim, zlim
        are two-component tuples specifying lower and upper bounds
        relative to the crop point.

    im2_crop_about_centre: bool, default=False
        For the case where im2_crop_focus is the name of an ROI:
        if True, im2 is cropped to the centre point of the ROI plus margins;
        if False, im2 is cropped to the ROI extents plus margins.
        Ignored for the case where im2_crop_focus isn't the name of an ROI.

    alignment : tuple/dict/str, default=None
        Strategy to be used for aligning images prior to cropping
        so that they have the same size.  For strategy details, see
        documentation of skrt.image.get_alignment_translation().
        After alignment, im1 is cropped to the size of im2, then im2
        is cropped to the size of im1.  To omit cropping to the same
        size, set alginment to False.

    voxel_size : tuple/str/float, default=None
         Specification of voxel size for image resampling.
         For possible values, see documentation for function
         skrt.image.match_image_voxel_size().

    bands - dict, default=None
        Dictionary of value bandings to be applied to image data.
        Keys specify band limits, and values indicte the values
        to be assigned.  For more information, see documentation of
        method skrt.image.Image.apply_banding().
    """
    im1 = im1.clone_with_structure_set(ss1, roi_names, ss1_index, ss1_name)
    ss1 = im1.structure_sets[0] if im1.structure_sets else None
    im2 = im2.clone_with_structure_set(ss2, roi_names, ss2_index, ss1_name)
    ss2 = im2.structure_sets[0] if im2.structure_sets else None

    if alignment is not False:
        im1.crop_to_image(im2, alignment)
        im2.crop_to_image(im1, alignment)

    # Resample images to same voxel size.
    match_image_voxel_sizes(im1, im2, voxel_size)

    # Crop im2 to region around focus.
    if ss2 is not None and im2_crop_focus in ss2.get_roi_names():
        im2.crop_to_roi(ss2[im2_crop_focus], crop_margins=im2_crop_margins,
                        crop_about_centre=im2_crop_about_centre)
    elif ((skrt.core.is_list(im2_crop_focus) or im2_crop_focus is None)
          and skrt.core.is_list(im2_crop_margins)):
        im2.crop_about_point(im2_crop_focus, *im2_crop_margins)

    # Crop images to same size.
    if alignment is not False:
        im1.crop_to_image(im2, alignment)
        im2.crop_to_image(im1, alignment)
        if (im1.get_voxel_size() == im2.get_voxel_size()
            and im1.get_n_voxels() != im2.get_n_voxels()):
            im1.resize(im2.get_n_voxels(), method="nearest")

    # Perform banding of image grey levels.
    im1.apply_selective_banding(bands)
    im2.apply_selective_banding(bands)

    # Reset structure-set images.
    for im, ss in [(im1, ss1), (im2, ss2)]:
        if ss:
            im.clear_structure_sets()
            ss.set_image(im)

    return (im1, im2)

def match_images_for_comparison(
        im1, im2, ss1=None, ss2=None, ss1_index=-1, ss2_index=-1,
        ss1_name=None, ss2_name=None, roi_names=None, alignment=None,
        voxel_size=None):
    """
    Process pair of images, to allow their comparison.

    Images are optionally aligned, then their geometries are matched,
    so that they can be handled by the image-comparison methods
    Image.get_comparison() and Image.get_quality().

    This function provides a subset of the processing options offered
    by match_images(), and in addition can change the origin of the first
    image, to align better with the second image.

    **Parameters:**

    im1, im2 : skrt.image.Image
        Images to be matched for comparison.

    ss1, ss2 : skrt.structures.StructureSet, default=None
        Structure sets to associate with images im1, im2.  If a value
        is None, and image matching involves ROI alignment, a structure
        set already associated with the relevant image is used
        (im1.structure_sets[ss1_index], im2.structure_sets[ss2_index]).

    ss1_index, ss2_index : int, default=-1
        Structure set indices to use for ROI alignment based on
        structure sets already associated with images.

    ss1_name, ss2_name : str, default=None
        Names to be assigned to structure sets associated with the
        returned image objects.  If a value is None, the original
        name is kept.

    roi_names : dict, default=None
        Dictionary for renaming and filtering ROIs for inclusion
        in the structure sets associated with the output images.
        Keys are names for ROIs to be kept, and values are lists of
        alternative names with which these ROIs may have been
        labelled in the input structure sets.  The alternative names
        can contain wildcards with the '*' symbol.  If a value of
        None is given, all ROIs in the input structure sets are kept,
        with no renaming.

    alignment : tuple/dict/str, default=None
        Strategy to be used for aligning images prior to cropping
        so that they have the same size.  For strategy details, see
        documentation of skrt.image.get_alignment_translation().
        After alignment, im1 is cropped to the size of im2, then im2
        is cropped to the size of im1.  To omit cropping to the same
        size, set alginment to False.

    voxel_size : tuple/str/float, default=None
         Specification of voxel size for image resampling.
         For possible values, see documentation for function
         skrt.image.match_image_voxel_size().
    """
    im1 = im1.clone_with_structure_set(ss1, roi_names, ss1_index, ss1_name)
    ss1 = im1.structure_sets[0] if im1.structure_sets else None
    im2 = im2.clone_with_structure_set(ss2, roi_names, ss2_index, ss2_name)
    ss2 = im2.structure_sets[0] if im2.structure_sets else None

    # Resample images to same voxel size.
    if voxel_size is not None:
        match_image_voxel_sizes(im1, im2, voxel_size)

    # Translate first image, to try to align with second image.
    if alignment is not None:
        translation = im1.get_alignment_translation(im2, alignment)
        im1.translate_origin(translation)

    # Crop images to one another.
    im2.crop_to_image(im1)
    im1.crop_to_image(im2)

    # Match geometries.
    im1.match_size(im2, method="nearest")

    return (im1, im2)

def match_image_voxel_sizes(im1, im2, voxel_size=None, order=1):
    """
    Resample pair of images to same voxel size.

    **Parameters:**

    im1, im2 : skrt.image.Image
        Images to resample.

    voxel_size : tuple/str/float, default=None
         Specification of voxel size for image resampling.
         Possible values are:

         - None: no resampling performed;
         - "dz_max": the image with smaller slice thickness is resampled
           to have the same voxel size as the image with larger
           slice thickness;
         - "dz_min": the image with larger slice thickness is resampled
           to have the same voxel size as the image with smaller
           slice thickness;
         - (dx, dy, dz): both images are resampled, to have voxels with the
           specified dimensions in mm.
         - dxyz: both images are resampled, to have voxels with the
           specified dimension in mm along all axes.

    order: int, default = 1
        Order of the b-spline used in interpolating voxel intensity values.
    """
    # No resampling needed.
    if not voxel_size:
        return (im1, im2)

    # Initialise voxel sizes for resampled images.
    vs1 = None
    vs2 = None

    # If voxel size specified by a single number,
    # convert to a list with this number repeated.
    if isinstance(voxel_size, numbers.Number):
        voxel_size = 3 * [voxel_size]

    # Set target voxel size to be the same for both images.
    if skrt.core.is_list(voxel_size):
        if im1.get_voxel_size() != list(voxel_size):
            vs1 = voxel_size
        if im2.get_voxel_size() != list(voxel_size):
            vs2 = voxel_size

    # Resample to voxel size of image with smaller slice thickness.
    elif "dz_min" == voxel_size:
        if im1.get_voxel_size()[2] > im2.get_voxel_size()[2]:
            vs1 = im2.get_voxel_size()
        elif im1.get_voxel_size()[2] < im2.get_voxel_size()[2]:
            vs2 = im1.get_voxel_size()

    # Resample to voxel size of image with larger slice thickness.
    elif "dz_max" == voxel_size:
        if im1.get_voxel_size()[2] < im2.get_voxel_size()[2]:
            vs1 = im2.get_voxel_size()
        elif im1.get_voxel_size()[2] > im2.get_voxel_size()[2]:
            vs2 = im1.get_voxel_size()

    # Perform the resampling.
    # Either or both of vs1, vs2 may be None.  This is dealt with
    # in the resample() method.
    im1.resample(vs1, order)
    im2.resample(vs2, order)

    return (im1, im2)

def crop_by_amounts(obj, dx=None, dy=None, dz=None):
    """
    Crop image or ROI by the amounts dx, dy, dz in mm.

    **Parameters:**

    obj : skrt.image.Image/skrt.structures.ROI
        Object (Image or ROI) for which cropping is to be performed.

    dx : float/tuple, default=None
        Amount(s) by which to crop object in x-direction.  If dx
        is a float, the object is cropped by this amount on both sides.
        If dx is a tuple, the object is cropped on the sides at lower
        and higher x by the amounts of the tuple's first and second
        elements.  No cropping is performed for a crop amount set to None.

    dy : float/tuple, default=None
        Amount(s) by which to crop object in y-direction.  If dy
        is a float, the object is cropped by this amount on both sides.
        If dy is a tuple, the object is cropped on the sides at lower
        and higher y by the amounts of the tuple's first and second
        elements.  No cropping is performed for a crop amount set to None.

    dz : float/tuple, default=None
        Amount(s) by which to crop object in z-direction.  If dz
        is a float, the object is cropped by this amount on both sides.
        If dz is a tuple, the object is cropped on the sides at lower
        and higher z by the amounts of the tuple's first and second
        elements.  No cropping is performed for a crop amount set to None.
    """
    # Define lists of initial object extents, and amounts by which to reduce.
    xyz_lims = [list(extents) for extents in obj.get_extents()]
    xyz_reductions = [skrt.core.to_list(dxyz, 2) for dxyz in [dx, dy, dz]]

    # Crop along each axis in turn.
    for i_ax, reductions in enumerate(xyz_reductions):

        if (skrt.core.is_list(reductions) and
                (reductions[0] or reductions[1])):
            # Set new object extents, after reductions on each side.
            if reductions[0]:
                xyz_lims[i_ax][0] += reductions[0]
            if reductions[1]:
                xyz_lims[i_ax][1] -= reductions[1]
        else:
            # No reduction to be performed, so set null object extents.
            xyz_lims[i_ax] = None

    # Crop object to new extents.
    obj.crop(*xyz_lims)

def checked_crop_limits(crop_limits=None):
    """
    Check input crop limits, returning (xlim, ylim, zlim) tuple
    that can be passed to skrt.image.Image.crop().

    **Parameter:**

    crop_limits : float/tuple/list/None default=None
        Specification of crop limits:

        - If a float or a one-element tuple or list, crop limits
          along each axis are taken to be minus and plus the 
          value given.

        - If a three-element tuple or list, the three elements are
          taken to correspond to x, y, z limits; an element that is
          a float or a one-element tuple or list is taken to indicate
          limits along the relevant axis of minus and plus the value given.

        - For all other inputs, a value of (None, None, None) is returned.
    """
    # Handle case where input is a single number.
    if isinstance(crop_limits, numbers.Number):
        crop_limits_checked = tuple((-crop_limits, crop_limits)
                for idx in range(3))

    # Handle case where input is a single-element tuple or list.
    elif skrt.core.is_list(crop_limits) and (1 == len(crop_limits)):
        crop_limits_checked = tuple((-crop_limits[0], crop_limits[0])
                for idx in range(3))

    # Handle case where input is a three-element tuple or list.
    elif skrt.core.is_list(crop_limits) and (3 == len(crop_limits)):
        crop_limits_checked = []
        for item in crop_limits:
            if isinstance(item, numbers.Number):
                crop_limits_checked.append((-item, item))
            elif skrt.core.is_list(item) and (1 == len(item)):
                crop_limits_checked.append((-item[0], item[0]))
            else:
                crop_limits_checked.append(item)
        crop_limits_checked = tuple(crop_limits_checked)

    # Accept fallback solution.
    else:
        crop_limits_checked = (None, None, None)

    return crop_limits_checked

def entropy(p, base=2):
    """
    Calculate entropy for variable(s) from probability distribution.

    **Parameters:**

    p : numpy.array
        One-dimensional numpy array representing the probabilities of
        different values of a random variable, or the joint probabilities
        of different combinations ov values of a set of random variables.

    base : int/None, default=2
        Base to use when taking logarithms.  If None, use base e.
    """
    if base is None:
        base = math.e
    p_non_zero = (p > 0)
    return -np.sum(p[p_non_zero] * np.log(p[p_non_zero])) /np.log(base)

def rescale_images(images, v_min=0.0, v_max=1.0, constant=0.5, clone=True):
    """
    For one or more images, linearly rescale greyscale values
    so that they span a specified range.

    Returns the input images if no rescaling is performed.  Otherwise
    returns the input images rescaled (<clone> set to False) or
    rescaled clones of the input images (<clone> set to True).

    image : list
        List of skrt.image.Image objects, for which rescaling
        is to be performed.

        v_min: float, default=0.0
            Minimum greyscale value after rescaling.  If None,
            no rescaling is performed.

        v_max: float, default=1.0
            Maximum greyscale value after rescaling.  If None,
            no rescaling is performed.

        constant: float, default=0.5
            Greyscale value to assign after rescaling if all values
            in the original image are the same.  If None,
            original value is kept.

    clone : bool, default=True
        If True, clone each input image before rescaling.
    """
    if v_min is None or v_max is None:
        return images

    # Try to ensure list of images.
    if not skrt.core.is_list(images):
        images = [images]
    elif not isinstance(images, list):
        images = list(images)

    # Clone if required.
    if clone:
        images = [image.clone() for image in images]

    # Perform rescaling.
    for image in images:
        image.rescale(v_min=v_min, v_max=v_max, constant=constant)

    return images

def get_image_comparison_metrics():
    """
    Get list of image-comparison metrics.

    This returns the combined list of metrics based on mutual information
    (returned by get_mi_metrics()) and quality metrics
    (returned by get_quality_metrics()).
    """
    return (get_mi_metrics() + get_quality_metrics())

def get_mi_metrics():
    """
    Get list of metrics based on mutual information.

    All metrics listed here should be recognised by
    Image.get_mutual_information(), and all metrics recognised by
    Image.get_mutual_information() should be listed here.
    """
    return [
            "iqr",
            "information_quality_ratio",
            "mi",
            "mutual_information",
            "nmi",
            "normalised_mutual_information",
            "rajski",
            "rajski_distance",
            ]

def get_quality_metrics():
    """
    Get list of metrics measuring quality of one metric relative to another.

    All metrics listed here should be recognised by Image.get_quality(),
    and all metrics recognised by Image.get_quality() should be listed here.
    """
    return [
            "correlation_quality",
            "fidelity",
            "relative_structural_content",
            ]
