"""Classes related to ROIs and structure sets."""

from pathlib import Path

from scipy import interpolate, ndimage
from skimage import draw
from shapely import affinity
from shapely import geometry
from shapely import ops
import fnmatch
import logging
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import defaultParams
import numpy as np
import os
import pandas as pd
import pydicom
import re
import shutil
import skimage.measure
import time
import warnings

import skrt.core
import skrt.image
from skrt.dicom_writer import DicomWriter

# Set global defaults.
skrt.core.Defaults({"by_slice": "union"})
skrt.core.Defaults({"slice_stats": ["mean"]})
skrt.core.Defaults({"shapely_log_level": logging.ERROR})

class ROIDefaults:
    """Singleton class for assigning default ROI names and colours."""

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __ROIDefaults:
        def __init__(self):

            self.n_rois = 0
            self.n_structure_sets = 0
            self.n_colors_used = 0

            self.colors = (
                list(matplotlib.cm.Set1.colors)[:-1]
                + list(matplotlib.cm.Set2.colors)[:-1]
                + list(matplotlib.cm.Set3.colors)
                + list(matplotlib.cm.tab20.colors)
            )
            for i in [9, 10]:  # Remove greys
                del self.colors[i]

    def __init__(self, reset=False):
        """Constructor of ROIDefaults singleton class."""

        if not ROIDefaults.instance:
            ROIDefaults.instance = ROIDefaults.__ROIDefaults()
        elif reset:
            ROIDefaults.instance.__init__()

    def get_default_roi_name(self):
        """Get a default name for an ROI."""

        ROIDefaults.instance.n_rois += 1
        return f"ROI {ROIDefaults.instance.n_rois}"

    def get_default_roi_color(self):
        """Get a default roi color."""

        if ROIDefaults.instance.n_colors_used >= len(ROIDefaults.instance.colors):
            return np.random.rand(3)
        color = ROIDefaults.instance.colors[ROIDefaults.instance.n_colors_used]
        ROIDefaults.instance.n_colors_used += 1
        return color

    def get_default_structure_set_name(self):
        """Get a default name for a structure set."""

        ROIDefaults.instance.n_structure_sets += 1
        return f"StructureSet {ROIDefaults.instance.n_structure_sets}"


ROIDefaults()

class ROI(skrt.core.Archive):
    """
    Class representing a single region of interest (ROI).

    Attributes of an ROI object should usually be accessed via
    their getter methods, rather than directly, to ensure that
    attribute values have been loaded.
    """

    def __init__(
        self,
        source=None,
        name=None,
        color=None,
        load=True,
        image=None,
        shape=None,
        affine=None,
        voxel_size=None,
        origin=None,
        mask_threshold=0.25,
        default_geom_method="auto",
        overlap_level=None,
        alpha_over_beta=None,
        **kwargs,
    ):

        """Load ROI from mask or contour.

        **Parameters:**
        
        source : str/array/nifti, default=None
            Source of image data to load. Can be either:

                (a) The path to a nifti file containing a binary mask;
                (b) A numpy array containing a binary mask;
                (c) The path to a file containing a numpy array;
                (d) The path to a dicom structure set file.
                (e) Dictionary of contours in the x-y orientation, where the
                    keys are z positions in mm and values are lists of lists of
                    3D contour points in order (x, y, z) or 2D contour points in
                    order (x, y) on each slice.
                (f) Dictionary of shapely polygons and/or multipolygons
                    in the x-y orientation, where the keys are z positions
                    in  mm and values are lists of polygons/multipolygons
                    on each slice.

        name : str, default=None
            Name of the ROI. If <source> is a file and no name is given,
            the name will be inferred from the filename.

        color : matplotlib color, default=None
            Color in which this ROI will be plotted. If None, a color
            will be assigned.

        load : bool, default=True
            If True, contours/mask will be created during initialisation;
            otherwise they will be created on-demand.

        image : Image/str, default=None
            Associated image from which to extract shape and affine matrix.
            Can either be an existing Image object, or a path from which
            an Image can be created.

        shape : list, default=None
            Number of voxels in the image to which the ROI belongs, in
            order (x, y, z); needed to create mask if contours are provided
            in <source> but no associated <image> is assigned.

        affine : np.ndarray, default=None
            Affine matrix of the image to which the ROI belongs.
            Needed to create mask if contours are provided
            in <source> but no associated <image> is assigned, and no 
            <voxel_size> and <origin> are given. If <affine> is given, it
            supercedes the latter two arguments.

        voxel_size : tuple, default=(1, 1, 1)
            Voxel sizes in mm in order (x, y, z); used if <affine> and <image>
            are not provided.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm in order (x, y, z); used if <affine> and <image>
            are not provided.

        mask_threshold : float, default=0.25
            Used if the ROI is created from a non-boolean pixel array. Values 
            in the array exceeding this level are taken to be inside the ROI
            when it is converted to a boolean array.

        default_geom_method : str, default="auto"
            Default method for computing geometric quantities (area, centroid, etc).
            Can be any of:

                (a) "contour": geometric quantities will be calculated from 
                    Shapely polygons made from contour points.
                (b) "mask": geometric quantities will be calculated from 
                    numpy array representing a binary mask.
                (c) "auto": the "contour" or "mask" method will be used, 
                    depending on whether the input that create the ROI comprised
                    of contours or an array, respectively.

        overlap_level : float, default=None
            Used when converting an input ROI contour to a mask; required 
            overlap of a pixel with the ROI contour in order for the 
            pixel to be added to the mask. If None, the output of 
            skimage.draw.polygon2mask will be returned with no border checks, 
            which is faster than performing border checks.

        
        alpha_over_beta : float, default=None
            Ratio for ROI tissue of coefficients, alpha, beta,
            of linear-quadratic equation.  This ratio is used in
            calculating biologically effective dose, BED, from
            physical dose, D, delivered over n equal fractions:
                BED = D * (1 + (D/n) / (alpha/beta)).
            If None, the biologically effective dose is taken to be
            equal to the physical dose (beta >> alpha).

        kwargs : dict, default=None
            Extra arguments to pass to the initialisation of the parent
            Image object.
        """

        # Clone from another ROI object
        if issubclass(type(source), ROI):
            source.clone_attrs(self)
            return

        # Process contour dictionary
        self.input_contours = None
        if isinstance(source, dict):
            self.source = None
            self.input_contours = {}
            for z, contours in source.items():
                self.input_contours[z] = []
                for contour in contours:
                    if isinstance(contour, geometry.polygon.Polygon):
                        self.input_contours[z].append(
                            np.column_stack(contour.exterior.xy)
                        )
                    elif isinstance(contour,
                            geometry.multipolygon.MultiPolygon):
                        for polygon in contour.geoms:
                            self.input_contours[z].append(
                                np.column_stack(polygon.exterior.xy)
                        )
                    else:
                        self.input_contours[z].append(contour)
        else:
            self.source = (skrt.core.fullpath(source)
                           if isinstance(source, (str, Path)) else source)

        # Assign other properties
        self.custom_color = color is not None
        self.set_color(color)
        self.mask = None
        self.image = image
        if image and not isinstance(image, skrt.image.Image):
            self.image = skrt.image.Image(image)
        self.shape = shape
        self.affine, self.voxel_size, self.origin = \
                skrt.image.get_geometry(affine, voxel_size, origin)
        self.slice_thickness_contours = None
        self.mask_threshold = mask_threshold
        self.overlap_level = overlap_level
        self.contours = {}
        self.default_geom_method = default_geom_method
        self.kwargs = kwargs
        self.source_type = None
        self.dicom_dataset = None
        self.contours_only = False
        self.plans = []
        self.alpha_over_beta = alpha_over_beta

        # Properties relating to plan dose constraints
        self.roi_type = None
        self.constraint = None

        # Create name
        self.name = name
        if self.name is None:
            if isinstance(self.source, str):
                basename = os.path.basename(self.source).split(".")[0]
                name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+", "", basename)
                #  self.name = name[0].upper() + name[1:]
                self.name = name
            else:
                self.name = ROIDefaults().get_default_roi_name()
        self.original_name = name
        self.title = None

        # ROI number as defined in DICOM RTSTRUCT file.
        # Left to None for ROI loaded from non-DICOM source.
        self.number = None

        # Load ROI data
        self.loaded = False
        self.loaded_mask = False
        if load:
            self.load()

        # Initialise as skrt.core.Archive object
        path = self.source if isinstance(self.source, str) else ""
        skrt.core.Archive.__init__(self, path)

    def __eq__(self, other):
        return other is self

    def load(self, force=False):
        """Load ROI from file or source. The loading sequence is as follows:

        1. If self.image is not None, ensure that the image is loaded and 
        assign its geometric properties to this ROI.

        2. Attempt to load data from self.source:

        (a) If this is an Image object, the ROI will be created by applying a 
        threshold to the Image's array. Set self.mask to the thresholded image
        and self.input_type = "mask".
        (b) If self.source is a string, attempt to load from a file
        conforming to transformix output (.txt extension) or from a
        dicom structure set (in both cases setting self.input_contours).
        (c) If transformix or dicom loading fails, attempt to load
        self.source as an Image object to create an ROI mask. If this
        finds a valid Image, set self.mask to that image and set
        self.input_type = "mask".
        (d) If self.source is None, do nothing with it.

        3. Check whether self.input_contours contains data. This could arise
        in two scenarios:
        
        (a) The ROI was initialised from a dict of contours, which were
        assigned to self.input_contours in self.__init(); or
        (b) A set of contours was successfully loaded from a transformix or
        dicom file found at self.source in step 1(b).

        If input contours are found, these are assigned to self.contours["x-y"]
        and self.source_type is set to "contour". Additionally, if self.image
        is not None, set self.mask to an array of zeros the same size as the 
        image.

        4. Set a flag indicating whether this ROI has contours only (i.e no
        mask or image from which to draw origin/voxel sizes), and set
        default geometric calculation method to self.source_type. Set
        self.loaded to True.
        """

        if self.loaded and not force:
            return

        if self.image:
            self.image.load()
            self.shape = self.image.get_data().shape
            self.affine = self.image.get_affine()
            self.voxel_size = self.image.get_voxel_size()
            self.origin = self.image.get_origin()

        rois = []

        # Load from an existing Image
        if issubclass(type(self.source), skrt.image.Image):
            if not self.image:
                self.image = self.source
            self.mask = skrt.image.Image(
                self.source.get_data() > self.mask_threshold,
                affine=self.source.get_affine(),
                dtype=bool
            )
            self.source_type = "mask"
            self.loaded = True
            self.create_mask()

        elif isinstance(self.source, str):
            if self.source.endswith('.txt'):
                # Try loading from transformix-compatible point cloud
                points = self.load_transformix_points()

                # Extract slice-by-slice dictionary of (x, y) points
                contours = {}
                for point in sorted(points):
                    x_point, y_point, z_point = points[point]['OutputPoint']
                    key = f'{z_point:.2f}'
                    if not key in contours:
                        contours[key] = {}
                    c_index = points[point]['ContourIndex']
                    if not c_index in contours[key]:
                        contours[key][c_index] = []
                    contours[key][c_index].append([x_point, y_point])

                # Store the list of contours.
                self.input_contours = {}
                for key in sorted(contours):
                    self.input_contours[float(key)] = []
                    for c_index in sorted(contours[key]):
                        self.input_contours[float(key)].append(
                            np.array(contours[key][c_index]))
                rois.append(self.name)

            elif os.path.isdir(self.source) or not self.source.endswith('.nii'):
                # Try loading from dicom structure set
                rois, ds = load_rois_dicom(self.source, names=self.name)
                if len(rois):
                    number = list(rois.keys())[0]
                    roi = rois[number]
                    self.name = roi["name"]
                    self.number = number
                    self.input_contours = roi["contours"]
                    if not self.custom_color:
                        self.set_color(roi["color"])
                    self.dicom_dataset = ds

        # Load ROI mask
        if not self.loaded and not len(rois) and self.source is not None:
            self.mask = skrt.image.Image(
                self.source, 
                affine=self.affine, 
                voxel_size=self.voxel_size, 
                origin=self.origin, 
                dtype=bool,
                **self.kwargs
            )
            self.loaded = True
            self.source_type = "mask"
            self.create_mask()

        # Deal with input contours
        if self.input_contours is not None:

            # Create Image object
            if self.image is not None:
                self.shape = self.image.data.shape
                self.mask = skrt.image.Image(
                    np.zeros(self.shape), 
                    affine=self.image.get_affine(), 
                    dtype=bool,
                    **self.kwargs
                )

            # Set x-y contours with z positions as keys
            self.contours["x-y"] = {}
            for z, contours in self.input_contours.items():
                self.contours["x-y"][z] = [contour[:, :2] for contour in contours]
            self.source_type = "contour"
            self.loaded = True

        # Store flag for cases with no associated image or geometric info
        has_image = self.image is not None
        has_geom = self.shape is not None and (
            self.affine is not None or (
                self.voxel_size is not None and self.origin is not None
            ))
        self.contours_only = self.source_type == "contour" \
                and not has_image and not has_geom
        if self.default_geom_method == "auto":
            self.default_geom_method = self.source_type

        # Store flag for emptiness
        if self.source_type == "contour":
            self.empty = not len(self.input_contours)
            self.slice_thickness_contours = self.get_slice_thickness_contours()
        elif self.source_type == "mask":
            self.empty = not np.any(self.mask.get_data())
        else:
            self.empty = True

        self.loaded = True

    def _load_from_file(self, filename):
        """Attempt to load ROI from a dicom or nifti file."""

    def load_transformix_points(self):
        '''
        Load point coordinates from file of the format produced by Transformix.
        '''
        def get_coordinates(in_data=''):
            '''
            Helper function, to unpack string of data into list of coordinates. 
            '''
            coordinates = [eval(s)
                    for s in re.findall(r'[-\d\.]+', in_data)]
            return coordinates

        # Read file.
        with open(self.source, 'r', encoding='ascii') as in_file:
            data_rows = in_file.readlines()

        # Extract point data to dictionary.
        points = {}
        for row in data_rows:
            point, input_index, input_point, output_index_fixed, \
                    output_point, deformation = row.split(';')

            point = int(point.split()[-1])
            points[point] = {}
            points[point]['InputIndex'] = get_coordinates(input_index)
            points[point]['InputPoint'] = get_coordinates(input_point)
            points[point]['OutputIndexFixed'] = get_coordinates(
                    output_index_fixed)
            points[point]['OutputPoint'] = get_coordinates(output_point)
            points[point]['Deformation'] = get_coordinates(deformation)

            # If the Transformix input was written using ROI.write(),
            # a point z-coordinate will have six digits after the decimal
            # point, with the contour index in the last three.
            # (Input points are written to the Transformix output
            # with six digits after the decimal point for all coordinates.)
            zd_in = re.findall(r'[-\d\.]+', input_point)[-1].rsplit(".", 1)[-1]
            if len(zd_in) == 6:
                points[point]["ContourIndex"] = int(zd_in[3:])
            else:
                points[point]["ContourIndex"] = 0

        # After application of a registration transform, points originally
        # in the same may end up with slightly different z-coordinates.
        # For most purposes (and slice thicknesses) it's probably
        # a reasonable approximation to map all points to the mean
        # z-coordinate of all points originally in the same plane.
        keep_original_planes = getattr(self, 'keep_original_planes', True)
        if keep_original_planes:
            # Sort points by slice z-coordinate,
            # keeping relative order within a slice.
            slices = {}
            for point in sorted(points):
                i, j, k = points[point]["InputIndex"]
                x, y, z = points[point]["OutputPoint"]
                if k not in slices:
                    slices[k] = {'points': [], 'z_points': []}
                slices[k]["points"].append(point)
                slices[k]["z_points"].append(z)

            # For points in each slice, reset z-coordinates to mean value.
            for k in sorted(slices):
                z_mean = np.mean(slices[k]["z_points"])
                for point in slices[k]["points"]:
                    x, y, z = points[point]["OutputPoint"]
                    points[point]["OutputPoint"] = (x, y, z_mean)

        return points

    def clone_attrs(self, obj, copy_data=True):
        """Assign all attributes of <self> to another object, <obj>,  ensuring 
        that own mask's data gets correctly copied if <copy_data> is True."""

        skrt.core.Data.clone_attrs(self, obj, copy_data=copy_data)
        if copy_data and self.mask is not None:
            obj.mask = self.mask.clone(copy_data=True)

    def get_dicom_dataset(self):
        """Return pydicom.dataset.FileDataset object associated with this ROI
        if loaded from dicom; otherwise, return None."""

        self.load()
        return self.dicom_dataset

    def reset_contours(self, contours=None, most_points=False):
        """Reset x-y contours to a given dict of slices and contour lists, 
        and ensure that mask and y-z/x-z contours will be recreated. If 
        contours is None, contours will be reset using own x-y contours.
        If most_points is True, only the contour with most points for
        each slice is considered."""

        # Check format is correct
        if contours is None:
            contours = self.get_contours()
        if not isinstance(contours, dict):
            raise TypeError("contours should be a dict")
        for z, c in contours.items():
            if not isinstance(c, list):
                raise TypeError(f"contours[{z}] should be a list of contours "
                                f"on slice {z}")

        # For each slice, keep only contour with most points.
        if most_points:
            contours = {key: [max(value, key=len)]
                    for key, value in contours.items()}

        self.input_contours = contours
        self.empty = not len(self.input_contours)
        self.contours = {"x-y": contours}
        self.loaded_mask = False
        self.mask = None

    def reset_mask(self, mask=None):
        """Set mask to either a numpy array or an Image object, and ensure 
        that contours will be recreated. If mask is None, contours will be 
        reset using own mask."""

        if isinstance(mask, skrt.image.Image):
            self.mask = mask
            self.affine = mask.get_affine()
            self.voxel_size = mask.get_voxel_size()
            self.origin = mask.get_origin()
            self.shape = mask.get_data().shape

        elif isinstance(mask, np.ndarray):
            self.mask = skrt.image.Image(
                mask,
                affine = self.affine,
                voxel_size=self.voxel_size,
                origin=self.origin,
                dtype=bool
            )

        elif mask is not None:
            raise TypeError("mask should be an Image or a numpy array")

        self.mask.get_standardised_data(force=True)
        self.loaded_mask = True
        self.contours = {}
        self.empty = not np.any(self.mask.get_data())
        self.input_contours = None

    def get_contours(self, view="x-y", idx_as_key=False, most_points=False):
        """
        Get dict of contours in a given orientation.

        **Parameters:**

        view : str, default="x-y"
            View in which to obtain contours.

        idx_as_key : bool, default=False
            If True, use slice indices as dictionary keys; if False,
            use slice z-coordinates as dictionary keys.

        most_points : bool, default=False
            If True, return only the contour with most points for each slice;
            if False, return all contours for each slice.
        """

        self.load()
        if view not in self.contours:
            self.create_contours(view)
        contours = self.contours[view]

        # Convert keys to indices rather than positions
        if idx_as_key:
            contours = {self.pos_to_idx(key, skrt.image._slice_axes[view]): 
                        value for key, value in contours.items()}

        # For each slice, keep only contour with most points.
        if most_points:
            contours = {key: [max(value, key=len)]
                    for key, value in contours.items()}

        return contours

    def get_contours_on_slice(self, view="x-y", sl=None, idx=None, pos=None):

        self.load()
        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)
        try:
            return self.get_contours(view, idx_as_key=True)[idx]
        except KeyError:
            return []

    def get_slice_positions(self, other=None, view="x-y", position_as_idx=False,
            method="union"):
        """
        Get ordered list of slice positions for self and other.

        **Parameters:**

        other : skrt.structures.ROI, default=None
            Second ROI of a pair for which slice positions are be obtained.
            Disregarded if None.

        view : str, default="x-y"
            View in which to obtain slices.

        position_as_idx : bool, default=False
            If True, return positions as slice indices; if False,
            return positions as slice z-coordinates (mm).

        method: str, default="union"
            String specifying slices for which positions are to be obtained,
            for ROIs self and other.

            - "left" (or other is None): return positions of
              slices containing self;
            - "right": return positions of slices containing other;
            - "union": return positions of slices containing either self
              or other;
            - "intersection": return positions of slices containing both
              self and other.
        """
        return get_slice_positions(self, other, view, position_as_idx,
                method)

    def get_affine(self, force_standardise=False, **kwargs):
        """Load self and get affine matrix."""

        self.load()
        if self.loaded_mask:
            return self.mask.get_affine(force_standardise=force_standardise, 
                                        **kwargs)
        elif self.image is not None:
            return self.image.get_affine(force_standardise=force_standardise,
                                         **kwargs)
        return self.affine

    def get_translation_to_align(
            self, other, z_fraction1=None, z_fraction2=None):
        """
        Determine translation for aligning <self> to <other>.

        **Parameters:**

        other : ROI/StructureSet/list
            ROI with which alignment is to be performed.  This can be
            specified directly as a single ROI.  Alternatively, it can
            be a StructureSet, or a list or ROI/StructureSet objects,
            in which case the individual ROIs will be combined.

        z_fraction1 : float, default=None
            Position along z axis of slice through <self> on which
            to align.  If None, alignment is to the centroid of the
            whole ROI volume.  Otherwise, alignment is to the
            centroid of the slice at the specified distance
            from the ROI's most-inferior point: 0 corresponds to
            the most-inferior point (lowest z); 1 corresponds to the
            most-superior point (highest z).  Values for z_fraction
            outside the interval [0, 1] result in a RuntimeError.

        z_fraction2 : float, default=None
            Position along z axis of slice through <other> on which
            to align.  If None, alignment is to the centroid of the
            whole ROI volume.  Otherwise, alignment is to the
            centroid of the slice at the specified distance
            from the ROI's most-inferior point: 0 corresponds to
            the most-inferior point (lowest z); 1 corresponds to the
            most-superior point (highest z).  Values for z_fraction
            outside the interval [0, 1] result in a RuntimeError.
            """
        return get_translation_to_align(
                self, other, z_fraction1, z_fraction2)

    def get_voxel_size(self):
        """Load self and get voxel_size."""

        self.load()
        return self.voxel_size

    def get_np_voxel_size(self, view=None):
        """
        Get voxel_size for numpy axis ordering and specified view.

        view : str
            Orientation considered for 2D image or projection.
            Can be "x-y", "y-z", or "x-z".
        """

        self.load()
        if view is None:
            vx, vy, vz = self.voxel_size
            voxel_size = [vy, vx, vz]
        else:
            voxel_size = [self.voxel_size[i]
                    for i in skrt.image._plot_axes[view]]

        return voxel_size

    def get_origin(self):
        """Load self and get origin."""

        self.load()
        return self.origin

    def get_name(self, original=False):
        """
        Load self and get name. If <original> is True, get the original name.
        """

        self.load()
        return (self.original_name if original else self.name)

    def get_mask(self, view="x-y", flatten=False, standardise=False, 
                 force_standardise=False):
        """Get binary mask, optionally flattened in a given orientation."""

        self.load()
        self.create_mask()
        if not flatten:
            return self.mask.get_data(standardise=standardise, 
                                      force_standardise=force_standardise)
        return np.sum(
            self.mask.get_data(standardise=standardise, 
                               force_standardise=force_standardise),
            axis=skrt.image._slice_axes[view]
        ).astype(bool)

    def get_mask_image(self):
        """Get image object representing ROI mask."""

        self.load()
        self.create_mask()
        return self.mask

    def get_polygons(self, view="x-y", idx_as_key=False):
        """Get dict of polygons for each slice."""

        polygons = {}
        for i, contours in self.get_contours(view, idx_as_key).items():
            polygons[i] = [contour_to_polygon(c) for c in contours]
        return polygons

    def get_polygons_on_slice(self, view="x-y", sl=None, idx=None, pos=None):
        """Get shapely polygon objects corresponding to a given slice."""

        contours = self.get_contours_on_slice(view, sl, idx, pos)
        return [contour_to_polygon(c) for c in contours]

    def create_contours(self, view="all", force=False):
        """Create contours in all orientations."""

        if not force:
            if view == "all" and all([v in self.contours for v in 
                                      skrt.image._plot_axes]):
                return
            if view in self.contours:
                return

        if not self.loaded:
            self.load()

        self.create_mask()
        if force:
            self.contours = {}

        # Create contours in every orientation
        views = list(skrt.image._plot_axes.keys()) if view == "all" else [view]
        for v in views:

            if v in self.contours:
                continue

            # Make new contours from mask
            z_ax = skrt.image._slice_axes[v]
            self.contours[v] = {}
            for iz in self.get_indices(v, method="mask"):

                # Get slice of mask array
                mask_slice = self.get_slice(v, idx=iz).T
                if mask_slice.max() < 0.5:
                    continue

                points = self.mask_to_contours(mask_slice, v)
                if points:
                    self.contours[v][self.idx_to_pos(iz, z_ax)] = points

    def mask_to_contours(self, mask, view, invert=False):
        """Create contours from a mask."""

        contours = skimage.measure.find_contours(mask, 0.5, "low", "low")

        # Convert indices to positions in mm
        x_ax, y_ax = skrt.image._plot_axes[view]
        points = []
        for contour in contours:
            contour_points = []
            for ix, iy in contour:
                px = self.idx_to_pos(ix, x_ax)
                py = self.idx_to_pos(iy, y_ax)
                if invert:
                    px, py = py, px
                contour_points.append([px, py])
            points.append(np.array(contour_points))

        return points

    def create_mask(
        self, 
        force=False, 
        overlap_level=None, 
        voxel_size=None, 
        shape=None
    ):
        """Create binary mask representation of this ROI. If the ROI was created
        from contours, these contours will be converted to a mask; if the ROI
        was created from a mask, this mask will be cast to a boolean array and
        its geoemtric properties will be assigned to this ROI. If a mask has 
        already been created, nothing will happen unless force=True.

        **Parameters:**
        
        force : bool, default=False
            If True, the mask will be recreated even if it already exists.

        overlap_level : float, default=0.25
            Required overlap of a pixel with the ROI contour in order for the 
            pixel to be added to the mask. If None, the value of
            self.overlap_level will be used; otherwise,
            self.overlap_level will be overwritten with this value. If both are 
            None, the output of skimage.draw.polygon2mask will be returned 
            with no border checks, which is faster than performing border checks.

        voxel_size : list, default=None
            Voxel sizes of the desired mask in the [x, y] direction. Only used if 
            the ROI was created from contours; ignored if ROI was created from 
            mask. Causes self.image to be replaced with a 
            blank dummy image with these voxel sizes.

        shape : list, default=None
            Shape of the desired mask in the [x, y] direction. Only used if 
            <voxel_size> is None and the ROI was created from contours; 
            ignored if ROI was created from mask. Causes self.image to be 
            replaced with a blank dummy image of this shape.
        """

        if self.loaded_mask and not force:
            return

        if not self.loaded:
            self.load()
        if overlap_level is not None:
            self.overlap_level = overlap_level

        # Set image to dummy image if needed
        if (force and (shape is not None or voxel_size is not None)) \
           or (self.contours_only and self.image is None):
            self.set_image_to_dummy(shape=shape, voxel_size=voxel_size)

        # Create mask from input x-y contours 
        if self.input_contours:

            # Initialise self.mask as image
            self.mask = skrt.image.Image(
                np.zeros((self.shape[1], self.shape[0], self.shape[2]),
                    dtype=bool),
                affine=self.affine, 
                voxel_size=self.voxel_size,
                origin=self.origin
            )

            # Create mask on each z layer
            for z, contours in self.input_contours.items():

                # Loop over each contour on the z slice
                iz = int(self.pos_to_idx(z, "z"))
                if iz >= self.mask.data.shape[2]:  # Ignore slices outside mask range
                    continue

                pos_to_idx_vec = np.vectorize(self.pos_to_idx)
                for points in contours:

                    # Require at least 3 points to define a polygon.
                    if len(points) < 3:
                        continue

                    # Convert (x, y) positions to array indices
                    points_idx = np.zeros((points.shape[0], 2))
                    for i in range(2):
                        points_idx[:, i] = pos_to_idx_vec(
                            points[:, i], i, return_int=False
                        )

                    # Create polygon in index space
                    polygon = contour_to_polygon(points_idx)

                    # Get mask of all pixels inside contour
                    mask = draw.polygon2mask([self.mask.data.shape[1], 
                                             self.mask.data.shape[0]],
                                             points_idx)

                    # If no pixel identified as being inside contour,
                    # label pixel containing contour centroid.
                    if not mask.sum():
                        ix, iy = [int(round(xy[0]))
                                for xy in polygon.centroid.xy]
                        if ((ix >= 0 and ix < mask.shape[0])
                                and (iy >= 0 and iy < mask.shape[1])):
                            mask[ix, iy] = True

                    # Check overlap of edge pixels
                    if self.overlap_level is not None:
                        conn = ndimage.generate_binary_structure(
                            2, 2)
                        edge = mask ^ ndimage.binary_dilation(
                            mask, conn)
                        if self.overlap_level >= 0.5:
                            edge += mask ^ ndimage.binary_erosion(
                                mask, conn)

                        # Check whether each edge pixel has sufficient overlap
                        for ix, iy in np.argwhere(edge):

                            # Make polygon of current pixel
                            pixel = geometry.Polygon(
                                [
                                    [ix - 0.5, iy - 0.5],
                                    [ix - 0.5, iy + 0.5],
                                    [ix + 0.5, iy + 0.5],
                                    [ix + 0.5, iy - 0.5],
                                ]
                            )

                            # Compute overlap
                            overlap = polygon.intersection(pixel).area
                            mask[ix, iy] = overlap > self.overlap_level

                    # Add to 3D mask
                    try:
                        self.mask.data[:, :, iz] += mask.T
                    except IndexError:
                        pass

        # Convert to boolean mask
        if hasattr(self.mask, "data"):
            if not self.mask.data.dtype == bool:
                self.mask.data = self.mask.data.astype(bool)
            self.loaded_mask = True

        # Set own geometric properties from mask
        self.voxel_size = self.mask.get_voxel_size()
        self.origin = self.mask.get_origin()
        self.affine = self.mask.get_affine()
        self.shape = self.mask.get_data().shape

    def is_empty(self):
        """Check whether this ROI is empty."""

        self.load()
        if self.default_geom_method == "contour":
            return not bool(len(self.get_contours("x-y")))
        else:
            return not np.any(self.get_mask())

    def resample(self, *args, **kwargs):
        self.create_mask()
        self.mask.resample(*args, **kwargs)

    def match_voxel_size(self, other, *args, **kwargs):

        if isinstance(other, ROI):
            other.create_mask()
            mask = other.mask
        else:
            mask = other
        self.create_mask()
        self.mask.match_voxel_size(mask, *args, **kwargs)

    def get_slice(self, *args, force_standardise=False, **kwargs):

        self.create_mask()
        data = self.mask.get_slice(*args, force_standardise=force_standardise, 
                                   **kwargs)
        if data.dtype != bool:
            data = data.astype(bool)
        return data

    def get_roi_slice(self, z_fraction=1, suffix=None):
        """
        Get ROI object corresponding to x-y slice through this ROI.

        **Parameter:**

        z_fraction : float, default=1
            Position along z axis at which to take slice through ROI.
            The position is specified as the fractional distance
            from the ROI's most-inferior point: 0 corresponds to
            the most-inferior point (lowest z); 1 corresponds to the
            most-superior point (highest z).  Values for z_fraction
            outside the interval [0, 1] result in a RuntimeError.

    suffix : str, default=None
        Suffix to append to own name, to form name of output ROI.  If
        None, append z_fraction, with value given to two decimal places.
        """
        return get_roi_slice(self, z_fraction)

    def get_indices(self, view="x-y", method=None):
        """Get list of slice indices on which this ROI exists."""

        self.load()
        if self.empty:
            return []

        if method is None:
            method = self.default_geom_method

        if method == "contour":
            return list(self.get_contours(view, idx_as_key=True).keys())
        else:
            ax = skrt.image._slice_axes[view]
            mask = self.get_mask(standardise=True).transpose(1, 0, 2)
            return list(np.unique(np.argwhere(mask)[:, ax]))

    def get_mid_idx(self, view="x-y", **kwargs):
        """Get central slice index of this ROI in a given orientation."""

        indices = self.get_indices(view, **kwargs)
        if not len(indices):
            return None
        mid = int(len(indices) / 2)
        return indices[mid]

    def idx_to_pos(self, idx, ax, standardise=True):
        """Convert an array index to a position in mm along a given axis."""

        self.load()

        if self.contours_only:

            # Use self.image's conversion function
            if self.image is not None:
                return self.image.idx_to_pos(idx, ax)

            # Otherwise count number of z slices in contours dict
            # If index does not correspond to a slice, return None
            elif ax in ["z", 2]:
                conversion = {i: p for i, p in 
                              enumerate(self.get_contours("x-y").keys())}
                return conversion.get(idx, None)

            # Otherwise, not possible to convert x/y points to indices
            # from contours alone
            else:
                print("Warning: cannot convert index to position in the x/y "
                      "directions without associated image geometry!")
                return

        # Otherwise, try to use mask or image's conversion
        elif self.mask is not None:
            return self.mask.idx_to_pos(idx, ax)
        elif self.image is not None:
            return self.image.idx_to_pos(idx, ax)

        # Else, try to calculate from own geometric properties
        else:
            i_ax = skrt.image._axes.index(ax) if ax in skrt.image._axes else ax
            return self.origin[i_ax] + idx * self.voxel_size[i_ax]

    def pos_to_idx(self, pos, ax, return_int=True, **kwargs):
        """Convert a position in mm to an array index along a given axis."""

        self.load()
        if self.contours_only:

            # Use self.image's conversion function
            if self.image is not None:
                return self.image.pos_to_idx(pos, ax, return_int)

            # Otherwise count number of z slices in contours dict
            # If position does not correspond to a slie, return None
            elif ax in ["z", 2]:
                conversion = {p: i for i, p in 
                              enumerate(self.get_contours("x-y").keys())}
                idx = conversion.get(pos, None)

            # Otherwise, not possible to convert x/y points to indices
            # from contours alone
            else:
                print("Warning: cannot convert position to index in the x/y "
                      "directions without associated image geometry!")
                return
            if return_int and idx is not None:
                return round(idx)
            else:
                return idx

        # Otherwise, try to use mask or image's conversion
        elif self.mask is not None:
            return self.mask.pos_to_idx(pos, ax, return_int)
        elif self.image is not None:
            return self.image.pos_to_idx(pos, ax, return_int)

        # Else, try to calculate from own geomtric properties
        else:
            i_ax = skrt.image._axes.index(ax) if ax in skrt.image._axes else ax
            idx = (pos - self.origin[i_ax]) / self.voxel_size[i_ax]
            if return_int:
                return round(idx)
            else:
                return idx

    def idx_to_slice(self, idx, ax):
        """Convert an array index to a slice number along a given axis."""

        self.load()
        if self.contours_only or (self.mask is None and self.image is None):

            # Count number of z slices in contours dict
            if ax in ["z", 2]:
                nz = self.get_nz_contours()
                return nz - idx
            else:
                return idx + 1

        # Try to use mask or image's conversion
        elif self.mask is not None:
            return self.mask.idx_to_slice(idx, ax)
        else:
            return self.image.idx_to_slice(idx, ax)

    def slice_to_idx(self, sl, ax):
        """Convert a slice number to an array index."""

        self.load()
        if self.contours_only or (self.mask is None and self.image is None):

            # Count number of z slices in contours dict
            if ax in ["z", 2]:
                nz = self.get_nz_contours()
                return nz - sl
            else:
                return sl - 1

        # Try to use mask or image's conversion
        elif self.mask is not None:
            return self.mask.slice_to_idx(sl, ax)
        else:
            return self.image.slice_to_idx(sl, ax)

    def pos_to_slice(self, pos, ax, return_int=True, standardise=True):
        """Convert a position in mm to a slice number along a given axis."""

        return skrt.image.Image.pos_to_slice(self, pos, ax, return_int)

    def slice_to_pos(self, sl, ax, standardise=True):
        """Convert a slice number to a position in mm along a given axis."""

        return skrt.image.Image.slice_to_pos(self, sl, ax)

    def get_idx(self, view, sl=None, idx=None, pos=None):
        """Get an array index from either a slice number, index, or
        position."""

        return skrt.image.Image.get_idx(self, view, sl=sl, idx=idx, pos=pos)

    def on_slice(self, view, sl=None, idx=None, pos=None):
        """Check whether this ROI exists on a given slice."""

        if sl is None and idx is None and pos is None:
            return False
        idx = self.get_idx(view, sl, idx, pos)
        if idx is None:
            return False
        return idx in self.get_indices(view)

    def interpolate_points(self, n_point=None, dxy=None,
        smoothness_per_point=0):
        '''
        Return new ROI object, with interpolated contour points.

        **Parameters:**

        n_point : int, default=None
            Number of points per contour, after interpolation.  This must
            be set to None for dxy to be considered.

        dxy : float, default=None
            Approximate distance required between contour points.  This is taken
            into account only if n_point is set to None.  For a contour of
            length contour_length, the number of contour points is then taken
            to be max(int(contour_length / dxy), 3).  

        smoothness_per_point : float, default=0
            Parameter determining the smoothness of the B-spline curve used
            in contour approximation for interpolation.  The product of
            smoothness_per_point and the number of contour points (specified
            directly via n_point, or indirectly via dxy) corresponds to the
            parameter s of scipy.interpolate.splprep - see documentation at:
        
            https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep

            A smoothness_per_point of 0 forces the B-spline to pass through
            all of the pre-interpolation contour points.
        '''

        new_contours = {}
        for z, contours in self.get_contours().items():
            new_contours[z] = []
            for contour in contours:
                new_contour = interpolate_points_single_contour(
                        contour, n_point, dxy, smoothness_per_point)
                new_contours[z].append(new_contour)

        # Clone self, then clear mask and reset contours of clone
        roi = ROI(self)
        roi.input_contours = new_contours
        roi.contours = {"x-y": new_contours}
        roi.loaded_mask = False
        roi.mask = None

        return roi

    def get_centroid(
        self,
        view="x-y",
        single_slice=False,
        sl=None,
        idx=None,
        pos=None,
        units="mm",
        method=None,
        force=True,
    ):
        """Get either 3D global centroid position or 2D centroid position on 
        a single slice. If single_slice=False, the calculated global centroid 
        will be cached in self._centroid[units] and returned if called again,
        unless force=True.

        **Parameters:**
        
        single_slice : bool, default=False
            If False, the global 3D centroid of the entire ROI will be returned;
            otherwise, the 2D centroid on a single slice will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get centroid. Only used if 
            single_slice=True. If using, <ax> must be an axis that lies along
            the slice in this orientation.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        units : str, default="mm"
            Units of centroid. Can be either of:
                - "mm": return centroid position in millimetres.
                - "voxels": return centroid position in terms of array indices.
                - "slice": return centroid position in terms of slice numbers.

            If units="voxels" or "slice" is requested but this ROI only has 
            contours and no mask shape/voxel size information, an error will be 
            raised (unless ax="z", in which case voxel size will be inferred 
            from spacing between slices).

        method : str, default=None
            Method to use for centroid calculation. Can be: 
                - "contour": get centroid of shapely contour(s).
                - "mask": get average position of voxels in binary mask.
                - None: use the method set in self.default_geom_method.

        force : bool, default=True
            If True, the global centroid will always be recalculated; 
            otherwise, it will only be calculated if it has not yet been cached 
            in self._volume.  Note that if single_slice=True, the centroid will 
            always be recalculated.
        """

        # If global centroid already cached, return cached value
        if not single_slice and not force:
            if hasattr(self, "_centroid") and units in self._centroid:
                return self._centroid[units]

        # Get default slice and method
        self.load()
        if method is None:
            method = self.default_geom_method
        if single_slice and pos is None and sl is None and idx is None:
            idx = self.get_mid_idx(view)

        # If single slice requested and not on current slice, return None, None
        if single_slice and not self.on_slice(view, sl=sl, idx=idx, pos=pos):
            return np.array([None, None])

        # Calculate centroid from Shapely polygons
        centroid = {}
        if method == "contour":

            # Get 2D or 3D centroids and areas of each polygon on the desired
            # slice(s)
            if single_slice:
                polygons = self.get_polygons_on_slice(view, sl, idx, pos)
                centroids = []
                areas = []
                for p in polygons:
                    centroid_xy = p.centroid.xy
                    centroids.append(np.array([
                        centroid_xy[0][0], centroid_xy[1][0]
                    ]))
                    areas.append(p.area)
            else:
                centroids = []
                areas = []
                polygon_dict = self.get_polygons()
                for z, polygons in polygon_dict.items():
                    for p in polygons:
                        centroid_xy = p.centroid.xy
                        centroids.append(np.array([
                            centroid_xy[0][0], centroid_xy[1][0], z
                        ]))
                        areas.append(p.area)

            # Get weighted average of polygon centroids
            weighted_sum = np.sum(
                [centroids[i] * areas[i] for i in range(len(centroids))],
                axis=0
            )
            centroid["mm"] = weighted_sum / sum(areas)

            # Calculate in voxels and slices if possible
            if self.voxel_size is None and self.shape is None:
                if units in ["voxels", "slice"]  and self.shape is None:
                    raise RuntimeError("Cannot compute centroid in voxels/slice"
                                       " numbers from contours without knowing voxel"
                                       " sizes and mask shape!")
            else:
                view_axes = ([0, 1, 2] if len(centroid['mm']) == 3 else
                        skrt.image._plot_axes[view])

                centroid["voxels"] = np.array([
                    self.pos_to_idx(c, ax=i, return_int=False) 
                    for i, c in zip(view_axes, centroid["mm"])
                ])

                centroid["slice"] = np.array([
                    self.pos_to_slice(c, ax=i, return_int=False) 
                    for i, c in zip(view_axes, centroid["mm"])
                ])

        # Otherwise, calculate centroid from binary mask
        else: 

            # Get 2D or 3D data from which to calculate centroid
            if single_slice:
                if not self.on_slice(view, sl=sl, idx=idx, pos=pos):
                    return np.array([None, None])
                data = self.get_slice(view, sl=sl, idx=idx, pos=pos)
                axes = skrt.image._plot_axes[view]
            else:
                self.create_mask()
                data = self.get_mask(standardise=True)
                axes = skrt.image._axes

            # Compute centroid from 2D or 3D binary mask
            non_zero = np.argwhere(data)
            if not len(non_zero):
                if data.ndim == 2:
                    return None, None
                else:
                    return None, None, None
            centroid_rowcol = list(non_zero.mean(0))
            centroid_voxels = [centroid_rowcol[1], centroid_rowcol[0]] \
                    + centroid_rowcol[2:]
            centroid["voxels"] = np.array(centroid_voxels)

            # Convert to mm and slices
            view_axes = ([0, 1, 2] if len(centroid['voxels']) == 3 else
                    skrt.image._plot_axes[view])

            centroid["mm"] = np.array([
                self.idx_to_pos(c, ax=i)
                for i, c in zip(view_axes, centroid["voxels"])
            ])

            centroid["slice"] = np.array([
                self.idx_to_slice(c, ax=i)
                for i, c in zip(view_axes, centroid["voxels"])
            ])

        # Cache global centroid
        if not single_slice:
            self._centroid = centroid

        # Return centroid in requested units
        return centroid[units]

    def set_slice_thickness_contours(self, dz):
        """
        Set nominal z spacing of contours.

        This may be useful in particular for  single-slice ROIs.
        """

        self.slice_thickness_contours = dz

    def get_slice_thickness_contours(self):
        """Get z voxel size using positions of contours."""

        contours = self.get_contours("x-y")
        z_keys = sorted(contours.keys())
        if len(z_keys) < 2:
            if self.source_type == "contour" and self.slice_thickness_contours:
                return self.slice_thickness_contours
            else:
                voxel_size = self.get_voxel_size()
                if voxel_size is not None:
                    return voxel_size[2]
                return
        diffs = [z_keys[i] - z_keys[i - 1] for i in range(1, len(z_keys))]
        return min(diffs)

    def get_nz_contours(self):
        """Get number of voxels in the z direction using positions of contours."""

        vz = self.get_slice_thickness_contours()
        z_keys = list(self.get_contours("x-y").keys())
        return int((max(z_keys) - min(z_keys)) / vz) + 1

    def get_centre(
        self, 
        view="x-y", 
        single_slice=False,
        sl=None, 
        idx=None, 
        pos=None, 
        method=None,
    ):
        """Get centre position in 2D or 3D in mm."""

        # Get default slice index and method
        self.load()
        if sl is None and idx is None and pos is None: 
            idx = self.get_mid_idx(view)
        if method is None:
            method = self.default_geom_method

        # Get list of axes to consider
        if single_slice:
            axes = skrt.image._plot_axes[view]
        else:
            axes = skrt.image._axes

        # Get centre in mm
        centre = []
        for ax in axes:
            extent = self.get_extent(
                ax=ax,
                single_slice=single_slice,
                view=view,
                sl=sl,
                idx=idx,
                pos=pos,
                method=method,
            )
            if None in extent:
                centre.append(None)
            else:
                centre.append(np.mean(extent))

        return np.array(centre)

    def get_volume(
        self, 
        units="mm", 
        method=None, 
        force=True
    ):
        """Get ROI volume. The calculated volume will be cached in 
        self._volume[units] and returned if called again, unless force=True.

        **Parameters:**
        
        units : str, default="mm"
            Units of volume. Can be any of:
                - "mm": return volume in millimetres cubed.
                - "ml": return volume in millilitres.
                - "cc": return volume in cubic centimetres (equivalent to "ml")
                - "voxels": return volume in number of voxels.

            If units="voxels" is requested but this ROI only has contours and no
            voxel size information, an error will be raised.

        method : str, default=None
            Method to use for volume calculation. Can be: 
                - "contour": compute volume by summing areas of shapely
                  polygons on each slice and multiplying by slice thickness.
                - "mask": compute volume by summing voxels inside the ROI and
                  multiplying by the volume of one voxel.
                - None: use the method set in self.default_geom_method.

            Note that if self.get_volume has already been called with one 
            method, the same cached result will be returned if calling with a 
            different method unless force=True.

        force : bool, default=True
            If True, the volume will always be recalculated; otherwise, it will
            only be calculated if it has not yet been cached in self._volume.
        """

        # If already cached and not forcing, return
        if hasattr(self, "_volume") and not force:
            if units in self._volume:
                return self._volume[units]

        self.load()
        if method is None:
            method = self.default_geom_method

        # Make cached property
        self._volume = {}

        # Calculate from polygon areas
        if method == "contour":

            # Check it's possible to calculate volume with the requested units
            if self.voxel_size is None and units == "voxels":
                raise RuntimeError("Cannot compute volume in voxels from "
                                   "contours without knowing voxel sizes!")

            # Calculate area on each slice
            slice_areas = []
            for idx in self.get_contours("x-y", idx_as_key=True):
                slice_areas.append(self.get_area("x-y", idx=idx, method=method))

            # Convert to volume in mm
            area_sum = sum(slice_areas)
            self._volume["mm"] = area_sum * self.get_slice_thickness_contours()

            # If possible, convert to volume in voxels
            if self.voxel_size is not None:
                xy_voxel_area = self.voxel_size[0] * self.voxel_size[1]
                self._volume["voxels"] = area_sum / xy_voxel_area

        # Otherwise, calculate from number of voxels in mask
        else:
            self.create_mask()
            self._volume["voxels"] = self.mask.data.astype(bool).sum()
            voxel_vol = abs(np.prod(self.get_voxel_size()))
            self._volume["mm"] = self._volume["voxels"] * voxel_vol

        # Get volume in ml or equivalently cc
        self._volume["ml"] = self._volume["mm"] / 1000
        self._volume["cc"] = self._volume["ml"]

        # Return volume in the requested units
        return self._volume[units]

    def get_mask_to_contour_volume_ratio(self):
        """
        Get ratio to ROI "mask" volume to ROI "contour" volume.

        In principle, volume estimates from "mask" and "contour" methods
        should be similar.  The two estimates may be different if, for
        example, the image from which a mask is created doesn't fully cover
        the contour extents.
        """
        return (self.get_volume(method="mask", force=True) /
                self.get_volume(method="contour", force=True))

    def get_area(
        self, 
        view="x-y", 
        sl=None, 
        idx=None, 
        pos=None, 
        units="mm", 
        method=None,
        flatten=False,
    ):
        """Get the area of the ROI on a given slice.

        **Parameters:**
        
        view : str, default="x-y"
            Orientation of slice for which to get area.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        units : str, default="mm"
            Units of area. Can be either of:

                * "mm": return area in millimetres squared.
                * "voxels": return area in number of voxels.

            If units="voxels" is requested but this ROI only has contours and no
            voxel size information, an error will be raised.

        method : str, default=None
            Method to use for area calculation. Can be: 

                * "contour": compute area from shapely polygon on the slice.
                * "mask": compute area by summing pixels on the slice and 
                  multiplying by the area of one pixel.
                * None: use the method set in self.default_geom_method.

        flatten : bool, default=False
            If True, all slices will be flattened in the given orientation and
            the area of the flattened slice will be returned. Only available
            if method="mask".
        """

        # Get default slice index and method
        self.load()
        if sl is None and idx is None and pos is None: 
            idx = self.get_mid_idx(view)
        if method is None:
            method = self.default_geom_method

        # If ROI is not on the given slice, return None
        if not self.on_slice(view, sl=sl, idx=idx, pos=pos):
            return

        if flatten:  # Flattening only possible with "mask"
            method = "mask"

        # Calculate area from shapely polygon(s)
        if method == "contour":

            # Check it's possible to calculate area with the requested units
            if self.voxel_size is None and units == "voxels":
                raise RuntimeError("Cannot compute area in voxels from "
                                   "contours without knowing voxel sizes!")

            polygons = self.get_polygons_on_slice(view, sl, idx, pos)
            area = sum([p.area for p in polygons])
            if units == "voxels":
                x_ax, y_ax = skrt.image._plot_axes[view]
                xy_area = abs(self.voxel_size[x_ax] * self.voxel_size[y_ax])
                area /= xy_area
            return area

        # Otherwise, calculate area from binary mask
        im_slice = self.get_slice(view, sl=sl, idx=idx, pos=pos, 
                                  flatten=flatten)
        area = im_slice.astype(bool).sum()
        if units == "mm":
            x_ax, y_ax = skrt.image._plot_axes[view]
            xy_area = abs(self.voxel_size[x_ax] * self.voxel_size[y_ax])
            area *= xy_area
        return area

    def get_bbox_centre_and_widths(self,
            buffer=None, buffer_units="mm", method=None):
        """
        Get centre and widths in mm along all three axes of a
        bounding box enclosing the ROI and optional buffer.  Centre
        and widths are returned as a tuple ([x, y, z], [dx, dy, dz]).

        Method parameters are passed to skrt.structures.ROI.get_extents()
        to obtain ROI extents.  For parameter explanations, see
        skrt.structures.ROI.get_extents() documentation.
        """
        extents = self.get_extents(buffer, buffer_units, method) 
        centre = [0.5 * (extent[0] + extent[1]) for extent in extents]
        widths = [(extent[1] - extent[0]) for extent in extents]
        return (centre, widths)

    def get_extents(self, buffer=None, buffer_units="mm", method=None,
            origin=None):
        """
        Get minimum and maximum extent of the ROI in mm along all three axes,
        returned in order [x, y, z]. Optionally apply a buffer to the extents
        such that they cover more than the ROI's area.

        **Parameters:**

        buffer : float, default=None
            Optional buffer to add to the extents. Units set by `buffer_units`.

        buffer_units : str, default="mm"
            Units for buffer, if using. Can be "mm", "voxels", or "frac" (which
            applies buffer as a fraction of total length in each dimension).

        method : str, default=None
            Method to use for extent calculation. Can be: 

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the 
                  binary mask.
                * None: use the method set in self.default_geom_method.

        origin : tuple, default=None
            Tuple specifying the (x, y, z) coordinates of the point
            with respect to which extents are to be determined.
            If None, then (0, 0, 0) is used.
        """
        # Ensure that origin is defined.
        origin = origin or (0, 0, 0)

        # Get extent in each direction
        extents = []
        for ax in skrt.image._axes:
            extents.append(self.get_extent(ax, method=method, origin=origin))

        # Apply buffer if requested
        if buffer:
            for ax in range(3):
                if buffer_units == "mm":
                    delta = buffer
                elif buffer_units == "voxels":
                    delta = abs(buffer * self.voxel_size[ax])
                elif buffer_units == "frac":
                    delta = buffer * abs(extents[ax][1] - extents[ax][0])
                else:
                    print(f"Unrecognised buffer units {buffer_units}. Should "
                          'be "mm", "voxels", or "frac".')
                    return
                extents[ax][0] -= delta
                extents[ax][1] += delta

        return extents

    def get_extent(self, 
                   ax="z", 
                   single_slice=False,
                   view="x-y",
                   sl=None, 
                   idx=None, 
                   pos=None, 
                   method=None, 
                   origin=None,
                  ):
        """Get minimum and maximum extent of the ROI in mm along a given axis.

        ax : str/int, default="z"
            Axis along which to return extent. Should be one of ["x", "y", "z"] 
            or [0, 1, 2]

        single_slice : bool, default=False
            If False, the 3D extent of the entire ROI will be returned;
            otherwise, the 2D extent of a single slice will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get extent. Only used if 
            single_slice=True. If using, <ax> must be an axis that lies along
            the slice in this orientation.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        method : str, default=None
            Method to use for extent calculation. Can be: 

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the 
                  binary mask.
                * None: use the method set in self.default_geom_method.

        origin : int/float/tuple, default=None
            Point with respect to which extents are to be determined.
            This can be an integer or float specifying the point's
            coordinate along axis <ax>; or it can be a tuple specifying
            the point's (x, y, z) coordinates, where only the coordinate
            along axis <ax> will be considered.  If None, the origin
            coordinate is taken to be 0.
        """

        # Get default slice and method
        self.load()
        if method is None:
            method = self.default_geom_method
        if single_slice:
            if idx is None and sl is None and pos is None:
                idx = self.get_mid_idx()

        # If using single slice, check the axis lies along the slice
        i_ax = skrt.image._axes.index(ax) if isinstance(ax, str) else ax
        if single_slice:
            if i_ax not in skrt.image._plot_axes[view]:
                raise RuntimeError(
                    f"Cannot compute extent of axis {ax} in the {view} plane!")
            i_ax = skrt.image._plot_axes[view].index(i_ax)

        # Ensure that origin is a single numeric value.
        origin = origin or 0
        if not isinstance(origin, (float, int)):
            origin = origin[i_ax]

        # Calculate extent from contours
        if method == "contour":

            # Calculate full z extent from contour positions
            if (ax == "z" or ax == 2) and not single_slice:
                z_keys = list(self.get_contours("x-y").keys())
                vz = self.get_slice_thickness_contours()
                if z_keys:
                    z_max = max(z_keys) +  vz / 2
                    z_min = min(z_keys) - vz / 2
                else:
                    z_min = None
                    z_max = None
                return [z_min, z_max]

            # Otherwise, calculate extent from min/max contour positions 
            points = []

            # Global: use every contour
            if not single_slice:
                for i, contours in self.get_contours("x-y").items():
                    for contour in contours:
                        points.extend([p[i_ax] for p in contour])

            # Single slice: just use the contour on the slice
            else:
                for contour in self.get_contours_on_slice(
                    view, sl=sl, idx=idx, pos=pos):
                    points.extend([p[i_ax] for p in contour])

            # Return min and max of the points in the contour(s)
            if len(points):
                return [min(points) - origin, max(points) - origin]
            else:
                return [None, None]

        # Otherwise, get extent from mask
        self.create_mask()
        if i_ax != 2:  # Transpose x-y axes
            i_ax = 1 - i_ax

        # Get positions of voxels inside the mask
        if not single_slice:
            nonzero = np.argwhere(self.get_mask(standardise=True))
        else:
            nonzero = np.argwhere(self.get_slice(
                view, sl=sl, idx=idx, pos=pos))
        vals = nonzero[:, i_ax]
        if not len(vals):
            return [None, None]

        # Find min and max voxels; add half a voxel either side to get 
        # full extent
        min_pos = min(vals) - 0.5
        max_pos = max(vals) + 0.5

        # Convert positions to mm
        return [self.idx_to_pos(min_pos, ax) - origin,
                self.idx_to_pos(max_pos, ax) - origin]

    def get_crop_limits(self, crop_margins=None, method=None):
        """
        Get crop limits corresponding to ROI extents plus margins.

        The tuples of limits returned, in the order (x, y, z) can
        be used, for example, as inputs to skrt.image.Image.crop().

        This method is similar to the method get_extents(), but allows
        different margins on each side of the ROI.

        **Parameters:**

        crop_margins : float/tuple, default=None
            Float or three-element tuple specifying the margins, in mm,
            to be added to ROI extents.  If a float, minus and plus the
            value specified are added to lower and upper extents respectively
            along each axis.  If a three-element tuple, elements are
            taken to specify margins in the order (x, y, z).  Elements
            can be either floats (minus and plus the value added respectively
            to lower and upper extents) or two-element tuples (elements 0 and 1
            added respectively to lower and upper extents).

        method : str, default=None
            Method to use for extent calculation. Can be:

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the
                  binary mask.
                * None: use the method set in self.default_geom_method.
        """
        crop_limits = self.get_extents(method=method)
        margins = skrt.image.checked_crop_limits(crop_margins)
        for idx1 in range(3):
            if margins[idx1] is not None:
                for idx2 in range(2):
                    crop_limits[idx1][idx2] += margins[idx1][idx2]

        return crop_limits

    def get_length(
        self, 
        ax="z", 
        single_slice=False,
        view="x-y",
        sl=None, 
        idx=None, 
        pos=None, 
        units="mm",
        method=None, 
        force=True
    ):
        """Get ROI length along a given axis. If single_slice=False (i.e. a 
        global length is requested), calculated length will be cached in 
        self._length[ax][units] and returned if called again, unless force=True.

        **Parameters:**
        
        ax : str/int, default="z"
            Axis along which to return length; should be one of ["x", "y", "z"]
            or [0, 1, 2].

        single_slice : bool, default=False
            If False, the length of the entire ROI will be returned;
            otherwise, the length on a single slice will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get length. Only used if 
            single_slice=True. If using, <ax> must be an axis that lies along
            the slice in this orientation.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        units : str, default="mm"
            Units of length. Can be either of:
                - "mm": return length in millimetres.
                - "voxels": return length in number of voxels.

            If units="voxels" is requested but this ROI only has contours and no
            voxel size information, an error will be raised (unless ax="z", 
            in which case voxel size will be inferred from spacing between
            slices).

        method : str, default=None
            Method to use for length calculation. Can be: 
                - "contour": get length from min/max positions of contour(s).
                - "mask": get length from min/max positions of voxels in the 
                  binary mask.
                - None: use the method set in self.default_geom_method.

        force : bool, default=True
            If True, the length will always be recalculated; otherwise, it will
            only be calculated if it has not yet been cached in self._volume.
            Note that if single_slice=True, the length will always be 
            recalculated.
        """

        # If global length already cached, return cached value
        if not single_slice and not force:
            if hasattr(self, "_length") and ax in self._length \
               and units in self._length[ax]:
                return self._length[ax][units]

        # If single slice requested and not on current slice, return None
        if single_slice and not self.on_slice(view, sl=sl, idx=idx, pos=pos):
            return None

        # Get length in mm from min and max positions
        self.load()
        min_pos, max_pos = self.get_extent(
            ax=ax, 
            single_slice=single_slice,
            view=view,
            sl=sl,
            idx=idx,
            pos=pos,
            method=method
        )
        length = {
            "mm": abs(max_pos - min_pos)
        }

        # Get length in voxels
        # Case where voxel size is known
        i_ax = skrt.image._axes.index(ax) if isinstance(ax, str) else ax
        if self.voxel_size is not None:
            length["voxels"] = length["mm"] / self.voxel_size[i_ax]

        # Deal with case where self.voxel_size is None
        else:
            if i_ax == 2:
                length["voxels"] = length["mm"] / self.get_slice_thickness_contours()
            elif units == "voxels":
                raise RuntimeError("Cannot compute length in voxels from "
                                   "contours without knowing voxel sizes!")

        
        # Cache the property if global
        if not single_slice:
            if not hasattr(self, "_length"):
                self._length = {}
            self._length[skrt.image._axes[i_ax]] = length
        
        # Return desired units
        return length[units]

    def get_geometry(
        self,
        metrics=None,
        vol_units="mm",
        area_units="mm",
        length_units="mm",
        centroid_units="mm",
        view="x-y",
        sl=None,
        pos=None,
        idx=None,
        method=None,
        units_in_header=False,
        global_vs_slice_header=False,
        name_as_index=True,
        nice_columns=False,
        decimal_places=None,
        force=True,
        html=False
    ):
        """Return a pandas DataFrame or html table of the geometric properties 
        listed in <metrics>.

        **Parameters:**
        
        metrics : list, default=None
            List of metrics to include in the table. Options:

                - "volume" : volume of entire ROI.
                - "area": area of ROI on a single slice.

                - "centroid": 3D centre-of-mass of ROI; will be split into
                  three columns corresponding to the three axes.
                - "centroid_slice": 2D centre of mass on a single slice. Will
                  be split into two columns corresponding to the two axes on
                  the slice.

                - "length": lengths of the ROI in each direction; will be split
                  into three columns corresponding to the three axes.
                - "length_slice": length of the ROI on a single slice; will be
                  split into two columns corresponding to the two axes on the
                  slice.

            If None, defaults to ["volume", "centroid"]

        vol_units : str, default="mm"
            Units to use for volume. Can be "mm" (=mm^3), "ml",
            or "voxels".

        area_units : str, default="mm"
            Units to use for areas. Can be "mm" (=mm^2) or "voxels".

        length_units : str, default="mm":
            Units to use for lengths. Can be "mm" or "voxels".

        centroid_units:
            Units to use for centroids. Can be "mm" or "voxels".

        view : str, default="x-y"
            Orientation in which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).

        sl : int, default=None
            Slice number on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            If <sl>, <idx>, and <pos> are all None, the central slice of the
            ROI will be used.

        idx : int, default=None
            Slice index array on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            Used if <sl> is None.

        pos : float, default=None
            Slice position in mm on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            Used if <sl> and <idx> are both None.

        method : str, default=None
            Method to use for metric calculation. Can be either "contour" 
            (use shapely polygons) or "mask" (use binary mask). If None,
            the value set in self.default_geom_method will be used.

        units_in_header : bool, default=False
            If True, units will be included in column headers.

        global_vs_slice_header : bool, default=False
            If True, the returned DataFrame will have an extra set of headers
            splitting the table into "global" and "slice" metrics.

        name_as_index : bool, default=True
            If True, the index column of the pandas DataFrame will 
            contain the ROI's name; otherwise, the name will appear in a column
            named "ROI".

        nice_columns : bool, default=False
            If False, column names will be the same as the input metrics names;
            if True, the names will be capitalized and underscores will be 
            replaced with spaces.

        decimal_places : int, default=None
            Number of decimal places to keep for each metric. If None, full
            precision will be used.

        force : bool, default=True
            If False, global metrics (volume, 3D centroid, 3D lengths) will
            only be calculated if they have not been calculated before;
            if True, all metrics will be recalculated.
        """

        # If no index given, use central slice
        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)

        # Default metrics
        if metrics is None:
            metrics = ["volume", "centroid"]

        # Compute metrics
        geom = {}
        slice_kwargs = {
            "view": view,
            "sl": sl,
            "idx": idx,
            "pos": pos
        }
        for m in metrics:

            # 3D volume
            if m == "volume":
                geom[m] = self.get_volume(
                    units=vol_units, 
                    method=method,
                    force=force
                )

            # Area on a slice
            elif m == "area":
                geom[m] = self.get_area(
                    units=area_units, 
                    method=method,
                    **slice_kwargs
                )

            # 3D centroid
            elif m == "centroid":
                centroid = self.get_centroid(
                    units=centroid_units, 
                    method=method,
                    force=force
                )
                for i, ax in enumerate(skrt.image._axes):
                    geom[f"centroid_{ax}"] = centroid[i]

            # 2D centroid
            elif m == "centroid_slice":
                centroid = self.get_centroid(
                    units=centroid_units, 
                    method=method,
                    single_slice=True,
                    **slice_kwargs
                )
                for i, i_ax in enumerate(skrt.image._plot_axes[view]):
                    ax = skrt.image._axes[i_ax]
                    geom[f"centroid_slice_{ax}"] = centroid[i]

            # Full lengths along each axis
            elif m == "length":
                for ax in skrt.image._axes:
                    geom[f"length_{ax}"] = self.get_length(
                        ax=ax, 
                        units=length_units,
                        method=method,
                        force=force
                    )

            # Lengths on a slice
            elif m == "length_slice":
                for i_ax in skrt.image._plot_axes[view]:
                    ax = skrt.image._axes[i_ax]
                    geom[f"length_slice_{ax}"] = self.get_length(
                        ax=ax, 
                        units=length_units,
                        method=method,
                        single_slice=True,
                        **slice_kwargs
                    )

            else:
                found = False

                # Axis-specific metrics
                for i, ax in enumerate(skrt.image._axes):

                    # Global centroid position on a given axis
                    if m == f"centroid_{ax}":
                        geom[m] = self.get_centroid(
                            units=centroid_units,
                            method=method,
                            force=force
                        )[i]
                        found = True

                    # Full length along a given axis
                    elif m == f"length_{ax}":
                        geom[m] = self.get_length(
                            ax=ax,
                            units=length_units, 
                            method=method,
                            force=force
                        )
                        found = True

                if not found:
                    raise RuntimeError(f"Metric {m} not recognised by "
                                       "ROI.get_geometry()")

        # Add units to metric names if requested
        geom_named = {}
        if not units_in_header:
            geom_named = geom
        else:
            for metric, val in geom.items():
                name = metric
                if "volume" in metric:
                    if vol_units == "mm":
                        name += " (mm^3)"
                    else:
                        name += " (" + vol_units + ")"
                elif "area" in metric:
                    if area_units == "mm":
                        name += " (mm^2)"
                    else:
                        name += " (" + area_units + ")"
                elif "length" in metric:
                    name += " (" + length_units + ")"
                elif "centroid" in metric:
                    name += " (" + centroid_units + ")"
                geom_named[name] = val

        # Adjust number of decimal places
        if decimal_places is not None:
            for metric, val in geom_named.items():
                fmt = f"{{val:.{decimal_places}f}}"
                if geom_named[metric] is not None:
                    geom_named[metric] = float(fmt.format(val=val))

        # Convert to pandas DataFrame
        df = pd.DataFrame(geom_named, index=[self.name])

        # Capitalize column names and remove underscores if requested
        if nice_columns and not global_vs_slice_header:
            df.columns = [col.capitalize().replace("_", " ") 
                          for col in df.columns]

        # Turn name into a regular column if requested
        if not name_as_index:
            df = df.reset_index().rename({"index": "ROI"}, axis=1)

        # Add global/slice headers
        if global_vs_slice_header:

            # Sort columns into global or slice
            headers = []
            slice_cols = []
            global_cols = []
            slice_name = "slice" if not nice_columns else "Slice"
            global_name = "global" if not nice_columns else "Global"
            for metric in geom_named:
                name = metric if not nice_columns else \
                        metric.capitalize().replace("_", "")
                if "area" in metric or "slice" in metric:
                    headers.append((slice_name, metric))
                    slice_cols.append(metric)
                else:
                    headers.append((global_name, metric))
                    global_cols.append(metric)

            # Reorder columns
            dfs_ordered = [df[global_cols], df[slice_cols]]
            if not name_as_index:
                dfs_ordered.insert(0, df["ROI"])
            df = pd.concat(dfs_ordered, axis=1)

            # Sort headers by global/slice
            headers = [h for h in headers if h[0] == global_name] \
                    + [h for h in headers if h[0] == slice_name]
            if not name_as_index:
                headers.insert(0, ("", "ROI"))

            # Add MultiIndex
            df.columns = pd.MultiIndex.from_tuples(headers)

        if html:
            return df_to_html(df)
        return df

    def get_centroid_distance(self, roi, single_slice=False, view="x-y",
            sl=None, idx=None, pos=None, units="mm", method=None, force=True,
            by_slice=None, value_for_none=None, slice_stat=None,
            **slice_stat_kwargs):
        """Get centroid displacement vector with respect to another ROI.

        **Parameters:**
        
        roi : ROI
            Other ROI with which to compare centroid.

        single_slice : bool, default=False
            If True, the centroid will be returned for a single slice.

        view : str, default="x-y"
            Orientation in which to get centroids if using a single slice or
            flattening.

        flatten : bool, default=False
            If True, the 3D centroid will be obtained and then the absolute 
            value along only the 2D axes in the orientation in <view> will
            be returned.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of each ROI
            will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of each ROI
            will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of each ROI
            will be used.

        units : str, default="mm"
            Units of absolute centroid distance. Can be any of:
                - "mm": return centroid position in millimetres.
                - "voxels": return centroid position in terms of array indices.
                - "slice": return centroid position in terms of slice numbers.

            If units="voxels" or "slice" is requested but either ROI only has 
            contours and no mask shape/voxel size information, an error will be 
            raised (unless ax="z", in which case voxel size will be inferred 
            from spacing between slices).

        method : str, default=None
            Method to use for centroid calculation. Can be: 
                - "contour": get centroid of shapely contour(s).
                - "mask": get average position of voxels in binary mask.
                - None: use the method set in self.default_geom_method.

        force : bool, default=True
            If True, the global centroid will always be recalculated; 
            otherwise, it will only be calculated if it has not yet been cached 
            in self._volume.  Note that if single_slice=True, the centroid will 
            always be recalculated.

        by_slice : str, default=None
            If one of "left", "right", "union", "intersection", calculate
            Dice scores slice by slice for each slice containing
            self and/or other:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other;
            - "union": consider slices containing either of self and other;
            - "intersection": consider slices containing both of self and other.

            If slice_stat is None, return a dictionary where keys are
            slice positions and values are the calculated Dice scores.
            Otherwise, return for the calculated Dice scores the statistic
            specified by slice_stat.

        value_for_none : float, default=None
            For single_slice and by_slice, value to be returned for
            slices where Dice score is undetermined (None).  For slice_stat,
            value to substitute for any None values among the inputs
            for calculating slice-based statistic.  If None in the latter
            case, None values among the inputs are omitted.

        slice_stat : str, default=None
            Single-variable statistic(s) to be returned for slice-by-slice
            Dice scores.  This should be the name of the function for
            calculation of the statistic(s) in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            Available options include: "mean", "median", "mode",
            "stdev", "quantiles".  Disregarded if None.

            If by_slice is None and slice_stat is different from None,
            skrt.core.Defaults().by_slice is used for the former.

        slice_stat_kwargs : dict, default=None
            Keyword arguments to be passed to function of statistics
            module for calculation relative to slice values.  For example,
            if quantiles are required for 10 intervals, rather than for
            the default of 4, this can be specified using:

            slice_stat_kwargs{"n" : 10}

            For available keyword options, see documentation of statistics
            module at:

            https://docs.python.org/3/library/statistics.html
        """
        if slice_stat:
            return self.get_slice_stat(roi, "centroid", slice_stat,
                    by_slice, value_for_none, view, method,
                    **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(roi, "centroid", by_slice,
                    view, method)

        this_centroid = np.array(self.get_centroid(
            single_slice=single_slice, view=view, sl=sl, idx=idx, pos=pos,
            units=units, method=method, force=force)
        )
        other_centroid = np.array(roi.get_centroid(
            single_slice=single_slice, view=view, sl=sl, idx=idx, pos=pos,
            units=units, method=method, force=force)
        )
        if None in this_centroid or None in other_centroid:
            if single_slice:
                return np.array([value_for_none, value_for_none])
            else:
                return np.array([value_for_none, value_for_none,
                    value_for_none])
        return other_centroid - this_centroid

    def get_abs_centroid_distance(
        self, 
        roi, 
        view="x-y", 
        single_slice=False,
        flatten=False, 
        sl=None,
        idx=None,
        pos=None,
        units="mm",
        method=None,
        force=True,
        by_slice=None,
        value_for_none=None,
        slice_stat=None,
        slice_stat_kwargs=None,
        ):
        """Get absolute centroid distance with respect to another ROI.

        **Parameters:**
        
        roi : ROI
            Other ROI with which to compare centroid.

        view : str, default="x-y"
            Orientation in which to get centroids if using a single slice or
            flattening.

        single_slice : bool, default=False
            If True, the centroid will be returned for a single slice.

        flatten : bool, default=False
            If True, the 3D centroid will be obtained and then the absolute 
            value along only the 2D axes in the orientation in <view> will
            be returned.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of each ROI
            will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of each ROI
            will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of each ROI
            will be used.

        units : str, default="mm"
            Units of absolute centroid distance. Can be any of:
                - "mm": return centroid position in millimetres.
                - "voxels": return centroid position in terms of array indices.
                - "slice": return centroid position in terms of slice numbers.

            If units="voxels" or "slice" is requested but either ROI only has 
            contours and no mask shape/voxel size information, an error will be 
            raised (unless ax="z", in which case voxel size will be inferred 
            from spacing between slices).

        method : str, default=None
            Method to use for centroid calculation. Can be: 
                - "contour": get centroid of shapely contour(s).
                - "mask": get average position of voxels in binary mask.
                - None: use the method set in self.default_geom_method.

        force : bool, default=True
            If True, the global centroid will always be recalculated; 
            otherwise, it will only be calculated if it has not yet been cached 
            in self._volume.  Note that if single_slice=True, the centroid will 
            always be recalculated.

        by_slice : str, default=None
            If one of "left", "right", "union", "intersection", calculate
            Dice scores slice by slice for each slice containing
            self and/or other:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other;
            - "union": consider slices containing either of self and other;
            - "intersection": consider slices containing both of self and other.

            If slice_stat is None, return a dictionary where keys are
            slice positions and values are the calculated Dice scores.
            Otherwise, return for the calculated Dice scores the statistic
            specified by slice_stat.

        value_for_none : float, default=None
            For single_slice and by_slice, value to be returned for
            slices where Dice score is undetermined (None).  For slice_stat,
            value to substitute for any None values among the inputs
            for calculating slice-based statistic.  If None in the latter
            case, None values among the inputs are omitted.

        slice_stat : str, default=None
            Single-variable statistic(s) to be returned for slice-by-slice
            Dice scores.  This should be the name of the function for
            calculation of the statistic(s) in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            Available options include: "mean", "median", "mode",
            "stdev", "quantiles".  Disregarded if None.

            If by_slice is None and slice_stat is different from None,
            skrt.core.Defaults().by_slice is used for the former.

        slice_stat_kwargs : dict, default=None
            Keyword arguments to be passed to function of statistics
            module for calculation relative to slice values.  For example,
            if quantiles are required for 10 intervals, rather than for
            the default of 4, this can be specified using:

            slice_stat_kwargs{"n" : 10}

            For available keyword options, see documentation of statistics
            module at:

            https://docs.python.org/3/library/statistics.html
        """
        if slice_stat:
            return self.get_slice_stat(roi, "abs_centroid", slice_stat,
                    by_slice, value_for_none, view, method,
                    **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(roi, "abs_centroid", by_slice,
                    view, method)

        # If flattening, need to get 3D centroid vector
        if flatten:
            single_slice = False

        # Get centroid vector
        centroid = self.get_centroid_distance(
            roi, 
            view=view, 
            single_slice=single_slice,
            sl=sl,
            idx=idx,
            pos=pos,
            units=units,
            method=method,
            force=force,
        )

        # If centroid wasn't available, return None
        if None in centroid:
            return value_for_none

        # If flattening, take 2 axes only
        if flatten:
            x_ax, y_ax = skrt.image._plot_axes[view]
            centroid = np.array([centroid[x_ax], centroid[y_ax]])

        # Return magnitude of vector
        return np.linalg.norm(centroid)

    def get_dice(
        self, 
        other, 
        single_slice=False,
        view="x-y", 
        sl=None, 
        idx=None, 
        pos=None, 
        method=None,
        flatten=False,
        by_slice=None,
        value_for_none=None,
        slice_stat=None,
        slice_stat_kwargs=None,
    ):
        """
        Get Dice score with respect to another ROI,
        globally, on a single slice, slice by slice, or slice averaged.

        **Parameters:**
        
        other : ROI
            Other ROI to compare with this ROI.

        single_slice : bool, default=False
            If False, the global 3D Dice score of the full ROIs will be returned;
            otherwise, the 2D Dice score on a single slice will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get Dice score. Only used if 
            single_slice=True. If using, <ax> must be an axis that lies along
            the slice in this orientation.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        method : str, default=None
            Method to use for Dice score calculation. Can be: 
                - "contour": get intersections and areas of shapely contours.
                - "mask": count intersecting voxels in binary masks.
                - None: use the method set in self.default_geom_method.

        flatten : bool, default=False
            If True, all slices will be flattened in the given orientation and
            the Dice score of the flattened slices will be returned. Only 
            available if method="mask".

        by_slice : str, default=None
            If one of "left", "right", "union", "intersection", calculate
            Dice scores slice by slice for each slice containing
            self and/or other:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other;
            - "union": consider slices containing either of self and other;
            - "intersection": consider slices containing both of self and other.

            If slice_stat is None, return a dictionary where keys are
            slice positions and values are the calculated Dice scores.
            Otherwise, return for the calculated Dice scores the statistic
            specified by slice_stat.

        value_for_none : float, default=None
            For single_slice and by_slice, value to be returned for
            slices where Dice score is undetermined (None).  For slice_stat,
            value to substitute for any None values among the inputs
            for calculating slice-based statistic.  If None in the latter
            case, None values among the inputs are omitted.

        slice_stat : str, default=None
            Single-variable statistic(s) to be returned for slice-by-slice
            Dice scores.  This should be the name of the function for
            calculation of the statistic(s) in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            Available options include: "mean", "median", "mode",
            "stdev", "quantiles".  Disregarded if None.

            If by_slice is None and slice_stat is different from None,
            skrt.core.Defaults().by_slice is used for the former.

        slice_stat_kwargs : dict, default=None
            Keyword arguments to be passed to function of statistics
            module for calculation relative to slice values.  For example,
            if quantiles are required for 10 intervals, rather than for
            the default of 4, this can be specified using:

            slice_stat_kwargs{"n" : 10}

            For available keyword options, see documentation of statistics
            module at:

            https://docs.python.org/3/library/statistics.html
        """
        if slice_stat:
            return self.get_slice_stat(other, "dice", slice_stat, by_slice,
                    value_for_none, view, method, **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(other, "dice", by_slice,
                    view, method)

        intersection, union, mean_size = self.get_intersection_union_size(
                other, single_slice, view, sl, idx, pos, method, flatten)

        return (intersection / mean_size if mean_size else value_for_none)

    def get_jaccard(
        self, 
        other, 
        single_slice=False,
        view="x-y", 
        sl=None, 
        idx=None, 
        pos=None, 
        method=None,
        flatten=False,
        by_slice=None,
        value_for_none=None,
        slice_stat=None,
        slice_stat_kwargs=None,
    ):
        """
        Get Jaccard index with respect to another ROI.

        The Jaccard index may be obtained either globally or on a 
        single slice.

        **Parameters:**
        
        other : ROI
            Other ROI to compare with this ROI.

        single_slice : bool, default=False
            If False, the global 3D Jaccard index of the full ROIs will be
            returned; otherwise, the 2D Jaccard index on a single slice
            will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get Jaccard index. Only used if 
            single_slice=True. If using, <ax> must be an axis that lies along
            the slice in this orientation.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        method : str, default=None
            Method to use for Dice score calculation. Can be: 
                - "contour": get intersections and areas of shapely contours.
                - "mask": count intersecting voxels in binary masks.
                - None: use the method set in self.default_geom_method.

        flatten : bool, default=False
            If True, all slices will be flattened in the given orientation and
            the Dice score of the flattened slices will be returned. Only 
            available if method="mask".

        by_slice : str, default=None
            If one of "left", "right", "union", "intersection", calculate
            Dice scores slice by slice for each slice containing
            self and/or other:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other;
            - "union": consider slices containing either of self and other;
            - "intersection": consider slices containing both of self and other.

            If slice_stat is None, return a dictionary where keys are
            slice positions and values are the calculated Jaccard indices.
            Otherwise, return for the calculated Jaccard indices the statistic
            specified by slice_stat.

        value_for_none : float, default=None
            For single_slice and by_slice, value to be returned for
            slices where Jaccard index is undetermined (None).  For slice_stat,
            value to substitute for any None values among the inputs
            for calculating slice-based statistic.  If None in the latter
            case, None values among the inputs are omitted.

        slice_stat : str, default=None
            Single-variable statistic(s) to be returned for slice-by-slice
            Jaccard indices.  This should be the name of the function for
            calculation of the statistic(s) in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            Available options include: "mean", "median", "mode",
            "stdev", "quantiles".  Disregarded if None.

            If by_slice is None and slice_stat is different from None,
            the former defaults to "union".

        slice_stat_kwargs : dict, default=None
            Keyword arguments to be passed to function of statistics
            module for calculation relative to slice values.  For example,
            if quantiles are required for 10 intervals, rather than for
            the default of 4, this can be specified using:

            slice_stat_kwargs{"n" : 10}

            For available keyword options, see documentation of statistics
            module at:

            https://docs.python.org/3/library/statistics.html
        """
        if slice_stat:
            return self.get_slice_stat(other, "jaccard", slice_stat, by_slice,
                    value_for_none, view, method, **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(other, "jaccard", by_slice,
                    view, method)

        intersection, union, mean_size = self.get_intersection_union_size(
                other, single_slice, view, sl, idx, pos, method, flatten)

        return (intersection / union if union else value_for_none)

    def get_volume_ratio(self, other, **kwargs):
        """Get ratio of another ROI's volume with respect to own volume."""

        own_volume = other.get_volume(**kwargs)
        other_volume = self.get_volume(**kwargs)
        if not other_volume or not own_volume:
            return None
        return own_volume / other_volume

    def get_area_ratio(self, other, view="x-y", sl=None, idx=None, pos=None,
            units="mm", method=None, flatten=False, by_slice=None,
            slice_stat=None, value_for_none=None, **slice_stat_kwargs):
        """Get ratio of another ROI's area with respect to own area."""

        if slice_stat:
            return self.get_slice_stat(other, "area_ratio", slice_stat,
                    by_slice, value_for_none, view, method,
                    **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(other, "area_ratio", by_slice,
                    view, method)

        own_area = other.get_area(view, sl, idx, pos, units, method, flatten)
        other_area = self.get_area(view, sl, idx, pos, units, method, flatten)
        if not other_area or not own_area:
            return value_for_none
        return own_area / other_area

    def get_volume_diff(self, other, **kwargs):
        """Get own volume minus volume of other ROI."""

        own_volume = self.get_volume(**kwargs)
        other_volume = other.get_volume(**kwargs)
        if not own_volume or not other_volume:
            return None
        return (own_volume - other_volume)

    def get_relative_volume_diff(self, other, **kwargs):
        """Get relative volume of another ROI with respect to own volume."""

        own_volume = self.get_volume(**kwargs)
        volume_diff = self.get_volume_diff(other, **kwargs)
        if volume_diff is None:
            return
        return volume_diff / own_volume

    def get_area_diff(self, other, view="x-y", sl=None, idx=None, pos=None,
            units="mm", method=None, flatten=False, by_slice=None,
            slice_stat=None, value_for_none=None, **slice_stat_kwargs):
        """Get absolute area difference between two ROIs."""

        if slice_stat:
            return self.get_slice_stat(other, "area_diff", slice_stat,
                    by_slice, value_for_none, view, method,
                    **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(other, "area_diff", by_slice,
                    view, method)

        own_area = self.get_area(view, sl, idx, pos, units, method, flatten)
        other_area = other.get_area(view, sl, idx, pos, units, method, flatten)
        if not own_area or not other_area:
            return value_for_none
        return own_area - other_area

    def get_relative_area_diff(self, other, view="x-y", sl=None, idx=None,
            pos=None, units="mm", method=None, flatten=False, by_slice=None,
            slice_stat=None, value_for_none=None, **slice_stat_kwargs):
        """Get relative area of another ROI with respect to own area."""

        if slice_stat:
            return self.get_slice_stat(other, "rel_area_diff", slice_stat,
                    by_slice, value_for_none, view, method,
                    **(slice_stat_kwargs or {}))

        if by_slice:
            return self.get_metric_by_slice(other, "rel_area_diff",
                    by_slice, view, method)

        own_area = self.get_area(view, sl, idx, pos, units, method, flatten)
        area_diff = self.get_area_diff(other, view, sl, idx, pos, units, method,
                flatten)

        if area_diff is None:
            return value_for_none
        return area_diff / own_area

    def get_intersection_union_size(
        self, 
        other, 
        single_slice=False,
        view="x-y", 
        sl=None, 
        idx=None, 
        pos=None, 
        method=None,
        flatten=False,
    ):
        """
        Get intersection, union and mean size with respect to another ROI.
        Values are returned as volumes for the full ROIs,
        or as areas for a single slice.

        **Parameters:**
        
        other : ROI
            Other ROI to compare with this ROI.

        single_slice : bool, default=False
            If False, the volumes of intersection and  union of the full ROIs,
            and the mean ROI volume will be returned;
            otherwise, the areas of intersection and union on a single slice,
            and the mean ROI area, will be returned.

        view : str, default="x-y"
            Orientation of slice on which to get areas of intersection
            and union, and mean ROI area.  Only used if single_slice=True.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of this ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of this ROI will be
            used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied
            but <single_slice> is True, the central slice of this ROI will
            be used.

        method : str, default=None
            Method to use for calculating intersection, union, and mean
            size. Can be: 
                - "contour": get intersections and areas of shapely contours.
                - "mask": count intersecting voxels in binary masks.
                - None: use the method set in self.default_geom_method.

        flatten : bool, default=False
            If True, all slices will be flattened in the given orientation and
            the areas of intersection and union, and the mean ROI area,
            will be returned for the flattened slices. Only available
            if method="mask".
        """

        self.create_mask()

        # Define voxel volume, and voxel area in view.
        voxel_volume = 1
        for dxyz in self.mask.voxel_size:
            voxel_volume *= dxyz
        z_ax = skrt.image._slice_axes[view]
        slice_thickness = self.mask.voxel_size[z_ax]
        voxel_area = voxel_volume / slice_thickness

        # Get default slice and method
        if single_slice:
            if sl is None and idx is None and pos is None:
                idx = self.get_mid_idx(view)
        if method is None:
            method = self.default_geom_method

        if flatten:  # Flattening only possible with "mask"
            method = "mask"

        # Calculate intersections and areas from polygons
        if method == "contour":

            # Get positions of slice(s)
            # on which to compare areas and total areas.
            if not single_slice:
                positions = set(self.get_contours("x-y").keys()).union(
                        set(other.get_contours("x-y").keys()))
                areas1 = [self.get_area("x-y", pos=p, method=method)
                        for p in positions]
                volumes1 = [a * slice_thickness for a in areas1 if a]
                volume1 = sum(volumes1)
                areas2 = [other.get_area("x-y", pos=p, method=method)
                        for p in positions]
                volumes2 = [a * slice_thickness for a in areas2 if a]
                volume2 = sum(volumes2)
                mean_size = 0.5 * (volume1 + volume2)
            else:
                positions = [
                    self.idx_to_pos(self.get_idx(view, sl, idx, pos),
                                    ax=skrt.image._slice_axes[view])
                ]
                area1 = self.get_area(view, sl, idx, pos, method=method)
                if area1 is None:
                    area1 = 0
                area2 = other.get_area(view, sl, idx, pos, method=method)
                if area2 is None:
                    area2 = 0
                mean_size = 0.5 * (area1 + area2)
            
            # Compute areas of intersection and union on slice(s)
            intersection = 0
            union = 0
            factor = 1 if (single_slice or flatten) else slice_thickness
            for p in positions:
                set1 = ops.unary_union(self.get_polygons_on_slice(view, pos=p))
                set2 = ops.unary_union(other.get_polygons_on_slice(view, pos=p))
                intersection += set1.intersection(set2).area * factor
                union += set1.union(set2).area * factor

        # Calculate intersections and areas from binary mask voxel counts
        else:

            # Use 3D mask
            if not single_slice:
                data1 = self.get_mask(view, flatten, standardise=True)
                data2 = other.get_mask(view, flatten, standardise=True)

            # Otherwise, use single 2D slice
            else:
                data1 = self.get_slice(view, sl, idx, pos)
                data2 = other.get_slice(view, sl, idx, pos)

            factor = voxel_area if (single_slice or flatten) else voxel_volume
            intersection = (data1 & data2).sum() * factor
            union = (data1 | data2).sum() * factor
            mean_size = 0.5 * (data1.sum() + data2.sum()) * factor

        return (intersection, union, mean_size)

    def get_metric_by_slice(self, other, metric=None, by_slice=None,
            view="x-y", method=None):
        """
        Get dictionary of slice-by-slice values for comparison metric.

        Each key of the returned dictionary is a slice position (mm), and
        the associated value is the metric value for the slice.

        **Parameters:**

        other : ROI
            Other ROI to compare with this ROI.

        metric : str, default=None
            Metric to be evaluated.

        by_slice: str, default=None
            Slices to be considered.  The valid specifications are:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other roi;
            - "union": consider slices containing either of self and other roi;
            - "intersection": consider slices containing both of self and
              other roi.

            If None, use skrt.core.Defaults().by_slice

        view : str, default="x-y"
            Orientation of slices on which to get metric values.

        method : str, default=None
            Method to use for metric calculation. Can be: 
                - "contour": calculate metric from shapely contours.
                - "mask": calcualte metric from binary masks.
                - None: use the method set in self.default_geom_method.
        """
        if metric not in get_comparison_metrics():
            raise RuntimeError(f"Metric {metric} not recognised by "
                        "ROI.get_metric_by_slice()")

        by_slice = by_slice or skrt.core.Defaults().by_slice

        return {pos: getattr(self, f"get_{get_metric_method(metric)}")(
            other, single_slice=True, view=view, pos=pos, method=method)
            for pos in self.get_slice_positions(other, view, method=by_slice)}

    def get_slice_stat(self, other, metric=None, slice_stat=None, by_slice=None,
            value_for_none=None, view="x-y", method=None, **slice_stat_kwargs):
        """
        Get statistic for slice-by-slice comparison metric.

        **Parameters:**

        other : ROI
            Other ROI to compare with this ROI.

        metric : str, default=None
            Metric for which statistic for slice-by-slice values
            is to be returned.

        slice_stat : str, default=None
            Single-variable statistic(s) to be returned for slice-by-slice
            Dice scores.  This should be the name of the function for
            calculation of the statistic(s) in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            Available options include: "mean", "median", "mode",
            "stdev", "quantiles".

        by_slice: str, default=None
            Slices to be considered.  The valid specifications are:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other roi;
            - "union": consider slices containing either of self and other roi;
            - "intersection": consider slices containing both of self and
              other roi.

            If None, use skrt.core.Defaults().by_slice

        value_for_none : float, default=None
            Value to substitute for any None values among slice
            metric values.  If None, None values among the metric values
            are omitted.

        view : str, default="x-y"
            Orientation of slices on which to get metric values.

        method : str, default=None
            Method to use for metric calculation. Can be: 
                - "contour": calculate metric from shapely contours.
                - "mask": calcualte metric from binary masks.
                - None: use the method set in self.default_geom_method.

        slice_stat_kwargs : dict, default=None
            Keyword arguments to be passed to function of statistics
            module for calculation relative to slice values.  For example,
            if quantiles are required for 10 intervals, rather than for
            the default of 4, this can be specified using:

            slice_stat_kwargs{"n" : 10}

            For available keyword options, see documentation of statistics
            module at:

            https://docs.python.org/3/library/statistics.html
        """
        by_slice = by_slice or skrt.core.Defaults().by_slice
        slice_stat_kwargs = slice_stat_kwargs or {}

        return skrt.core.get_stat(self.get_metric_by_slice(
            other, metric, by_slice, view, method),
            value_for_none, slice_stat, **slice_stat_kwargs)

    def get_slice_stats(self, other, metrics=None, slice_stats=None,
            default_by_slice=None, method=None, view="x-y",
            separate_components=False):
        """
        Get dictionary of statistics for slice-by-slice comparison metrics.

        **Parameters:**

        other : ROI
            Other ROI to compare with this ROI.

        metrics : str/list, default=None
            Metric(s) for which statistic(s) for slice-by-slice calculations
            are to be returned.  A single metric can by speficied as a string.
            Multiple metrics can be specified as a list of strings.

        slice_stats : str/list/dict, default=None
            Specification of statistics to be calculated relative
            to slice-by-slice metric values.  Used for all metrics with
            suffix "_slice_stats" in the <metrics> list.

            A statistic is specified by the name of the function for
            its calculation in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            For its use here, the relevant function must require
            a single-variable input, must not require any keyword
            arguments, and must return a single number.

            Available options include: "mean", "median", "mode",
            "stdev".

           Statistics to be calculated can be specified using any of
           the following:

           - String specifying single statistic to be calculated,
             with slices considered as given by default_by_slice,
             for example: "mean";
           - List specifying multiple statistics to be calculated,
             with slices considered as given by default_by_slice,
             for example: ["mean", "stdev"];
           - Dictionary where keys specify slices to be considered,
             and values specify statistics (string or list), for
             example: {"union": ["mean", "stdev"]}.  Valid slice
             specifications are as listed for default_by_slice.

        default_by_slice: str, default=None
            Default specification of slices to be considered when
            calculating slice-by-slice statistics.  The valid specifications
            are:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other roi;
            - "union": consider slices containing either of self and other roi;
            - "intersection": consider slices containing both of self and
              other roi.

            If None, use skrt.core.Defaults().by_slice

        view : str, default="x-y"
            Orientation of slices on which to get metric values.

        method : str, default=None
            Method to use for metric calculation. Can be: 
                - "contour": calculate metric from shapely contours.
                - "mask": calculate metric from binary masks.
                - None: use the method set in self.default_geom_method.

        separate_components: bool, default=False
            If False, return a metric represented by a vector as a
            list of component values.  If true, return a separate metric for
            each component, labelling by the component axes determined
            from <view>.
        """
        if metrics is None or slice_stats is None:
            return {}

        default_by_slice = default_by_slice or skrt.core.Defaults().by_slice

        # Ensure that metrics is a non-empty list.
        if isinstance(metrics, str):
            metrics = [metrics]
        elif not skrt.core.is_list(metrics):
            return {}
        metrics = [metric for metric in metrics
                if f"{metric}_slice_stats" in get_comparison_metrics()]
        if not metrics:
            return {}

        slice_stats = expand_slice_stats(slice_stats, default_by_slice)
        if not slice_stats:
            return {}

        # Calculate statistic(s) for metric values calculated slice by slice.
        calculated_slice_stats = {}
        for metric in metrics:
            for by_slice, stats in slice_stats.items():
                values = self.get_metric_by_slice(
                        other, metric, by_slice, view, method)
                for stat in stats:
                    key = f"{metric}_slice_{by_slice}_{stat}"
                    calculated_slice_stats[key] = (
                            skrt.core.get_stat(values, stat=stat))
                    if (skrt.core.is_list(calculated_slice_stats[key])
                            and separate_components):
                        vector = calculated_slice_stats.pop(key)
                        for i, i_ax in enumerate(skrt.image._plot_axes[view]):
                            ax = skrt.image._axes[i_ax]
                            ax_key = f"{metric}_{ax}_slice_{by_slice}_{stat}"
                            calculated_slice_stats[ax_key] = vector[i]

        return calculated_slice_stats

    def match_mask_voxel_size(self, other, voxel_size=None,
            voxel_dim_tolerance=0.1):
        '''
        Ensure that the mask voxel sizes of <self> and <other> match.

        The mask voxel_sizes for both may optionally be set to a new
        common value.

        **Paramters:**

        other : skrt.structures.ROI
            ROI with which voxel size is to be matched.

        voxel_size : tuple, default=None
            Voxel size (dx, dy, dz) in mm to be used for ROI masks
            from which to calculate surface distances.  If None,
            the mask voxel size of <other> is used if not None;
            otherwise the default voxel size for dummy images,
            namely (1, 1, 1), is used.  If an element of the
            tuple specifying voxel size is None, the value for
            the corresponding element of the mask voxel size of
            <other> is used.

        voxel_dim_tolerance : float, default=0.1
            Maximum accepted value for the absolute difference in voxel size
            (any dimension) of the ROI masks for them to be considered
            as having the same voxel size.  If a negative value is given,
            the absolute difference is never below this, forcing creation
            of new masks with the the specified voxel size.  This can be
            useful for obtaining masks that are just large enough to contain
            both ROIs, potentially reducing the time for surface-distance
            calculations.
        '''
        # Create ROI clones, for which voxel sizes may be altered.
        roi1 = self.clone()
        roi2 = other.clone()

        # Associate new dummy image with ROIs if requested voxel size
        # is different from voxel size of either current image,
        # or if absolute difference in any dimension between voxel sizes
        # of current images don't match is greater than tolerance.
        voxel_size = voxel_size if voxel_size else roi2.get_voxel_size()
        if voxel_size and roi2.get_voxel_size():
            voxel_size = [voxel_size[idx] if voxel_size[idx] is not None
                    else roi2.get_voxel_size()[idx] for idx in range(3)]
        resize = not roi1.get_voxel_size() or not roi2.get_voxel_size()
        if not resize:
            for roi in [roi1, roi2]:
                matches = [dxyz1 for dxyz1, dxyz2 in
                        zip(voxel_size, roi.voxel_size)
                        if abs(dxyz1 - dxyz2) < voxel_dim_tolerance]
                if 3 != len(matches):
                    resize = True
        if resize:
            if voxel_size:
                slice_thickness = voxel_size[2]
                voxel_size_2d = voxel_size[:2]
                # Ensure that initial voxel size is set.
                # This is used to define slice thickness
                # for an ROI that has no associated image,
                # and is defined by one or more contours on a single
                # image slice.
                if not roi1.voxel_size:
                    roi1.voxel_size = voxel_size
                if not roi2.voxel_size:
                    roi2.voxel_size = voxel_size
            else:
                slice_thickness = None
                voxel_size_2d = None

            StructureSet([roi1, roi2]).set_image_to_dummy(
                    slice_thickness=slice_thickness, voxel_size=voxel_size_2d)

        return (roi1, roi2)

    def get_surface_distances(
        self,
        other,
        single_slice=False,
        signed=False,
        view="x-y",
        sl=None,
        idx=None,
        pos=None,
        connectivity=2,
        flatten=False,
        symmetric=None,
        conformity=False,
        in_slice=True,
        voxel_size=None,
        voxel_dim_tolerance=0.1,
    ):
        """
        Get vector of surface distances between two ROIs.

        Surface distances may be signed or unsigned, and may be for a single
        slice through the ROIs, or for the 3D volumes.  

        This function uses ideas outlined at:
        https://mlnotebook.github.io/post/surface-distance-function/

        **Parameters:**

        single_slice : bool, default=False
            If False, return surface distances relative to 3D volumes.
            Otherwise, return surface distances for a single image slice.

        signed : bool, default=False
            If False, return absolute surface distances.
            If True, return signed surface distances.  A point on the
            surface of <other> is taken to be at a positive distance if
            outside <self>, and at a negative distance if inside.

        view : str, default="x-y"
            Orientation of slice for which to obtain surface distances.
            Only used if either <single_slice> or <flatten> is True.

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> is supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> is 
            supplied but <single_slice> is True, the central slice of
            the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> is supplied
            but <single_slice> is True, the central slice of the ROI will
            be used.

        connectivity: int, default=2
            Integer defining which voxels to consider as neighbours
            when performing erosion operation to identify ROI surfaces.  This
            is passed as <connectivity> parameter to:
            scipy.ndimage.generate_binary_structure
            For further details, see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html

        flatten : bool, default=False
            If True, all slices for each ROI will be flattened in the given
            orientation, and surface distances relative to the flattened
            slice will be returned.

        symmetric : bool, default=None
            If True, the distances returned are from the surface of <other>
            to the surface of <self> and from the surface of <self> to the
            surface of <other>.  If False, distances are from the surface
            of <other> to the surface of <self> only.  If None, set to
            the opposiste of <signed>.

         conformity : bool, default=False
             If True, distances returned are signed distances to conformity,
             as defined in:
             https://doi.org/10.1259/bjr/27674581
             In this case, the values of <signed> and <symmetric>
             are disregarded.

        in_slice : bool, default=True
            If True, only intra-slice connectivity is considered when
            performing erosion to identify ROI surfaces.

        voxel_size : tuple, default=None
            Voxel size (dx, dy, dz) in mm to be used for ROI masks
            from which to calculate surface distances.  If None,
            the mask voxel size of <other> is used.  If an individual
            element is None, the value of the corresponding element for
            the mask voxel size of <other> is used.

        voxel_dim_tolerance : float, default=0.1
            Tolerence used when determining whether <voxel_size> is
            different from the voxel size of the images associated with
            each ROI.
        """

        # Set view to None if considering 3D mask.
        if not single_slice and not flatten:
            view = None

        # Set to False parameters that are to be ignored
        # when calculating distances to conformity.
        if conformity:
            symmetric = signed = False

        # Default to symmetric or one-way distances,
        # depending on whether distances are signed or unsigned.
        if symmetric is None:
            symmetric = False if signed is True else True

        # Obtain ROI clones, with mask voxel sizes matched to voxel_size
        # if non-null, or otherwise to the mask voxel size of other.
        roi1, roi2 = self.match_mask_voxel_size(other, voxel_size,
                voxel_dim_tolerance)

        # Check whether ROIs are empty
        if not np.any(roi1.get_mask()) or not np.any(roi2.get_mask()):
            return

        # Get voxel size relative to axes of numpy array.
        voxel_size = roi1.get_np_voxel_size(view)

        # Get binary masks
        if single_slice:
            mask1 = roi1.get_slice(view, sl=sl, idx=idx, pos=pos)
            mask2 = roi2.get_slice(view, sl=sl, idx=idx, pos=pos)
        else:
            mask1 = roi1.get_mask(view, flatten=flatten)
            mask2 = roi2.get_mask(view, flatten=flatten)

        # Make structuring element
        conn2 = ndimage.generate_binary_structure(2, connectivity)
        if mask1.ndim == 2:
            conn = conn2
        else:
            if in_slice:
                conn = np.zeros((3, 3, 3), dtype=bool)
                conn[:, :, 1] = conn2
            else:
                conn = ndimage.generate_binary_structure(3, connectivity)

        # Get outer pixel of binary maps
        surf1 = mask1 ^ ndimage.binary_erosion(mask1, conn)
        surf2 = mask2 ^ ndimage.binary_erosion(mask2, conn)

        # Make arrays of distances to surface of each ROI
        dist1 = ndimage.distance_transform_edt(~surf1, voxel_size)
        dist2 = ndimage.distance_transform_edt(~surf2, voxel_size)

        # Get signed distances
        if signed:
            dist1 = dist1 * ~mask1 - dist1 * mask1
            dist2 = dist2 * ~mask2 - dist2 * mask2

        # Make vector containing all distances
        if conformity:
            dist1[mask1 & mask2] = 0
            dist1[mask1 & ~mask2] = -dist2[mask1 & ~mask2]
            sds = np.ravel(dist1[mask1 | mask2])
        else:
            sds = np.ravel(dist1[surf2 != 0])
            if symmetric:
                sds = np.concatenate([sds, np.ravel(dist2[surf1 != 0])])
        return sds

    def get_mean_surface_distance(self, other, **kwargs):
        '''
        Obtain mean distance between surface of <other> and surface of <self>.

        For parameters that may be passed, see documentation for:
        skrt.ROI.get_surface_distances().
        '''

        sds = self.get_surface_distances(other, **kwargs)
        if sds is None:
            return
        return sds.mean()

    def get_rms_surface_distance(self, other, **kwargs):

        sds = self.get_surface_distances(other, **kwargs)
        if sds is None:
            return
        return np.sqrt((sds ** 2).mean())

    def get_hausdorff_distance(self, other, **kwargs):

        sds = self.get_surface_distances(other, **kwargs)
        if sds is None:
            return
        return sds.max()

    def get_mean_distance_to_conformity(self, other, vol_units="mm", **kwargs):
        '''
        Obtain mean distance to conformity for <other> relative to <self>.

        The mean distance to conformity is as defined in:
        https://doi.org/10.1259/bjr/27674581
        Technically the mean distance to conformity isn't a distance,
        but is a distance divided by a volume.

        This function returns a skrt.core.Data object, with the following
        information:

        n_voxel : number of voxels in the union of the ROIs considered.

        voxel_size : voxel size (dx, dy, dz) in millimetres, in the
        ROI masks used in distance calculations.

        mean_under_contouring : sum of absolute values of negative distances
        to conformity, divided by volume in union of ROIs considered.

        mean_over_contouring : sum of positive distances to conformity,
        divided by volume in union of ROIs considered.

        mean_distance_to_conformity : sum of mean_under_contouring and
        mean_over_contouring.

        **Parameters:**
        vol_units : str, default="mm"
            Units to use for volume in denominator when calculating
            mean_under_contouring and mean_overcontouring. Can be
            "mm" (=mm^3), "ml", or "voxels".

        kwargs
            Keyword arguments are passed to
            skrt.structures.ROI.get_surface_distances().
            See documentation of this method for parameter details.
        '''

        # Obtain distances to conformity.
        kwargs["conformity"] = True
        sds = self.get_surface_distances(other, **kwargs)

        # Obtain mask array representing union of ROIs.
        union = StructureSet([self, other]).combine_rois()
        view = kwargs.get("view", "x-y")
        if kwargs.get("single_slice", False):
            sl = kwargs.get("sl", None)
            idx = kwargs.get("idx", None)
            pos = kwargs.get("pos", None)
            mask_data = union.get_slice(view, sl, idx, pos)
        else:
            flatten = kwargs.get("flatten", False)
            mask_data = union.get_mask(view, flatten)

        # Calculate conformity scores.
        conformity = skrt.core.Data()
        conformity.n_voxel = mask_data.sum()
        conformity.voxel_size = union.get_voxel_size()
        if sds is None:
            conformity.mean_under_contouring = None
            conformity.mean_over_contouring = None
            conformity.mean_distance_to_conformity = None
        else:
            conformity.mean_under_contouring = (abs(sds[sds < 0].sum())
                    / union.get_volume(vol_units))
            conformity.mean_over_contouring = (sds[sds > 0].sum()
                    / union.get_volume(vol_units))
            conformity.mean_distance_to_conformity = (
                    conformity.mean_under_contouring +
                    conformity.mean_over_contouring)

        return conformity

    def get_surface_distance_metrics(self, other, **kwargs):
        """Get the mean surface distance, RMS surface distance, and Hausdorff
        distance."""

        sds = self.get_surface_distances(other, **kwargs)
        if sds is None:
            return
        return sds.mean(), np.sqrt((sds ** 2).mean()), sds.max()

    def plot_surface_distances(self, other, save_as=None, signed=False, **kwargs):
        """Plot histogram of surface distances."""

        sds = self.get_surface_distances(other, signed=signed, **kwargs)
        if sds is None:
            return
        fig, ax = plt.subplots()
        ax.hist(sds)
        xlabel = (
            "Surface distance (mm)" if not signed else "Signed surface distance (mm)"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of voxels")
        if save_as:
            plt.tight_layout()
            fig.savefig(save_as)

    def get_comparison(
        self,
        roi,
        metrics=None,
        vol_units="mm",
        area_units="mm",
        centroid_units="mm",
        view="x-y",
        sl=None,
        pos=None,
        idx=None,
        method=None,
        slice_stats=None,
        default_by_slice=None,
        units_in_header=False,
        global_vs_slice_header=False,
        name_as_index=True,
        nice_columns=False,
        decimal_places=None,
        force=True,
        voxel_size=(1, 1, 1),
        match_lengths=False,
        match_lengths_strategy=1
    ):
        """Return a pandas DataFrame of the comparison metrics listed in 
        <metrics> with respect to another ROI.

        **Parameters:**
        
        roi : ROI
            Other ROI with which to compare this ROI.

        metrics : list, default=None
            List of metrics to include in the table. Options:

                * "dice" : global Dice score.
                * "dice_flat": Dice score of ROIs flattened in the orientation
                  specified in <view>.
                * "dice_slice": Dice score on a single slice.
                * "dice_slice_stats": statistics specified in <slice_stats>
                  for slice-by-slice Dice scores.

                * "jaccard" : global Jaccard index.
                * "jaccard_flat": Jaccard index of ROIs flattened in
                  the orientation specified in <view>.
                * "jaccard_slice": Jaccard index on a single slice.
                * "jaccard_slice_stats": statistics specified in <slice_stats>
                  for slice-by-slice Jaccard indices.

                * "centroid": 3D centroid distance vector.
                * "abs_centroid": magnitude of 3D centroid distance vector.
                * "abs_centroid_flat": magnitude of 3D centroid distance vector
                  projected into the plane specified in <view>.
                * "centroid_slice": 2D centroid distance vector on a single slice.
                * "centroid_slice_stats": statistics specified in
                  <slice_stats> for slice-by-slice 2D centroid distance vectors.
                * "abs_centroid_slice": magnitude of 2D centroid distance 
                  vector on a single slice.
                * "abs_centroid_slice_stats": statistics specified in
                  <slice_stats> for slice-by-slice magnitudes of 2D
                  centroid distance vectors.

                * "volume_diff": volume difference (own volume - other volume).
                * "rel_volume_diff": volume difference divided by own volume.
                * "volume_ratio": volume ratio (own volume / other volume).

                * "area_diff": area difference (own area - other area) on a 
                  single slice.
                * "rel_area_diff": area difference divided by own area on a 
                  single slice.
                * "area_ratio": area ratio (own area / other area).
                * "area_diff_flat": area difference of ROIs flattened in the
                  orientation specified in <view>.
                * "area_diff_slice_stats": statistics specified in
                  <slice_stats> for slice-by-slice area differences.
                * "rel_area_diff_flat": relative area difference of ROIs
                  flattened in the orientation specified in <view>.
                * "rel_area_diff_slice_stats": statistics specified in
                  <slice_stats> for relative area differences of ROIs.
                * "area_ratio_flat": area ratio of ROIs flattened in the
                  orientation specified in <view>.
                * "area_ratio_slice_stats": statistics specified in
                  <slice_stats> for slice-by-slice area ratios.

                * "mean_signed_surface_distance": mean signed surface distance.
                * "mean_signed_surface_distance_flat": mean signed surface
                  distance of ROIs flattened in the orientation specified
                  in <view>.
                * "mean_surface_distance": mean surface distance.
                * "mean_surface_distance_flat": mean surface distance of ROIs
                  flattened in the orientation specified in <view>.
                * "rms_surface_distance": RMS surface distance.
                * "rms_surface_distance_flat": RMS surface distance of ROIs
                  flattened in the orientation specified in <view>.
                * "rms_signed_surface_distance": RMS signed surface distance.
                * "rms_signed_surface_distance_flat": RMS signed surface
                  distance of ROIs flattened in the orientation specified
                  in <view>.
                * "hausdorff_distance": Hausdorff distance.
                * "hausdorff_distance_flat": Hausdorff distance of ROIs
                  flattened in the orientation specified in <view>.
                * "mean_under_contouring": Sum of absolute values of
                  negative distances to conformity, divided by volume
                  in union of ROIs compared.
                * "mean_under_contouring_flat": Sum of absolute values of
                  negative distances to conformity, divided by volume,
                  flattened in the orientation specified in <view>.
                  in union of ROIs compared.
                * "mean_over_contouring": Sum of positive distances
                  to conformity, divided by volume in union of ROIs compared.
                * "mean_over_contouring_flat": Sum of positive distances
                  to conformity, divided by volume in union of ROIs compared,
                  flattened in the orientation specified in <view>.
                * "mean_distance_to_conformity": Sum of mean_under_contouring
                  and mean_over_contouring.
                * "mean_distance_to_conformity_flat": Sum of
                  mean_under_contouring_flat and mean_over_contouring_flat.

            If None, defaults to ["dice", "centroid"].

        vol_units : str, default="mm"
            Units to use for volume. Can be "mm" (=mm^3), "ml",
            or "voxels".

        area_units : str, default="mm"
            Units to use for areas. Can be "mm" (=mm^2) or "voxels".

        centroid_units:
            Units to use for centroids. Can be "mm" or "voxels".

        view : str, default="x-y"
            Orientation in which to compute metrics. Only relevant for 
            single-slice metrics or flattened metrics.

        sl : int, default=None
            Slice number on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            If <sl>, <idx>, and <pos> are all None, the central slice of the
            ROI will be used.

        idx : int, default=None
            Slice index array on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            Used if <sl> is None.

        pos : float, default=None
            Slice position in mm on which to compute metrics. Only relevant for 
            single-slice metrics (area, centroid_slice, length_slice).
            Used if <sl> and <idx> are both None.

        method : str, default=None
            Method to use for metric calculation. Can be either "contour" 
            (use shapely polygons) or "mask" (use binary mask). If None,
            the value set in self.default_geom_method will be used. Note 
            that flattened metrics and surface distance metrics enforce use
            of the "mask" method.

        slice_stats : str/list/dict, default=None
            Specification of statistics to be calculated relative
            to slice-by-slice metric values.  Used for all metrics with
            suffix "_slice_stats" in the <metrics> list.

            A statistic is specified by the name of the function for
            its calculation in the Python statistics module:

            https://docs.python.org/3/library/statistics.html

            For its use here, the relevant function must require
            a single-variable input, must not require any keyword
            arguments, and must return a single number.

            Available options include: "mean", "median", "mode",
            "stdev".

           Statistics to be calculated can be specified using any of
           the following:

           - String specifying single statistic to be calculated,
             with slices considered as given by default_by_slice,
             for example: "mean";
           - List specifying multiple statistics to be calculated,
             with slices considered as given by default_by_slice,
             for example: ["mean", "stdev"];
           - Dictionary where keys specify slices to be considered,
             and values specify statistics (string or list), for
             example: {"union": ["mean", "stdev"]}.  Valid slice
             specifications are as listed for default_by_slice.

        default_by_slice: str, default=None
            Default specification of slices to be considered when
            calculating slice-by-slice statistics.  The valid specifications
            are:

            - "left": consider only slices containing self;
            - "right": consider only slices containing other roi;
            - "union": consider slices containing either of self and other roi;
            - "intersection": consider slices containing both of self and
              other roi.

            If None, use skrt.core.Defaults().by_slice

        units_in_header : bool, default=False
            If True, units will be included in column headers.

        global_vs_slice_header : bool, default=False
            If True, the returned DataFrame will have an extra set of headers
            splitting the table into "global" and "slice" metrics.

        name_as_index : bool, default=True
            If True, the index column of the pandas DataFrame will 
            contain the ROI's names; otherwise, the name will appear in a column
            named "ROI".

        nice_columns : bool, default=False
            If False, column names will be the same as the input metrics names;
            if True, the names will be capitalized and underscores will be 
            replaced with spaces.

        decimal_places : int, default=None
            Number of decimal places to keep for each metric. If None, full
            precision will be used.

        force : bool, default=True
            If False, global metrics for each ROI (volume, 3D centroid, 
            3D lengths) will only be calculated if they have not been 
            calculated before; if True, all metrics will be recalculated.

        voxel_size : tuple, default=(1, 1, 1)
            Voxel size (dx, dy, dz) in mm to be used for ROI masks
            from which to calculate surface distances.  If None,
            the mask voxel size of <other> is used.  If an individual
            element is None, the value of the corresponding element for
            the mask voxel size of <other> is used.  A value
            other than None forces creation of masks that just cover
            the volume occupied by the two ROIs being compared,
            potentially reducing the time for surface-distance calculations.

        match_lengths : bool/str/list, default=False
            If set to True, to the name of the other ROI, or to a list
            of names including the name of the other ROI, match ROI lengths
            (z-dimensions) before comparison.  The strategy used is defined
            by the value of match_lengths_strategy.

        match_lengths_strategy : int, default=1
            Strategy to use for matching ROI lengths (z-dimensions):

            - 0 : Crop this ROI to the z-extents of the other ROI;
            - 1 : Crop the other ROI to the z-exents of this ROI;
            - 2 : Crop this ROI to the z-extents of the other ROI, then
                  crop the other ROI to the cropped z-extents of this ROI.

            For a value other than 0, 1, 2, no length matching is performed.
        """
        if voxel_size is not None:
            # Create ROI masks, just covering volume occupied
            # by ROIs being compared, with specified voxel size.
            # (The resizing is performed on clones on the ROIs,
            # without changing the originals.)
            roi0, roi = self.match_mask_voxel_size(roi,
                    voxel_size=voxel_size, voxel_dim_tolerance=-1)
        else:
            # Don't explicitly create ROI masks.
            roi0 = self

        # Optionally match ROI lengths.
        if ((match_lengths is True) or (roi.name == match_lengths)
                or (isinstance(match_lengths, (list, tuple, set))
                    and roi.name in match_lengths)):
            # Create ROI clones for cropping; the originals will be unchanged.
            if match_lengths_strategy in [0, 1, 2] and voxel_size is None:
                roi0 = self.clone()
                roi = roi.clone()
            # Crop this ROI to z-extents of other ROI.
            if match_lengths_strategy in [0, 2]:
                roi0.crop_to_roi_length(roi)
            # Crop other ROI to z-extents of this ROI.
            if match_lengths_strategy in [1, 2]:
                roi.crop_to_roi_length(roi0)

        # Default metrics
        if metrics is None:
            metrics = ["dice", "centroid"]

        # Default slices to consider for slice-by-slice metric calculations.
        default_by_slice = skrt.core.Defaults().by_slice

        # Initialise variables for distance metrics.
        conformity = None
        conformity_flat = None
        distances = None
        distances_flat = None
        signed_distances = None
        signed_distances_flat = None

        # If no index given, use central slice
        if sl is None and idx is None and pos is None:
            idx = roi0.get_mid_idx(view)

        # Compute metrics
        comp = {}
        slice_kwargs = {
            "single_slice": True,
            "view": view,
            "sl": sl,
            "idx": idx,
            "pos": pos
        }

        for m in metrics:
            if m not in get_comparison_metrics():
                raise RuntimeError(f"Metric {m} not recognised by "
                        "ROI.get_comparison()")

        for m in metrics:

            # Dice score
            if m == "dice":
                comp[m] = roi0.get_dice(roi, method=method)
            elif m == "dice_flat":
                comp[m] = roi0.get_dice(
                    roi, 
                    view=view,
                    method=method,
                    flatten=True, 
                )
            elif m == "dice_slice":
                comp[m] = roi0.get_dice(roi, method=method, **slice_kwargs)

            elif m == "dice_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="dice",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view))

            # Jaccard index
            elif m == "jaccard":
                comp[m] = roi0.get_jaccard(roi, method=method)
            elif m == "jaccard_flat":
                comp[m] = roi0.get_jaccard(
                    roi, 
                    view=view,
                    method=method,
                    flatten=True, 
                )
            elif m == "jaccard_slice":
                comp[m] = roi0.get_jaccard(roi, method=method, **slice_kwargs)

            elif m == "jaccard_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="jaccard",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view))

            # Centroid distances
            elif m == "centroid":
                centroid = roi0.get_centroid_distance(
                    roi, 
                    units=centroid_units, 
                    method=method,
                    force=force
                )
                for i, ax in enumerate(skrt.image._axes):
                    comp[f"centroid_{ax}"] = centroid[i]
            elif m == "abs_centroid":
                comp[m] = roi0.get_abs_centroid_distance(
                    roi, 
                    units=centroid_units,
                    method=method, 
                    force=force
                )
            elif m == "abs_centroid_flat":
                comp[m] = roi0.get_abs_centroid_distance(
                    roi,
                    view=view,
                    units=centroid_units,
                    method=method,
                    flatten=True,
                    force=force
                )
            elif m == "centroid_slice":
                centroid = roi0.get_centroid_distance(
                    roi,
                    units=centroid_units,
                    method=method,
                    **slice_kwargs
                )
                for i, i_ax in enumerate(skrt.image._plot_axes[view]):
                    ax = skrt.image._axes[i_ax]
                    comp[f"centroid_slice_{ax}"] = centroid[i]

            elif m == "centroid_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="centroid",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view,
                        separate_components=True))

            elif m == "abs_centroid_slice":
                comp[m] = roi0.get_abs_centroid_distance(
                    roi,
                    units=centroid_units,
                    method=method,
                    **slice_kwargs
                )

            elif m == "abs_centroid_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="abs_centroid",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view))

            # Volume metrics
            elif m == "volume_diff":
                comp[m] = roi0.get_volume_diff(
                    roi,
                    units=vol_units,
                    method=method,
                    force=force
                )
            elif m == "rel_volume_diff":
                comp[m] = roi0.get_relative_volume_diff(
                    roi,
                    method=method,
                    force=force
                )
            elif m == "volume_ratio":
                comp[m] = roi0.get_volume_ratio(
                    roi,
                    method=method,
                    force=force
                )

            # Area metrics
            elif m == "area_diff":
                comp[m] = roi0.get_area_diff(
                    roi,
                    units=area_units,
                    method=method,
                    **slice_kwargs
                )
            elif m == "rel_area_diff":
                comp[m] = roi0.get_relative_area_diff(
                    roi,
                    method=method,
                    **slice_kwargs
                )
            elif m == "area_ratio":
                comp[m] = roi0.get_area_ratio(
                    roi,
                    method=method,
                    **slice_kwargs
                )
            elif m == "area_diff_flat":
                comp[m] = roi0.get_area_diff(
                    roi,
                    units=area_units,
                    method=method,
                    view=view,
                    flatten=True,
                )

            elif m == "area_diff_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="area_diff",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view))

            elif m == "rel_area_diff_flat":
                comp[m] = roi0.get_relative_area_diff(
                    roi,
                    view=view,
                    method=method,
                    flatten=True,
                )

            elif m == "rel_area_diff_slice_stats":
                comp.update(roi0.get_slice_stats(roi,
                    metrics="rel_area_diff",
                    slice_stats=slice_stats,
                    default_by_slice=default_by_slice,
                    method=method, view=view))

            elif m == "area_ratio_flat":
                comp[m] = roi0.get_area_ratio(
                    roi,
                    view=view,
                    method=method,
                    flatten=True,
                )

            elif m == "area_ratio_slice_stats":
                comp.update(roi0.get_slice_stats(roi, metrics="area_ratio",
                        slice_stats=slice_stats,
                        default_by_slice=default_by_slice,
                        method=method, view=view))

            # Surface distance metrics
            elif m in ["mean_surface_distance",
                    "rms_surface_distance", "hausdorff_distance"]:
                distances = (distances
                        or roi0.get_surface_distance_metrics(roi))
                if m == "mean_surface_distance":
                    comp[m] = distances[0]
                elif m == "rms_surface_distance":
                    comp[m] = distances[1]
                elif m == "hausdorff_distance":
                    comp[m] = distances[2]
            elif m in ["mean_surface_distance_flat",
                    "rms_surface_distance_flat", "hausdorff_distance_flat"]:
                distances_flat = (distances_flat
                        or roi0.get_surface_distance_metrics(roi,
                        view=view, flatten=True))
                if m == "mean_surface_distance_flat":
                    comp[m] = distances_flat[0]
                elif m == "rms_surface_distance_flat":
                    comp[m] = distances_flat[1]
                elif m == "hausdorff_distance_flat":
                    comp[m] = distances_flat[2]
            elif m in ["mean_signed_surface_distance",
                    "rms_signed_surface_distance"]:
                signed_distances = (signed_distances
                        or roi0.get_surface_distance_metrics(roi, signed=True))
                if m == "mean_signed_surface_distance":
                    comp[m] = signed_distances[0]
                elif m == "rms_signed_surface_distance":
                    comp[m] = signed_distances[1]
            elif m in ["mean_signed_surface_distance_flat",
                    "rms_signed_surface_distance_flat"]:
                signed_distances_flat = (signed_distances_flat
                        or roi0.get_surface_distance_metrics(roi, signed=True,
                            view=view, flatten=True))
                if m == "mean_signed_surface_distance_flat":
                    comp[m] = signed_distances_flat[0]
                elif m == "rms_signed_surface_distance_flat":
                    comp[m] = signed_distances_flat[1]
            elif m in ["mean_under_contouring", "mean_over_contouring",
                    "mean_distance_to_conformity"]:
                conformity = (conformity
                        or roi0.get_mean_distance_to_conformity(
                            roi, vol_units))
                if m == "mean_under_contouring":
                    comp[m] = conformity.mean_under_contouring
                if m == "mean_over_contouring":
                    comp[m] = conformity.mean_over_contouring
                if m == "mean_distance_to_conformity":
                    comp[m] = conformity.mean_distance_to_conformity
            elif m in ["mean_under_contouring_flat",
                    "mean_over_contouring_flat",
                    "mean_distance_to_conformity_flat"]:
                conformity_flat = (conformity_flat
                        or roi0.get_mean_distance_to_conformity(
                            roi, vol_units, view=view, flatten=True))
                if m == "mean_under_contouring_flat":
                    comp[m] = conformity_flat.mean_under_contouring
                if m == "mean_over_contouring_flat":
                    comp[m] = conformity_flat.mean_over_contouring
                if m == "mean_distance_to_conformity_flat":
                    comp[m] = conformity_flat.mean_distance_to_conformity
            else:
                # This code block can be reached if this method doesn't handle
                # a metric that's listed by get_comparison_metrics().
                found = False

                # Axis-specific metrics
                for i, ax in enumerate(skrt.image._axes):

                    # Global centroid position on a given axis
                    if m == f"centroid_{ax}":
                        comp[m] = roi0.get_abs_centroid_distance(
                            roi,
                            units=centroid_units,
                            method=method,
                            force=force
                        )
                        found = True

                if not found:
                    raise RuntimeError(f"Metric {m} not recognised by "
                                       "ROI.get_comparison()")

        # Add units to metric names if requested
        comp_named = {}
        if not units_in_header:
            comp_named = comp
        else:
            for metric, val in comp.items():
                name = metric
                if metric == "volume_diff":
                    if vol_units == "mm":
                        name += " (mm^3)"
                    else:
                        name += " (" + vol_units + ")"
                elif metric == "area_diff":
                    if area_units == "mm":
                        name += " (mm^2)"
                    else:
                        name += " (" + area_units + ")"
                elif "centroid" in metric:
                    name += " (" + centroid_units + ")"
                comp_named[name] = val

        # Adjust number of decimal places
        if decimal_places is not None:
            for metric, val in comp_named.items():
                fmt = f"{{val:.{decimal_places}f}}"
                if comp_named[metric] is not None:
                    comp_named[metric] = float(fmt.format(val=val))

        # Convert to pandas DataFrame
        df = pd.DataFrame(comp_named, index=[roi0.name])

        # Capitalize column names and remove underscores if requested
        if nice_columns and not global_vs_slice_header:
            df.columns = [col.capitalize().replace("_", " ").replace("rms", "RMS")
                          for col in df.columns]

        # Turn name into a regular column if requested
        if not name_as_index:
            df = df.reset_index().rename({"index": "ROI"}, axis=1)

        # Add global/slice headers
        if global_vs_slice_header:

            # Sort columns into global or slice
            headers = []
            slice_cols = []
            global_cols = []
            slice_name = "slice" if not nice_columns else "Slice"
            global_name = "global" if not nice_columns else "Global"
            for metric in comp_named:
                if "slice" in metric or \
                   ("area" in metric and "flat" not in metric):
                    headers.append((slice_name, metric))
                    slice_cols.append(metric)
                else:
                    headers.append((global_name, metric))
                    global_cols.append(metric)

            # Reorder columns
            dfs_ordered = [df[global_cols], df[slice_cols]]
            if not name_as_index:
                dfs_ordered.insert(0, df["ROI"])
            df = pd.concat(dfs_ordered, axis=1)

            # Sort headers by global/slice
            headers = [h for h in headers if h[0] == global_name] \
                    + [h for h in headers if h[0] == slice_name]
            if not name_as_index:
                headers.insert(0, ("", "ROI"))

            # Add MultiIndex
            df.columns = pd.MultiIndex.from_tuples(headers)

        return df

    def get_comparison_name(self, roi, camelcase=False, colored=False,
                            grey=False, roi_kwargs={}):
        """Get name of comparison between this ROI and another."""

        own_name = self.name
        other_name = roi.name

        if colored:
            own_color = self.get_color_from_kwargs(roi_kwargs)
            own_name = get_colored_roi_string(self, grey, own_color)
            other_color = roi.get_color_from_kwargs(roi_kwargs)
            other_name = get_colored_roi_string(roi, grey, other_color)

        if self.name == roi.name:
            if camelcase:
                return own_name.replace(" ", "_")
            return own_name
        else:
            if not colored:
                if camelcase:
                    return f"{own_name}_vs_{other_name}".replace(" ", "_")
                return f"{own_name} vs. {other_name}"
            return f"{own_name}{other_name}"

    def set_alpha_over_beta(self, alpha_over_beta=None):
        """
        Set ratio for ROI tissue of coefficients of linear-quadratic equation.

        **Parameter:**

        alpha_over_beta : float, default=None
            Ratio for ROI tissue of coefficients, alpha, beta,
            of linear-quadratic equation.  This ratio is used in
            calculating biologically effective dose, BED, from
            physical dose, D, delivered over n equal fractions:
                BED = D * (1 + (D/n) / (alpha/beta)).
            If None, the biologically effective dose is taken to be
            equal to the physical dose (beta >> alpha).
        """
        self.alpha_over_beta = alpha_over_beta

    def get_alpha_over_beta(self):
        """
        Get ratio for ROI tissue of coefficients of linear-quadratic equation.
        """
        return self.alpha_over_beta

    def set_color(self, color):
        """Set plotting color."""

        if color is not None and not matplotlib.colors.is_color_like(color):
            print(f"Warning: {color} not a valid color!")
            color = None
        if color is None:
            color = ROIDefaults().get_default_roi_color()
        self.color = matplotlib.colors.to_rgba(color)

    def plot(
        self,
        view="x-y",
        plot_type="contour",
        sl=None,
        idx=None,
        pos=None,
        ax=None,
        gs=None,
        figsize=skrt.image._default_figsize,
        opacity=None,
        linewidth=None,
        contour_kwargs=None,
        mask_kwargs=None,
        zoom=None,
        zoom_centre=None,
        color=None,
        show=True,
        save_as=None,
        include_image=False,
        no_invert=False,
        **kwargs,
    ):
        """Plot this ROI as either a mask or a contour.

        **Parameters:**

        view : str, default="x-y"
            Orientation in which to plot. Can be "x-y", "y-z", or "x-z".

        plot_type : str, default="contour"
            Plotting type. If None, will be either "contour" or "mask" 
            depending on the input type of the ROI. Options:

                - "contour"
                - "mask"
                - "centroid" (contour with marker at centroid)
                - "filled" (transparent mask + contour)
                - "filled centroid" (filled with marker at centroid)

        sl : int, default=None
            Slice number to plot. If none of <sl>, <idx> or <pos> are supplied, 
            the central slice of the ROI will be 
            used.

        idx : int, default=None
            Array index of slice to plot. If none of <sl>, <idx> or <pos> are 
            supplied, the central slice of the ROI will be used.

        pos : float, default=None
            Position in mm of slice to plot. If none of <sl>, <idx> or <pos> 
            are supplied, the central slice of the ROI will be used.

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot. If None, new axes will be created.

        gs : matplotlib.gridspec.GridSpec, default=None
            If not None and <ax> is None, new axes will be created on the
            current matplotlib figure with this gridspec.

        figsize : float, default=skrt.image._default_figsize
            Figure height in inches; only used if <ax> and <gs> are None.

        opacity : float, default=None
            Opacity to use if plotting mask (i.e. plot types "mask", "filled", 
            or "filled centroid"). If None, opacity will be 1 by default for 
            solid mask plots and 0.3 by default for filled plots.

        linewidth : float, default=None
            Width of contour lines. If None, the matplotlib default setting 
            will be used.

        contour_kwargs : dict, default=None
            Extra keyword arguments to pass to matplotlib contour plotting.

        mask_kwargs : dict, default=None
            Extra keyword arguments to pass to matplotlib mask plotting.

        zoom : int/float/tuple, default=None
            Factor by which to zoom in. If a single int or float is given,
            the same zoom factor will be applied in all directions. If a tuple
            of three values is given, these will be used as the zoom factors
            in each direction in the order (x, y, z). If None, the image will
            not be zoomed in.

        zoom_centre : tuple, default=None
            Position around which zooming is applied. If None, the centre of
            the image will be used.

        color : matplotlib color, default=None
            Color with which to plot the ROI; overrides the ROI's own color. If 
            None, self.color will be used.

        show : bool, default=True
            If True, the plot will be displayed immediately.

        save_as : str, default=None
            If set to a string, the plot will be saved to the filename in the 
            string.

        include_image : bool, default=False
            If True and self.image is not None, the ROI's image will be plotted
            in the background.

        no_invert : bool, default=False
            If False, contour plots will be automatically inverted when plotting
            without an underlying image to account for the y axis increasing 
            in the opposite direction. Otherwise, the plot will be left as
            it appeared when drawn by matplotlib.

        `**`kwargs :
            Extra keyword arguments to pass to the relevant plot function.
        """
        self.load()
        if self.empty:
            return

        if plot_type is None:
            plot_type = kwargs.get("roi_plot_type", self.default_geom_method)

        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)

        show_centroid = "centroid" in plot_type
        if zoom and zoom_centre is None:
            zoom_centre = self.get_zoom_centre(view)
        if color is None:
            color = self.color
        else:
            color = matplotlib.colors.to_rgba(color)

        if opacity is None:
            opacity = kwargs.get(
                    "roi_opacity", 0.3 if "filled" in plot_type else 1)

        # Set up axes
        self.set_ax(plot_type, include_image, view, ax, gs, figsize)

        # Adjust centroid marker size
        if contour_kwargs is None:
            contour_kwargs = {}
        if "centroid" in plot_type or "contour" == plot_type:
            if linewidth is None:
                linewidth = kwargs.get("roi_linewidth",
                        defaultParams["lines.linewidth"][0])
        if "centroid" in plot_type:
            contour_kwargs.setdefault("markersize", 7 * np.sqrt(linewidth))
            contour_kwargs.setdefault("markeredgewidth", np.sqrt(linewidth))

        # Plot a mask
        if plot_type == "mask":
            self._plot_mask(
                view,
                idx,
                mask_kwargs,
                opacity,
                zoom=zoom,
                zoom_centre=zoom_centre,
                show=False,
                include_image=include_image,
                color=color,
                **kwargs,
            )

        # Plot a contour
        elif plot_type in ["contour", "centroid"]:
            self._plot_contour(
                view,
                idx,
                contour_kwargs,
                linewidth,
                centroid=show_centroid,
                zoom=zoom,
                zoom_centre=zoom_centre,
                color=color,
                show=False,
                include_image=include_image,
                **kwargs,
            )

        # Plot transparent mask + contour
        elif "filled" in plot_type:
            if opacity is None:
                opacity = 0.3
            self._plot_mask(view, idx, mask_kwargs, opacity, color=color,
                           show=False, include_image=include_image, **kwargs)
            kwargs["ax"] = self.ax
            self._plot_contour(
                view,
                idx,
                contour_kwargs,
                linewidth,
                centroid=show_centroid,
                zoom=zoom,
                zoom_centre=zoom_centre,
                color=color,
                show=False,
                include_image=False,
                **kwargs,
            )

        # Check whether y axis needs to be inverted
        if view == "x-y" and self.ax.get_ylim()[1] > self.ax.get_ylim()[0]:
            self.ax.invert_yaxis()

        # Draw legend
        legend = kwargs.get("legend", False)
        if legend:
            name = kwargs.get("name", self.name)
            roi_handle = self.get_patch(
                    plot_type, color, opacity, linewidth, name)
            if roi_handle:
                bbox_to_anchor = kwargs.get("legend_bbox_to_anchor", None)
                loc = kwargs.get("legend_loc", "lower left")

                self.ax.legend(handles=[roi_handle],
                        bbox_to_anchor=bbox_to_anchor, loc=loc,
                        facecolor="white", framealpha=1
                        )

        # Display image

        plt.tight_layout()
        if show:
            plt.show()

        # Save to file
        if save_as:
            self.fig.savefig(save_as)
            plt.close()

    def get_dummy_image(self, **kwargs):
        """Make a dummy image that covers the area spanned by this ROI. 
        Returns an Image object.

        **Parameters:**
        
        voxel_size : list, default=None
            Voxel size in mm in the dummy image in the x-y plane, given as 
            [vx, vy]. If <shape> and <voxel_size> are both None, voxel sizes of
            [1, 1] will be used by default. The voxel size in the z direction 
            is defined by <slice_thickness>.

        shape : list, default=None
            Number of voxels in the dummy image in the x-y plane, given as
            [nx, ny]. Only used if <voxel_size> is None. 
            The number of voxels in the z direction will be taken from the 
            number of slices in the x-y contours dictionary.

        fill_val : int/float, default=1e4
            Value with which the voxels in the dummy image should be filled. 

        buffer : int, default=1
            Number of empty buffer voxels to add outside the ROI in each
            direction.

        slice_thickness : float, default=None
            Voxel size in mm in the dummy image in the z direction.  If None,
            the value used is the minimum distance between slice positions
            in the x-y contours dictionary.
        """

        extents = [self.get_extent(ax=ax) for ax in skrt.image._axes]
        slice_thickness = kwargs.pop("slice_thickness", None)
        slice_thickness = (slice_thickness or
                self.get_slice_thickness_contours())
        return create_dummy_image(extents, slice_thickness, **kwargs)

    def set_image_to_dummy(self, **kwargs):
        """Assign self.image property to a dummy image covering the area 
        spanned by this ROI. The ROI's mask and x-z/y-z contours will be 
        cleared so that they will be recreated when get_mask() or get_contours()
        are next called.

        **Parameters:**
        
        kwargs : 
            Keyword arguments to pass to self.get_dummy_image() when creating
            the dummy image. See documentation of ROI.get_dummy_image().
        """

        # Make image
        im = self.get_dummy_image(**kwargs)

        # Clear mask and contours
        self.input_contours = self.contours["x-y"]
        self.contours = {"x-y": self.input_contours}
        self.loaded_mask = False
        self.mask = None

        # Assign image
        self.set_image(im)

    def set_image(self, im):
        """Set self.image to a given image and adjust geometric properties
        accordingly. Note that self.mask will be removed if the current mask's 
        shape doesn't match the image. 
        """

        self.load()
        self.image = im
        self.contours_only = False
        if im is None:
            return

        # Set ROI voxel size to that of new image.
        # It will be needed to define slice thickness
        # in the case of an ROI with contour(s) at a single z-position.
        self.voxel_size = im.get_voxel_size()

        # If the z-distance between contours is greater than the voxel
        # z-dimension for the new image, first ensure that the ROI
        # mask is created with the inter-contour z-distance, then resize
        # to the image voxel size.
        if self.get_slice_thickness_contours() > self.image.get_voxel_size()[2]:
            if not (self.mask and (self.voxel_size == self.mask.voxel_size)):
                self.create_mask(voxel_size=self.voxel_size[0:2], force=True)
                # The mask creation will have redefined the image association,
                # so reassign the new image.
                self.image = im
            self.mask.match_size(self.image, method="nearest")
            if hasattr(self.mask, "data"):
                if not self.mask.data.dtype == bool:
                    self.mask.data = self.mask.data.astype(bool)

        # If a mask has been loaded, but doesn't match the image geometry,
        # delete the mask data.  (This should mean that the z-distance
        # between contours is less that the voxel z-dimension for the
        # new image.)  A new mask will be created at the next call
        # to access the mask data.
        if self.mask and not im.has_same_geometry(self.mask):
            if getattr(self, "input_contours", None) is None:
                self.input_contours = self.get_contours("x-y")
            self.mask = None
            self.loaded_mask = False

        # Set geoemtric info
        data_shape = im.get_data().shape
        self.shape = [data_shape[1], data_shape[0], data_shape[2]]
        self.voxel_size = im.get_voxel_size()
        self.origin = im.get_origin()
        self.affine = im.get_affine()

    def get_image(self):
        """Return Image object associated with this ROI."""
        return self.image

    def view(self, include_image=False, voxel_size=[1, 1], buffer=5, **kwargs):
        """View the ROI.

        **Parameters:**
        
        include_image : bool, default=True
            If True and this ROI has an associated image (in self.image),
            the image will be displayed behind the ROI.

        voxel_size : list, default=[1, 1]
            If the ROI does not have an associated image and is described only
            by contours, this will be the voxel sizes used in the x-y direction
            when converting the ROI to a mask if a mask plot type is selected.

        buffer : int, default=5
            If the ROI does not have an associated image and is described only
            by contours, this will be the number of buffer voxels 
            (i.e. whitespace) displayed around the ROI.
        """

        from skrt.better_viewer import BetterViewer
        self.load()
        
        # Set initial zoom amount and centre
        if self.image is not None and not self.contours_only:
            view = kwargs.get("init_view", "x-y")
            axes = skrt.image._plot_axes[view]
            extents = [self.get_length(ax=ax) for ax in axes]
            im_lims = [self.image.get_length(ax=ax) for ax in axes]
            zoom = min([im_lims[i] / extents[i] for i in range(2)]) * 0.8
            zoom_centre = self.get_centre()
            if zoom > 1:
                kwargs.setdefault("zoom", zoom)
                kwargs.setdefault("zoom_centre", zoom_centre)
            if "init_pos" not in kwargs and "init_slice" not in kwargs:
                kwargs["init_pos"] = \
                        self.get_centre()[skrt.image._slice_axes[view]]

        kwargs["show"] = False
        if "plot_type" in kwargs:
            kwargs["roi_plot_type"] = kwargs.pop("plot_type")

        # View with image
        if include_image and self.image is not None:
            bv = BetterViewer(self.image, rois=self, **kwargs)

        # View without image
        else:

            # Make dummy image for background
            roi_tmp = self.clone(copy_data=False)
            im = self.get_dummy_image(
                voxel_size=voxel_size,
                buffer=buffer
            )
            im.title = self.name
            roi_tmp.set_image(im)

            """
            roi_tmp = self

            # Make dummy image for background
            if self.contours_only:
                im = self.get_dummy_image(
                    voxel_size=voxel_size,
                    buffer=buffer
                )
                im.title = self.name
                roi_tmp = self.clone(copy_data=False)
                roi_tmp.set_image(im)
            else:
                im = skrt.image.Image(
                    np.ones(self.shape) * 1e4,
                    affine=self.affine,
                    title=self.name
                )
            """

            # Create viewer
            bv = BetterViewer(im, rois=roi_tmp, **kwargs)

            # Adjust UI
            no_ui = kwargs.get("no_ui", skrt.core.Defaults().no_ui)
            if not no_ui:
                bv.make_ui(no_roi=True, no_intensity=True)
                bv.show()
        return bv

    def _plot_mask(
        self,
        view,
        idx,
        mask_kwargs=None,
        opacity=None,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        color=None,
        flatten=False,
        show=True,
        **kwargs,
    ):
        """Plot the ROI as a mask."""

        self.create_mask()
        if self.empty:
            return
        mask_slice = self.get_slice(view, idx=idx, flatten=flatten)

        # Make colormap
        norm = matplotlib.colors.Normalize()
        cmap = matplotlib.cm.hsv
        s_colors = cmap(norm(mask_slice))
        if color is None:
            color = self.color
        s_colors[mask_slice > 0, :] = color
        s_colors[mask_slice == 0, :] = (0, 0, 0, 0)

        # Get plotting arguments
        if mask_kwargs is None:
            mask_kwargs = {}
        mask_kwargs.setdefault("alpha", opacity)
        mask_kwargs.setdefault("interpolation", "none")

        # Make plot
        if include_image and self.image is not None:
            im_mask = kwargs.get("mask", None)
            im_masked = kwargs.get("masked", True)
            im_invert_mask = kwargs.get("invert_mask", False)
            im_mask_color = kwargs.get("mask_color", "black")
            self.image.plot(view, idx=idx, ax=self.ax, show=False,
                    mask=im_mask, masked=im_masked, invert_mask=im_invert_mask,
                    mask_color=im_mask_color)
        else:
            if kwargs.get("title", None) is None:
                kwargs["title"] = self.name

        self.ax.imshow(s_colors, extent=self.mask.plot_extent[view], 
                       **mask_kwargs)

        # Adjust axes
        skrt.image.Image.label_ax(self, view, idx, **kwargs)
        skrt.image.Image.zoom_ax(self, view, zoom, zoom_centre)
        if show:
            plt.show()

    def _plot_contour(
        self,
        view,
        idx,
        contour_kwargs=None,
        linewidth=None,
        centroid=False,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        color=None,
        flatten=False,
        show=True,
        **kwargs,
    ):
        """Plot the ROI as a contour."""

        self.load()
        if (not self.on_slice(view, idx=idx) and not flatten) or self.empty:
            return

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs.setdefault("color", color)
        contour_kwargs.setdefault("linewidth", linewidth)

        # Plot underlying image
        if include_image and self.image is not None:
            im_mask = kwargs.get("mask", None)
            im_masked = kwargs.get("masked", True)
            im_invert_mask = kwargs.get("invert_mask", False)
            im_mask_color = kwargs.get("mask_color", "black")
            self.image.plot(view, idx=idx, ax=self.ax, show=False,
                    mask=im_mask, masked=im_masked, invert_mask=im_invert_mask,
                    mask_color=im_mask_color)
        else:
            if kwargs.get("title", None) is None:
                kwargs["title"] = self.name

        # Get contour points
        if flatten:
            mask = self.get_slice(view, idx=idx, flatten=True)
            contours = self.mask_to_contours(mask, view, invert=True)
        else:
            contours = self.get_contours(view, idx_as_key=True)[idx]

        # Plot contour
        for points in contours:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            self.ax.plot(points_x, points_y, **contour_kwargs)

        # Plot centroid point
        if centroid:
            if not flatten:
                centroid_points = self.get_centroid(
                    view, single_slice=True, idx=idx)
            else:
                centroid_3d = self.get_centroid()
                x_ax, y_ax = skrt.image._plot_axes[view]
                centroid_points = [centroid_3d[x_ax], centroid_3d[y_ax]]
            self.ax.plot(
                *centroid_points,
                "+",
                **contour_kwargs,
            )

        # Adjust axes
        self.ax.set_aspect("equal")
        skrt.image.Image.label_ax(self, view, idx, **kwargs)
        skrt.image.Image.zoom_ax(self, view, zoom, zoom_centre)
        if show:
            plt.show()

    def plot_comparison(
        self, 
        other, 
        view="x-y",
        sl=None,
        idx=None,
        pos=None,
        mid_slice_for_both=False,
        legend=True, 
        save_as=None, 
        names=None, 
        show=True, 
        include_image=False,
        legend_bbox_to_anchor=None,
        legend_loc="lower left",
        **kwargs
    ):
        """Plot comparison with another ROI. If no sl/idx/pos are given,
        the central slice of this ROI will be plotted, unless 
        mid_slice_for_both=True, in which case the central slice of both ROIs
        will be plotted (even though this may not correspond to the same 
        point in space).

        **Parameters:**
        
        other : ROI
            Other ROI with which to plot comparison.

        view : str, default="x-y"
            Orientation in which to plot ROIs. 

        sl : int, default=None
            Slice number. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        idx : int, default=None
            Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        pos : float, default=None
            Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
            <single_slice> is True, the central slice of the ROI will be used.

        mid_slice_for_both : bool, default=False
            If True and no sl/idx/pos are given, the central slices of both
            ROIs will be plotted, even if this does not correspond to the same
            point in space; otherwise, the position of the central slice of
            this ROI will be plotted for both.

        legend : bool, default=True
            If True, a legend will be added to the plot containing the names 
            of the two ROIs.

        save_as : str, default=None
            Path to a file at which the plot will be saved. If None, the plot
            will not be saved.

        names : list, default=None
            Custom names to use for the ROI legend in order [this ROI, other ROI].
            If None, the names in self.name and roi.name will be used.

        show : bool, default=True
            If True, the plot will be displayed.
        """

        # Ensure ROIs are plotted in different colours
        if self.color == other.color:
            roi2_color = ROIDefaults().get_default_roi_color()
        else:
            roi2_color = other.color

        # Get index to plot
        z_ax = skrt.image._slice_axes[view]

        # Compute position in mm to plot
        if sl is None and pos is None and idx is None:
            pos1 = self.idx_to_pos(self.get_mid_idx(view), z_ax)
        else:
            pos1 = self.idx_to_pos(
                self.get_idx(view, sl=sl, idx=idx, pos=pos), 
                z_ax
            )

        # Plot the same position for other ROI, unless mid_slice_for_both=True
        if not mid_slice_for_both:
            pos2 = pos1
        else:
            pos2 = other.idx_to_pos(other.get_mid_idx(view), z_ax)

        # Plot self
        self.plot(show=False, view=view, pos=pos1, include_image=include_image, 
                  **kwargs)

        # Adjust kwargs for plotting second ROI
        kwargs["ax"] = self.ax
        kwargs["color"] = roi2_color
        other.plot(show=False, view=view, pos=pos2, include_image=False, 
                   **kwargs)
        self.ax.set_title(self.get_comparison_name(other))

        # Create legend
        if legend:
            plot_type = kwargs.get("plot_type", None)
            if plot_type is None:
                plot_type = kwargs.get("roi_plot_type", None)
            linewidth = kwargs.get("linewidth", None)
            opacity = kwargs.get("opacity", None)
            if names:
                roi1_name = names[0]
                roi2_name = names[1]
            else:
                roi1_name = self.name
                roi2_name = other.name
            handles = [
                self.get_patch(
                    plot_type, self.color, opacity, linewidth, roi1_name),
                self.get_patch(
                    plot_type, roi2_color, opacity, linewidth, roi2_name),
            ]
            self.ax.legend(
                handles=handles, framealpha=1, facecolor="white", 
                bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc
            )

        # Show/save
        if show:
            plt.show()
        if save_as:
            plt.tight_layout()
            self.fig.savefig(save_as)

    def get_aspect_ratio(
        self, view, *args, **kwargs
    ):
        """Get aspect ratio for this structure in a given orientation."""

        x_ax, y_ax = skrt.image._plot_axes[view]
        return self.get_length(ax=skrt.image._axes[x_ax]) \
                / self.get_length(ax=skrt.image._axes[y_ax])

    def set_ax(self, plot_type, include_image, *args, **kwargs):
        """Set up axes."""

        if plot_type in ["contour", "centroid"] and not include_image:
            aspect_getter = self.get_aspect_ratio
        else:
            if self.image is not None:
                aspect_getter = self.image.get_plot_aspect_ratio
            else:
                self.create_mask()
                aspect_getter = self.mask.get_plot_aspect_ratio
        skrt.image.set_ax(self, *args, aspect_getter=aspect_getter, **kwargs)

    def get_zoom_centre(self, view):
        """Get coordinates to zoom in on this ROI."""

        zoom_centre = [None, None, None]
        x_ax, y_ax = skrt.image._plot_axes[view]
        x, y = self.get_centre(view=view, single_slice=True)
        zoom_centre[x_ax] = x
        zoom_centre[y_ax] = y
        return zoom_centre

    def write(self, outname=None, outdir=".", ext=None, overwrite=False,
            header_source=None, patient_id=None, root_uid=None,
            verbose=True, header_extras={}, keep_source_rois=True, **kwargs):

        self.load()

        # Generate output name if not given
        possible_ext = [".dcm", ".nii.gz", ".nii", ".npy", ".txt"]
        if ext is not None and not ext.startswith("."):
            ext = f".{ext}"
        if outname is None:
            if ext is None:
                ext = ".nii"
            else:
                if ext not in possible_ext:
                    raise RuntimeError(f"Unrecognised file extension: {ext}")
            if not ext.startswith("."):
                ext = f".{ext}"
            outname = f"{outdir}/{self.name}{ext}"

        # Otherwise, infer extension from filename
        else:

            # Find any of the valid file extensions
            for pos in possible_ext:
                if outname.endswith(pos):
                    ext = pos
            if ext not in possible_ext:
                raise RuntimeError(f"Unrecognised output file type: {outname}")

            outname = os.path.join(outdir, outname)

        # Write points to text file, in format that can be read by transformix
        if ext == ".txt":
            self.get_contours()

            points = []
            for z, contours in self.contours["x-y"].items():
                for idx, contour in enumerate(contours):
                    for point in contour:
                        # Add three digits to store the contour index,
                        # which is unlikely to need more than a single digit.
                        # This has negligible effect on precision,
                        # and allows contour-point associations
                        # to be identified in Transformix output.
                        z_out = f"{z:.3f}{idx:03}"
                        points.append(f"{point[0]:.3f} {point[1]:.3f} {z_out}")

            with open(outname, "w") as file:
                file.write("point\n")
                file.write(f"{len(points)}\n")
                file.write("\n".join(points))

            return

        # Write array to nifti or npy
        elif ext != ".dcm":
            self.create_mask()
            self.mask.write(outname, verbose=verbose, **kwargs)
        else:
            if header_source and keep_source_rois:
                structure_set = StructureSet(header_source)
            else:
                structure_set = StructureSet()
            structure_set.add_roi(self)
            structure_set.write(outdir=outdir, ext=ext, overwrite=overwrite,
                    header_source=header_source, patient_id=patient_id,
                    root_uid=root_uid, verbose=verbose,
                    header_extras=header_extras)

    def transform(self, scale=1, translation=[0, 0, 0], rotation=[0, 0, 0],
            centre=[0, 0, 0], resample="fine", restore=True, 
            fill_value=None, force_contours=False):
        """
        Apply three-dimensional similarity transform to ROI.

        If the transform corresponds to a translation and/or rotation
        about the z-axis, and either the roi source type is "dicom"
        or force-contours is True, the transform is applied to contour
        points and the roi mask is set as unloaded.  Otherwise the transform
        is applied to the mask and contours are set as unloaded.

        The transform is applied in the order: translation, scaling,
        rotation.  The latter two are about the centre coordinates.

        **Parameters:**
        
        force_contours : bool, default=False
            If True, and the transform corresponds to a translation
            and/or rotation about the z-axis, apply transform to contour
            points independently of the original data source.

        For other parameters, see documentation for
        skrt.image.Image.transform().  Note that the ``order``
        parameter isn't available for roi transforms - a value of 0
        is used always.
        """

        # Check whether transform is to be applied to contours
        small_number = 1.e-6
        transform_contours = False
        if self.source_type == 'contour' or force_contours:
            if abs(scale - 1) < small_number:
                if abs(rotation[0]) < small_number \
                        and abs(rotation[1]) < small_number:
                    transform_contours = True

        if transform_contours:

            # Apply transform to roi contours
            translation_2d = (translation[0], translation[1])
            centre_2d = (centre[0], centre[1])
            angle = rotation[2]
            new_contours = {}
            for key, contours in self.get_contours().items():
                new_key = key + translation[2]
                new_contours[new_key] = []
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    polygon = contour_to_polygon(contour)
                    polygon = affinity.translate(polygon, *translation_2d)
                    polygon = affinity.rotate(polygon, angle, centre_2d)
                    polygon = affinity.scale(polygon, scale, scale, scale,
                            centre_2d)
                    new_contours[new_key].append(polygon_to_contour(polygon))
                if not new_contours[new_key]:
                    new_contours.pop(new_key)

            self.reset_contours(new_contours)

        else:
            # Apply transform to roi mask
            self.create_mask()
            self.mask.transform(scale, translation, rotation,
                    centre, resample, restore, 0, fill_value)
            self.reset_mask()

    def crop(self, xlim=None, ylim=None, zlim=None):
        """
        Crop ROI to specified range in x, y, z.

        **Parameters:**

        xlim : tuple, default=None
            Lower and upper bounds in mm for cropping along x-axis.
            If set to None, no cropping is performed.  If a bound
            is set to None, that bound is disregarded.

        ylim : tuple, default=None
            Lower and upper bounds in mm for cropping along y-axis.
            If set to None, no cropping is performed.  If a bound
            is set to None, that bound is disregarded.

        zlim : tuple, default=None
            Lower and upper bounds in mm for cropping along z-axis.
            If set to None, no cropping is performed.  If a bound
            is set to None, that bound is disregarded.

        If only zlim is not set to None and the ROI source type is "contour",
        cropping is performed by discarding contours outside the required
        range.  Othwise cropping is performed on the ROI mask.
        """
        # For cropping along z, and where ROI is defined from contours,
        # discard contours outside crop range.
        if self.source_type == "contour" and zlim and not xlim and not ylim:
            contours = self.get_contours()
            if zlim[0] is None:
                zlim[0] = min(contours) - 1
            if zlim[1] is None:
                zlim[1] = max(contours) + 1
            contours = {z: z_contours for z, z_contours in contours.items()
                    if (z > zlim[0] and z < zlim[1])}

            # Reset contours.
            self.reset_contours(contours)

        # For all other cases, frop ROI mask.
        elif xlim or ylim or zlim:
            self.create_mask()
            
            # Ensure that crop range along each axis is None
            # or a two-element tuple of floats.
            lims = [xlim, ylim, zlim]
            mask_extents = self.mask.get_extents()
            for i_ax, lim in enumerate(lims):
                if lim:
                    for idx in [0, 1]:
                        if lim[idx] is None:
                            lims[i_ax][idx] = mask_extents[i_ax][idx]
                else:
                    lims[i_ax] = None

            # Loop over axes.
            for i_ax, lim in enumerate(lims):

                if lim is None:
                    continue

                # Find array indices at which to crop.
                i1 = self.mask.pos_to_idx(
                        lims[i_ax][0], ax=i_ax, return_int=False)
                i2 = self.mask.pos_to_idx(lims[i_ax][1], ax=i_ax,
                        return_int=False)
                i_big, i_small = i2, i1
                if i1 > i2:
                    i_big, i_small = i_small, i_big
                i_small = int(np.floor(i_small + 0.5))
                i_big = int(np.floor(i_big + 0.5))

                # Crop the mask.
                ax_to_slice = self.mask.get_axes().index(i_ax)
                if i_small > 0 and i_small < self.mask.n_voxels[i_ax]:
                    self.mask.data.swapaxes(0, ax_to_slice)[:i_small, :, :] = 0
                if i_big > 0 and i_big < self.mask.n_voxels[i_ax]:
                    self.mask.data.swapaxes(0, ax_to_slice)[i_big:, :, :] = 0

            # Reset mask.
            self.reset_mask(self.mask)

    def crop_by_amounts(self, dx=None, dy=None, dz=None):
        """
        Crop ROI by the amounts dx, dy, dz in mm.

        This method calls the function skrt.image.crop_by_amounts(), with
        self passed as object for cropping.

        The amount of cropping along each direction should be one of:
        - float : the ROI is cropped by this amount on both sides;
        - two-element tuple: the ROI is cropped on the sides of lower
          and higher values by the amounts specified;
        - None : no cropping is performed.

        For more details, see documentation of skrt.image.crop_by_amounts().
        """
        skrt.image.crop_by_amounts(self, dx, dy, dz)

    def crop_to_roi_length(self, other, ax="z"):
        """
        Crop to length of other ROI along a given axis.

        **Parameters:**
        other : skrt.structures.ROI
            ROI object to which to crop.
           
        ax : str/int, default="z"
            Axis along which to perform cropping. Should be one of
            ["x", "y", "z"] or [0, 1, 2].
        """
        # Determine crop range.
        lims = 3 * [None]
        i_ax = skrt.image._axes.index(ax) if isinstance(ax, str) else ax
        lims[i_ax] = other.get_extent(ax=ax)

        # Perform cropping.
        self.crop(*lims)

    def crop_to_roi(self, roi, **kwargs):
        """
        Crop ROI mask to region covered by an ROI.
        """
        self.create_mask()
        self.mask.crop_to_roi(roi, **kwargs)
        self.shape = self.get_mask().shape

    def add_plan(self, plan):
        """Add a Plan object to be associated with this ROI. This
        does not affect the structure set associated with the Plan object.

        **Parameters:**

        plan : skrt.dose.Plan
            A Plan object to assign to this structure set.
        """

        if not plan in self.plans:
            self.plans.append(plan)
            self.plans.sort()

    def clear_plans(self):
        """Clear all plan maps associated with this ROI."""

        self.plans = []

    def get_plans(self):
        """Return list of Plan objects associated with this ROI."""

        return self.plans

    def get_color_from_kwargs(self, kwargs={}, key='roi_colors'):
        '''
        Return ROI colour passed via dictionary of keyword arguments.

        **Parameters:**

        kwargs : dict, default={}
            Dictionary of keyword arguments, which may include
            parameter providing dictionary of ROI colours for plotting.

        key : str, default='roi_colors'
            Key in kwargs dictionary specifying parameter that, if
            present, provides dictionary of ROI colours.
        '''
        roi_colors = kwargs.get(key, {})
        return roi_colors.get(self.name, self.color)

    def split(self, voxel_size=None, names=None, order='x+'):
        '''
        Split a composite ROI into its components.

        A composite ROI may result because phyically separate ROIs are
        given the same label during manual contouring, or because
        multiple ROIs are returned by an auto-segmentation algorithm.

        The component ROIs are identified by labelling a mask of the
        composite, using scipy.ndimage.label().

        **Parameters:**

        voxel_size : tuple, default=None
            Voxel size (x, y) dimensions to use in creating mask for
            labelling.  This can be useful if the ROI source is contour
            points, and the resolution wanted is greater than is allowed
            by the voxel size of the ROI image.

        names: list, default=None
            List of names to be applied to the component ROIs.  Names
            are applied after any ROI ordering.

        order : str, default='x+'
            Specification of way it which ROIs are to be orderd:
            - 'x+' : in order of increasing x value.
            - 'x-' : in order of decreasing x value.
            - 'y+' : in order of increasing y value.
            - 'y-' : in order of decreasing y value.
            - 'z+' : in order of increasing z value.
            - 'z-' : in order of decreasing z value.
            If any other values is given, no ordering is performed.
        '''

        # If names not specified, make it an empty list.
        if names is None:
            names = []

        # Resize ROI mask if voxel size is to be changed.
        if voxel_size:
            roi = ROI(self)
            voxel_size_3d = (voxel_size[0], voxel_size[1], None)
            roi.create_mask()
            roi.mask.resize(voxel_size=voxel_size_3d, method='nearest',
                    image_size_unit='mm')
            roi.set_image(skrt.image.Image(roi.mask))
        else:
            roi = self
 
        # Label ROI components.
        label_mask, n_label = ndimage.label(roi.get_mask())

        if voxel_size and 'contour' == self.source_type:
            # Handle cases where composite ROI is from contour points,
            # and mask voxel size has been passed as an argument.
            # In this case component ROIs are initialised from contours
            # recreated from the mask, which may be higher-resolution
            # than the default.
            ss0 = StructureSet(label_mask, multi_label=True, image=roi.mask)
            ss = StructureSet()
            for roi in ss0.get_rois():
                ss.add_roi(ROI(source=roi.get_contours(), name=roi.name))
            ss.set_image(self.image)
        else:
            # Handle cases where component ROIs are initialised from mask.
            if voxel_size:
                # If voxel size changed for labelling,
                # recreate label mask with original voxel size.
                roi.mask.data = label_mask
                roi.mask.match_size(self.image, method='nearest')
                label_mask = roi.mask.data

            # Create structure set from label mask.
            ss = StructureSet(label_mask, multi_label=True, image=self.image)

        # Store ROIs in specified order.
        ss.order_rois(order)

        # Apply names.
        for i in range(min(len(names), len(ss.rois))):
            ss.rois[i].name = names[i]

        return ss

    def split_in_two(self, axis='x', v0=0, names=None,):
        '''
        Split a composite ROI into two component parts.

        Split an ROI into two components, either side of a specified
        plane.  This may be useful, for example, when left and right
        ROIs have been assigned a common label.

        For more general ROI splitting, see skrt.structure.ROI.split().
        Note that split() acts on binary masks, whereas split_in_two()
        acts on contours.

        **Parameters:**

        axis : 'str', default='x'
            Axis ('x', 'y' or 'z') perpendicular to plane about which
            ROI is to be split.

        v0 : float, default=0
            Coordinate along selected axis, specifying plane about which
            ROI is to be split.

        names: list, default=None
            List of names to be applied to the component ROIs.
        '''

        # If names not specified, initialise to empty list.
        if names is None:
            names = []

        # Based on selected axis, define view and default name suffixes.
        if 'x' == axis.lower():
            view = 'x-y'
            suffixes = ('right', 'left')
        elif 'y' == axis.lower():
            view = 'x-y'
            suffixes = ('anterior', 'posterior')
        elif 'z' == axis.lower():
            view = 'y-z'
            suffixes = ('inferior', 'superior')
            # In the 'y-z' view, y and z axes map to shapely x and y axes.
            axis = 'y'

        # Define polygon representing ROI contour as low or high,
        # depending on whether the polygon centroid is lower or higher
        # than the coordinate of the splitting plane.
        polygons_low = {}
        polygons_high = {}
        for z, polygons in self.get_polygons(view).items():
            for polygon in polygons:
                v = getattr(polygon.centroid, axis)
                if v < v0:
                    if not z in polygons_low:
                        polygons_low[z] = []
                    polygons_low[z].append(polygon)
                else:
                    if not z in polygons_high:
                        polygons_high[z] = []
                    polygons_high[z].append(polygon)


        # Create ROI from polygons below the splitting plane.
        if polygons_low:
            roi_low = ROI(source=polygons_low, image=self.image)
            roi_low.name = (names[0] if len(names) > 0
                    else f'{self.name}_{suffixes[0]}')
        else:
            roi_low = None

        # Create ROI from polygons above the splitting plane.
        if polygons_high:
            roi_high = ROI(source=polygons_high, image=self.image)
            roi_high.name = (names[1] if len(names) > 1
                    else f'{self.name}_{suffixes[1]}')
        else:
            roi_high = None

        # Create structure set for component ROI(s).
        rois = list(filter(None, [roi_low, roi_high]))
        if rois:
            ss = StructureSet(path=rois, name=self.name, image=self.image)
        else:
            ss = None

        return ss

    def get_patch(self, plot_type=None, color=None, opacity=None,
            linewidth=None, name=None):
        """
        Obtain patch reflecting ROI characteristics for plotting.

        The patch obtained can be used as a handle in legend creation.

        **Parameters:**

        plot_type : str, default=None
            Plotting type. If None, will be either "contour" or "mask" 
            depending on the input type of the ROI. Options:

                - "contour"
                - "mask"
                - "centroid" (contour with marker at centroid)
                - "filled" (transparent mask + contour)
                - "filled centroid" (filled with marker at centroid)

        color : matplotlib color, default=None
            Color with which to plot the ROI; overrides the ROI's own color. If 
            None, self.color will be used.

        opacity : float, default=None
            Opacity to use if plotting mask (i.e. plot types "mask", "filled", 
            or "filled centroid"). If None, opacity will be 1 by default for 
            solid mask plots and 0.3 by default for filled plots.

        linewidth : float, default=None
            Width of contour lines. If None, the matplotlib default setting 
            will be used.

        name : str, default=None
            Name identifying ROI.  If None, self.name will be used.
        """

        # Ensure that plotting characteristics are defined.
        if plot_type is None:
            plot_type = self.default_geom_method

        if color is None:
            color = self.color

        if opacity is None:
            opacity = 0.3 if "filled" in plot_type else 1

        if linewidth is None:
            linewidth = defaultParams["lines.linewidth"][0]

        if name is None:
            name = self.name

        # Set potentially different opacities for facecolor and edgecolor
        facecolor = matplotlib.colors.to_rgba(color, alpha=opacity)
        edgecolor = matplotlib.colors.to_rgba(color, alpha=1)

        # Define patch according to plot_type.
        patch = None
        if plot_type in ["contour", "centroid"]:
            patch = mpatches.Patch(edgecolor=edgecolor,
                    linewidth=linewidth, fill=False, label=name)
        elif plot_type in ["filled", "filled centroid"]:
            patch = mpatches.Patch(edgecolor=edgecolor,
                    facecolor=facecolor, linewidth=linewidth, label=name)
        elif plot_type in ["mask"]:
            patch = mpatches.Patch(facecolor=facecolor, label=name)

        if patch:
            patch.set_alpha(None)

        return patch

    def get_intensities_3d(self, image=None, standardise=True):
        """
        Return copy of an image array that has values retained
        inside self, and is set to zero elsewhere.

        **Parameter:**

        image : skrt.image.Image, default=None
            Image object for which 3D array with intensity values
            retained inside ROI, and NaN elsewhere, is to be obtained.
            If None, the image associated with the ROI is used.

        standardise : bool, default=False
            If False, the data array will be returned in the orientation in 
            which it was loaded; otherwise, it will be returned in standard
            dicom-style orientation such that [column, row, slice] corresponds
            to the [x, y, z] axes.
        """
        if image is None:
            # Consider image associated with self.
            self.create_mask()
            image = self.image
            roi = self
        else:
            # Clone self, and associate with clone the image specified in input.
            roi = self.clone()
            roi.set_image(image)

        # Obtain intensity values.
        im_data = image.get_data(standardise=standardise)
        mask = roi.get_mask(standardise=standardise)
        intensities_in_roi = np.full(im_data.shape, np.nan)
        intensities_in_roi[mask > 0] = im_data[mask > 0]
        return intensities_in_roi

    def get_intensities(self, image=None, standardise=True):
        """
        Return 1D numpy array containing all of the intensity values for the 
        voxels inside self.

        **Parameters:**

        image : skrt.image.Image, default=None
            Image object for which 1D array of intensity values is
            to be obtained.  If None, the image associated with the
            ROI is used.

        standardise : bool, default=False
            If False, the image's data array will be used in the orientation
            in which it was loaded; otherwise, it will be used in standard
            dicom-style orientation such that [column, row, slice] corresponds
            to the [x, y, z] axes.
        """
        intensities = self.get_intensities_3d(image, standardise)
        return intensities[~np.isnan(intensities)]
        
class StructureSet(skrt.core.Archive):
    """
    Class representing a radiotherapy structure set
    as a collection of regions of interest (ROIs).

    Attributes of a StructureSet object should usually be accessed via
    their getter methods, rather than directly, to ensure that
    attribute values have been loaded.
    """

    def __init__(
        self,
        path=None,
        name=None,
        image=None,
        load=True,
        names=None,
        keep_renamed_only=False,
        to_keep=None,
        to_remove=None,
        multi_label=False,
        colors=None,
        ignore_dicom_colors=False,
        auto_timestamp=False,
        alpha_beta_ratios=None,
        **kwargs
    ):
        """Load structure set from the source(s) given in <path>.

        **Parameters**:

        path : str/list, default=None
            Source(s) of ROIs to load into this structure set. Can be:

            - The path to a single dicom structure set file;
            - The path to a single nifti file containing an ROI mask;
            - A list of paths to nifti files containing ROI masks;
            - The path to a directory containing multiple nifti files 
            containing ROI masks;
            - A list of ROI objects;
            - A list of objects that could be used to initialise an ROI object
            (e.g. a numpy array).

        name : str, default=None
            Optional name to assign to this structure set. If None, will 
            attempt to infer from input filename.

        image : skrt.image.Image, default=None
            Image associated with this structure set. If the input source is a
            dicom file containing ROI contours, this image will be used when
            creating ROI binary masks.

        load : bool, default=True
            If True, ROIs will immediately be loaded from sources. Otherwise,
            loading will not occur until StructureSet.load() is called.
        
        names : dict/list, default=None
            Optional dict of ROI names to use when renaming loaded ROIs. Keys 
            should be desired names, and values should be lists of possible
            input names or wildcards matching input names.

            If multi_label=True, names can also be a list of names to assign
            to each label found in order, from smallest to largest.

        to_keep : list, default=None
            Optional list of ROI names or wildcards matching ROI names of ROIs
            to keep in this StructureSet during loading. ROIs not matching will
            be discarded. Note that the names should reflect those after 
            renaming via the <names> dict. If None, all ROIs will be kept.

        to_remove : list, default=None
            Optional list of ROI names or wildcards matching ROI names of ROIs
            to remove from this StructureSet during loading. ROIs matching will
            be discarded. Note that the names should reflect those after 
            renaming via the <names> dict, and removal occurs after filtering
            with the <to_keep> list. If None, no ROIs will be removed.

        multi_label : bool, default=False
            If True, will look for multiple ROI masks with different labels 
            inside the array and create a separate ROI from each.

        colors : list/dict
            List or dict of colors. If a dict, the keys should be ROI names or
            wildcards matching ROI names, and the values should be desired 
            colors for ROIs matching that name. If a list, should contain
            colors which will be applied to loaded ROIs in order.

        auto_timestamp : bool default=False
            If true and no valid timestamp is found within the path string,
            timestamp generated from current date and time.

        alpha_beta_ratios : dict, default=None
            Dictionary where keys are ROI names and values are
            ratios for ROI tissues of coefficients, alpha, beta,
            in the linear-quadratic equation.  These ratios are
            used in calculating biologically effective dose, BED, from
            physical dose, D, delivered over n equal fractions:
                BED = D * (1 + (D/n) / (alpha/beta)).
            The attribute alpha_over_value is set to the dictionary value
            for an ROI identified by a key, or is set to None for any
            other ROIs.

        `**`kwargs :
            Additional keyword args to use when initialising new ROI objects.
        """

        # Clone from another StructureSet object
        if issubclass(type(path), StructureSet):
            path.clone_attrs(self)
            return

        self.name = name
        path = (skrt.core.fullpath(path)
                if isinstance(path, (str, Path)) else path)
        self.sources = path
        if self.sources is None:
            self.sources = []
        elif not skrt.core.is_list(path):
            self.sources = [path]
        self.rois = []
        self.set_image(image)
        self.to_keep = to_keep
        self.to_remove = to_remove
        self.names = names
        self.keep_renamed_only = keep_renamed_only
        self.multi_label = multi_label
        self.colors = colors
        self.ignore_dicom_colors = ignore_dicom_colors
        self.dicom_dataset = None
        self.alpha_beta_ratios = alpha_beta_ratios or {}
        self.roi_kwargs = kwargs
        self.plans = []
        path = path if isinstance(path, str) else ""
        skrt.core.Archive.__init__(self, path, auto_timestamp)
        if self.path and not self.name:
            name = Path(self.path).name.split('_')[0].lower()
            if name[0].isalpha():
                self.name = name
        self.summed_names = []

        self.loaded = False
        if load:
            self.load()

    def __getitem__(self, roi):
        if isinstance(roi, int):
            return self.get_rois()[roi] 
        elif isinstance(roi, str):
            return self.get_roi_dict()[roi]

    def __iter__(self):
        return StructureSetIterator(self)

    def __add__(self, other):
        '''
        Define addition of StructureSet instances.

        The result of the addition of self and other is a StructureSet
        object that contains the (cloned) ROIs of both.  A result ROI
        will be prefixed by the name of the StructureSet from which it
        originates.  If the ROI name before the addition has the StructureSet
        name as prefix, no new prefix is added.  There is no requirement
        that ROIs or their names be unique within a StructureSet.  Adding
        a StructureSet to itself will give a result including duplicate
        ROIs with duplicate names.
        '''
        # Ensure that StructureSet data are loaded.
        self.load()
        other.load()

        # Create new StructureSet instance for sum,
        # and store names of contributors.
        result = StructureSet()
        result.summed_names = []
        for names in [self.summed_names, other.summed_names,
                [self.name], [other.name]]:
            for name in names:
                if isinstance(name, str):
                    for sub_name in name.split("_"):
                        if sub_name not in result.summed_names:
                            result.summed_names.append(sub_name)
        result.summed_names.sort()

        # Loop over StructureSets and their ROIs.
        for ss in [self, other]:
            for roi in ss.get_rois():
                roi_clone = roi.clone()
                # Try to prefix ROI name with the name of the StructureSet
                # from which it originates.
                if (roi.name.split("_")[0] in result.summed_names
                        or ss.name is None):
                    roi_clone.name = roi.name
                else:
                    roi_clone.name = "_".join([ss.name, roi.name])
                # Add ROI to the result.
                result.add_roi(roi_clone)

        # As result image set the image of self if non-null,
        # or othwise the image of other.
        image = self.image if self.image else other.image
        result.set_image(image)

        # Set the result name to be the concatenation of the names
        # of the contributors.
        if result.summed_names:
            result.name = "_".join(result.summed_names)

        return result

    def __iadd__(self, other):
        '''
        Define in-place of StructureSet instances.

        The result of the addition of self and other is a StructureSet
        object that contains the (cloned) ROIs of both.  A result ROI
        will be prefixed by the name of the StructureSet from which it
        originates.  If the ROI name before the addition has the StructureSet
        name as prefix, no new prefix is added.  There is no requirement
        that ROIs or their names be unique within a StructureSet.  Adding
        a StructureSet to itself will give a result including duplicate
        ROIs with duplicate names.
        '''
        return self + other

    def load(self, sources=None, force=False):
        """Load structure set from source(s). If None, will load from own
        self.sources."""

        if self.loaded and not force and sources is None:
            return

        if sources is None:
            sources = self.sources

        # Laod from multi-label array
        if self.multi_label:

            if isinstance(sources, list) and len(sources) == 1:
                sources = sources[0]
            if not isinstance(sources, str) and not isinstance(sources, np.ndarray):
                raise TypeError("Input for a multi-label image must be filepath "
                                f"or numpy array. Type found: {type(sources)}.")
            single_source = True

            # Load input array into image
            array = skrt.image.Image(sources).get_data(standardise=True).astype(int)
            n = array.max()

            # Enforce Dicom convention for data array of image object,
            # so that affine matrix will be correctly defined for mask creation.
            im = skrt.image.Image(sources)
            if 'nifti' in im.source_type:
                im = im.astype('dcm')
            i_name = 0
            for i in range(0, n):
                if self.names is not None and i_name < len(self.names):
                    name = self.names[i_name]
                    i_name += 1
                else:
                    name = f"ROI {i}"

                #im2.data = (im1.data == i + 1)
                self.rois.append(ROI(
                    array == i + 1, 
                    affine = im.get_affine(),
                    voxel_size = None,
                    origin = None,
                    image=self.image,
                    name=name,
                    **self.roi_kwargs
                ))
            sources = []

            # For NIfTI source, revert to NIfTI convention for data array
            # of mask, so that it can be written correctly to file.
            if 'nifti' in im.source_type:
                for idx in range(len(self.rois)):
                    self.rois[idx].mask = self.rois[idx].mask.astype('nii')

        else:
            if not skrt.core.is_list(sources) or isinstance(sources, np.ndarray):
                sources = [sources]
                single_source = True
            else:
                single_source = False

        # Expand any directories
        sources_expanded = []
        for source in sources:
            if isinstance(source, str) and os.path.isdir(source):

                sources_expanded.extend(
                    [os.path.join(source, file) for file in os.listdir(source)]
                )

                # Auto-assign name to directory name
                if single_source and self.name is None:
                    self.name = source

            elif not self.loaded:
                sources_expanded.append(source)

        for source in sorted(sources_expanded):

            if isinstance(source, ROI):
                self.rois.append(source)
                continue

            if isinstance(source, str):
                if os.path.basename(source).startswith(".") or source.endswith(".txt"):
                    continue
                if os.path.isdir(source):
                    continue

            # Attempt to load from dicom
            rois = []
            if isinstance(source, str):
                if os.path.splitext(source)[1] != '.nii':
                    rois, ds = load_rois_dicom(source)
            if len(rois):
                for number, roi in rois.items():

                    # Ignore entries with no contours
                    if "contours" not in roi:
                        continue

                    # Ignore entries containing only a single point
                    contours = roi["contours"]
                    if len(contours) == 1:
                        single_contour = contours[list(contours.keys())[0]][0]
                        if single_contour.shape[0] == 1:
                            continue

                    color = roi["color"] if not self.ignore_dicom_colors else None
                    self.rois.append(
                        ROI(
                            roi["contours"],
                            name=roi["name"],
                            color=color,
                            image=self.image,
                            **self.roi_kwargs
                        )
                    )
                    self.rois[-1].dicom_dataset = ds
                    self.rois[-1].number = number
                self.dicom_dataset = ds

                # Auto-assign name from dicom filename
                if single_source and self.name is None:
                    self.name = os.path.basename(source).replace(".dcm", "")

            # Load from ROI mask
            else:
                try:
                    self.rois.append(ROI(
                        source, 
                        image=self.image,
                        **self.roi_kwargs
                    ))
                except RuntimeError:
                    continue

        self.loaded = True

        self.rename_rois(keep_renamed_only=self.keep_renamed_only)
        self.filter_rois()
        self.recolor_rois(self.colors)
        slice_thicknesses = []
        rois = []
        for roi in self.rois:
            roi.structure_set = self
            roi.set_alpha_over_beta(self.alpha_beta_ratios.get(roi.name, None))
            for plan in self.plans:
                roi.add_plan(plan)

            if roi.source_type == "contour":
                if roi.slice_thickness_contours is not None:
                    slice_thicknesses.append(roi.slice_thickness_contours)
                else:
                    rois.append(roi)

        # If slice thickness isn't set for an ROI from contours,
        # set its thickness to the minimum for all other ROIs
        # in the structure set.
        if rois and slice_thicknesses:
            slice_thickness = min(slice_thicknesses)
            for roi in rois:
                roi.slice_thickness_contours = slice_thickness

    def get_dicom_dataset(self):
        """Return pydicom.dataset.FileDataset object associated with this Image,
        if loaded from dicom; otherwise, return None."""

        self.load()
        return self.dicom_dataset

    def reset(self):
        """Reload structure set from original source(s)."""

        self.rois = []
        self.loaded = False
        self.load(force=True)

    def reset_contours(self, contours=None, most_points=False):
        """
        Reset x-y contours for ROIs of structure set.

        **Parameters:**
        
        contours : dict, default=None
            Dictionary where keys are ROI names and values are dictionaries
            of slice and contour lists.  If None, an empty dictionary is
            used.  If an ROI has a name not included in the top-level
            dictionary, or has a name with associated value None, the
            ROI contours will be reset using the ROI's own x-y contours.

        most_points : bool, default=False
            If True, only the contour with most points for each slice
            is retained after resetting.  If False, all contours for each
            slice are retained.
        """
        contours = contours or {}
        for roi in self.get_rois():
            roi.reset_contours(contours.get(roi.name, None), most_points)

    def set_image_to_dummy(self, **kwargs):
        """
        Set image for self and all ROIs to be a dummy image covering
        the region containing all ROIs in the structure set.  The mask
        and x-z/y-z contours of each ROI will be cleared, so that they
        will be recreated when get_mask() or get_contours() are next called.

        **Parameters:**
        
        kwargs : 
            Keyword arguments to pass to self.get_dummy_image() when
            creating the dummy image. See documentation of
            StructureSet.get_dummy_image().
     """
        # Make image
        im = self.get_dummy_image(**kwargs)

        # Clear mask and contours
        for roi in self.rois:
            roi.input_contours = roi.get_contours("x-y")
            roi.contours = {"x-y": roi.input_contours}
            roi.loaded_mask = False
            roi.mask = None

        # Assign image
        self.set_image(im)

    def set_image(self, image, add_to_image=True):
        """Set image for self and all ROIs.
        If add_to_image is True, image.add_structure_set(self)
        will also be called."""

        # Convert to Image object if needed
        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)

        # Assign to self and all ROIs
        self.image = image
        for roi in self.rois:
            roi.set_image(image)

        # Assign self to the image
        if image is not None and add_to_image:
            image.add_structure_set(self)

    def get_image(self):
        """Return Image object associated with this StructureSet."""
        return self.image

    def contains(self, roi_names, in_image=False):
        """
        Return whether structure set contains named ROIs,
        optionally testing whether the ROIs be fully contained in an image.

        **Parameters:**

        roi_names : list/str
            Single string, or list of strings, indicating name(s) of ROI(s)
            to be considered.

        in_image : bool/skrt.image.Image, default=False
            Specification of whether to test that ROIs are fully contained
            in an image.  If in_image is in Image object, the boolean returned
            indicates whether all named ROIs are contained within this image.
            If in_image is True, the boolean returned indicates whether
            all named ROIs are contained within the image associated with
            self.
        """
        if isinstance(roi_names, str):
            roi_names = [roi_names]

        # Check whether structure set contains all named ROIs.
        ss_roi_names = self.get_roi_names()
        for roi_name in roi_names:
            if roi_name not in ss_roi_names:
                return False

        # Determine whether to check that named ROIs are contained in an image.
        if not in_image:
            return True

        # Check that image is specified.
        # If yes,  ensure that ROI masks have the same voxel size as this image.
        if isinstance(in_image, skrt.image.Image):
            im = in_image
            ss = self.clone()
            ss.set_image(im, add_to_image=False)
        else:
            im = self.image
            ss = self
        if not isinstance(im, skrt.image.Image):
            return False

        # Check whether ROIs are contained in image.
        im_extents = im.get_extents()
        for roi_name in roi_names:
            roi_extents = ss[roi_name].get_extents(0.5, "voxels")
            for idx in range(3):
                if roi_extents[idx][0] < im_extents[idx][0]:
                    return False
                if roi_extents[idx][1] > im_extents[idx][1]:
                    return False

        return True

    def missing(self, roi_names, in_image=False):
        """
        Identify among named ROIs those that are missing from structure set,
        or are not fully contained in an image.

        **Parameters:**

        roi_names : list/str
            Single string, or list of strings, indicating name(s) of ROI(s)
            to be considered.

        in_image : bool/skrt.image.Image, default=False
            Specification of whether to test that ROIs are fully contained
            in an image.  If in_image is in Image object, an ROI is
            identified as missing if not contained within this image.
            If in_image is True, an ROI is identified as missing if not
            contained within the image associated with self.
        """
        if isinstance(roi_names, str):
            roi_names = [roi_names]
        missing_rois = set()

        # Check for ROIs not contained in structure set.
        ss_roi_names = self.get_roi_names()
        for roi_name in roi_names:
            if roi_name not in ss_roi_names:
                missing_rois.add(roi_name)

        # Determine whether to check that named ROIs are contained in an image.
        if not in_image:
            return sorted(list(missing_rois))

        # Check that image is specified.
        # If yes,  ensure that ROI masks have the same voxel size as this image.
        if isinstance(in_image, skrt.image.Image):
            im = in_image
            ss = self.clone()
            ss.set_image(im, add_to_image=False)
        else:
            im = self.image
            ss = self
        if not isinstance(im, skrt.image.Image):
            raise RuntimeError(f"Invalid image specification")

        # Check whether ROIs are contained in image.
        im_extents = im.get_extents()
        active_rois = set(roi_names) - missing_rois
        for roi_name in active_rois:
            roi_extents = ss[roi_name].get_extents(0.5, "voxels")
            for idx in range(3):
                if (roi_extents[idx][0] < im_extents[idx][0]
                    or roi_extents[idx][1] > im_extents[idx][1]):
                    missing_rois.add(roi_name)

        return sorted(list(missing_rois))

    def has_rois(self, names=None, include_keys=False):
        """
        Determine whether structure set contains specified ROIs,
        identified by a nominal name or by an alias.

        This method returns True if all specified ROI names are matched
        with the name of an ROI in the structure set, and returns False
        otherwise.  Matching is case insensitive.

        names : str/list/dict
            Specification of ROIs:
            - a single ROI name, optionally including wildcards;
            - a list of ROI names, each optionally including wildcards;
            - a dictionary, where keys are nominal ROI names,
            and values are lists of possible names, optionally including
            wildcards.

        include_keys : bool, default=True
            If True, consider both keys and values for matching if ROI names
            are specified as a dictionary.  If False, consider values only.
        """
        # Convert names to a dictionary, where each value is a list.
        if isinstance(names, str):
            names = {names: [names]}
        elif skrt.core.is_list(names):
            names = {name: [name] for name in names}
        elif isinstance(names, dict):
            for name in names:
                if isinstance(names[name], str):
                    names[name] = [names[name]]

        # Exit is names isn't a dictionary.
        if not isinstance(names, dict):
            return False

        # Try to match each ROI name with nominal name or any aliases.
        for name, aliases in names.items():
            if include_keys:
                names_to_match = [name] + aliases
            else:
                names_to_match = aliases
            for name_to_match in names_to_match:
                for roi_name in self.get_roi_names():
                    match = fnmatch.fnmatch(
                            roi_name.lower(), name_to_match.lower())
                    if match:
                        break
                if match:
                    break
            if not match:
                return False

        return True

    def rename_rois(
        self, names=None, first_match_only=True, keep_renamed_only=False
    ):
        """Rename ROIs in this StructureSet.
        <first_match_only> is True, only the first ROI matching the
        possible matches will be renamed.

        **Parameters:**
        
        names : dict, default=None
            Dictionary of names for renaming ROIs, where the keys are new 
            names and values are lists of possible names of ROIs that should
            be assigned the new name. These names can also contain wildcards
            with the '*' symbol.

        first_match_only : bool, default=True
            If True, only the first ROI matching the possible names in the 
            values of <names> will be renamed; this prevents name duplication
            if multiple ROIs in the StructureSet are a match.

        keep_renamed_only : bool, default=False
            If True, any ROIs that do not get renamed will be removed from
            the StructureSet.
        """

        if names is None:
            names = self.names
        if not names or not isinstance(names, dict):
            return

        # Loop through each new name
        already_renamed = []
        for name, matches in names.items():

            if not skrt.core.is_list(matches):
                matches = [matches]

            # Loop through all possible original names
            name_matched = False
            for m in matches:

                # Loop through ROIs and see if there's a match
                for i, s in enumerate(self.get_rois()):

                    # Don't rename an ROI more than once
                    if i in already_renamed:
                        continue

                    if fnmatch.fnmatch(s.name.lower(), m.lower()):
                        s.name = name
                        name_matched = True
                        already_renamed.append(i)
                        if first_match_only:
                            break

                # If first_match_only, don't rename more than one ROI
                # with this new name
                if name_matched and first_match_only:
                    break

        # Keep only the renamed ROIs if requested
        if keep_renamed_only:
            renamed_rois = [self.rois[i] for i in already_renamed]
            self.rois = renamed_rois

    def filter_rois(self, to_keep=None, to_remove=None):
        """Keep only the ROIs in the to_keep list and remove any in the
        to_remove list.

        **Parameters:**
        

        to_keep : list, default=None
            List of names of ROIs to keep in the copied StructureSet; all 
            others will be removed. These names can also contain wildcards
            with the '*' symbol. 

        to_remove : list, default=None
            List of names of ROIs to remove from the copied StructureSet; all 
            others will be removed. These names can also contain wildcards
            with the '*' symbol. Applied after filtering with <to_keep>.
        """

        # Ensure to_keep and to_remove are lists
        if to_keep is None:
            to_keep = self.to_keep
        elif not skrt.core.is_list(to_keep):
            to_keep = [to_keep]
        if to_remove is None:
            to_remove = self.to_remove
        elif not skrt.core.is_list(to_remove):
            to_remove = [to_remove]

        # Keep only the ROIs in to_keep
        self.load()
        if to_keep is not None:
            keep = []
            for roi in self.rois:
                if any([fnmatch.fnmatch(roi.name.lower(), k.lower()) 
                        for k in to_keep]):
                    keep.append(roi)
            self.rois = keep

        # Remove the ROIs in to_remove
        if to_remove is not None:
            keep = []
            for roi in self.rois:
                if not any(
                    [fnmatch.fnmatch(roi.name.lower(), r.lower()) 
                     for r in to_remove]
                ):
                    keep.append(roi)
            self.rois = keep

    def set_alpha_beta_ratios(
            self, alpha_beta_ratios=None, set_as_default=True):
        """
        Set ratios for ROI tissues of coefficients of linear-quadratic equation.

        Optionally input values for ratios as structure-set default.

        **Parameter:**

        alpha_beta_ratios : dict, default=None
            Dictionary where keys are ROI names and values are
            ratios for ROI tissues of coefficients, alpha, beta,
            in the linear-quadratic equation.  These ratios are
            used in calculating biologically effective dose, BED, from
            physical dose, D, delivered over n equal fractions:
                BED = D * (1 + (D/n) / (alpha/beta)).
            The attribute alpha_over_value is set to the dictionary value
            for an ROI identified by a key, or is set to None for any
            other ROIs.  If this parameter is set to None, values are set
            from self.alpha_beta_ratios.

        set_as_default : bool, default=True
            If True, and alpha_beta_ratios isn't None, set
            self.alpha_beta_ratios to the value of alpha_beta_ratios.
        """
        if alpha_beta_ratios is None:
            alpha_beta_ratios = self.alpha_beta_ratios
        elif set_as_default:
            self.alpha_beta_ratios = alpha_beta_ratios

        for roi in self.get_rois():
            roi.set_alpha_over_beta(alpha_beta_ratios.get(roi.name, None))

    def get_alpha_beta_ratios(self):
        """
        Get dictionary of ratio of coefficients of linear-quadratic equation.
        """
        return {roi_name : self[roi_name].get_alpha_over_beta()
                for roi_name in sorted(self.get_roi_names())}

    def get_colors(self):
        """Get dict of ROI colors for each name."""

        return {roi.name: roi.color for roi in self.get_rois()}

    def recolor_rois(self, colors):
        """Set colors of ROIs using a list or dict given in <colors>.

        **Parameters**:

        colors : list/dict
            List or dict of colors. If a dict, the keys should be ROI names or
            wildcards matching ROI names, and the values should be desired 
            colors for ROIs matching that name. If a list, should contain
            colors which will be applied to loaded ROIs in order.
        """

        if colors is None:
            return

        if isinstance(colors, dict):
            for name, color in colors.items():
                for roi in self.rois:
                    if fnmatch.fnmatch(roi.name.lower(), name.lower()):
                        roi.set_color(color)
        elif isinstance(colors, list):
            for i, color in enumerate(colors):
                if i >= len(self.rois):
                    return
                self.rois[i].set_color(color)
        else:
            raise TypeError("<colors> must be list or dict.")

    def add_rois(self, sources):
        """Add additional ROIs from source(s)."""

        if not skrt.core.is_list(sources):
            sources = [sources]
        self.sources.extend(sources)
        self.load(sources)

    def add_roi(self, source, **kwargs):
        """Add a single ROI with  optional kwargs."""

        self.sources.append(source)
        if isinstance(source, ROI):
            roi = source
        else:
            roi_kwargs = self.roi_kwargs.copy()
            roi_kwargs.update(kwargs)
            roi = ROI(source, **roi_kwargs)
        roi.structure_set = self
        self.rois.append(roi)

    def clone_attrs(self, obj, copy_rois=True, copy_roi_data=True):
        """Assign all attributes of <self> to another object, <obj>, ensuring
        that ROIs are copied if copy_rois=True."""

        skrt.core.Data.clone_attrs(self, obj)
        if copy_rois:
            obj.rois = []
            for roi in self.rois:
                obj.rois.append(roi.clone(copy_data=copy_roi_data))

    def filtered_copy(
        self,
        names=None,
        name=None,
        to_keep=None,
        to_remove=None,
        keep_renamed_only=False,
        copy_roi_data=True
    ):
        """Create a copy of this structure set with ROIs optionally
        renamed or filtered. Returns a new StructureSet object.

        **Parameters:**
        
        names : dict, default=None
            Dictionary of names for renaming ROIs, where the keys are new 
            names and values are lists of possible names of ROIs that should
            be assigned the new name. These names can also contain wildcards
            with the '*' symbol.

        name : str, default=None
            Name for the returned StructureSet.

        to_keep : list, default=None
            List of names of ROIs to keep in the copied StructureSet; all 
            others will be removed. These names can also contain wildcards
            with the '*' symbol. Applied after renaming with <names>.

        to_remove : list, default=None
            List of names of ROIs to remove from the copied StructureSet; all 
            others will be removed. These names can also contain wildcards
            with the '*' symbol. Applied after renaming with <names> and 
            filtering with <to_keep>.

        keep_renamed_only : bool, default=False
            If True, any ROIs that do not get renamed will be removed from
            the StructureSet.

        copy_roi_data : bool, default=True
            If True, the ROIs in the returned StructureSet will contain
            copies of the data from the original StructureSet. Otherwise,
            the new ROIs will contain references to the same data, e.g. the 
            same numpy ndarray object for the mask/same dict for the contours.
            Only used if <copy_rois> is True
        """

        # Load StructureSet, but prevent loading of ROIs.
        # Loading ROIs here consumes time and memory creating masks
        # for ROIs that may be discarded after filtering.
        roi_load = self.roi_kwargs.get("load", None)
        self.roi_kwargs["load"] = False
        self.load()

        ss = self.clone(copy_roi_data=copy_roi_data)
        if name is not None:
            ss.name = name
        ss.rename_rois(names, keep_renamed_only=keep_renamed_only)
        ss.filter_rois(to_keep, to_remove)

        # Ensure that self.roi_kwargs is the same as at input.
        if roi_load is None:
            self.roi_kwargs.pop("load")
        else:
            self.roi_kwargs["load"] = roi_load

        return ss

    def get_rois(self, names=None, ignore_empty=False):
        """Get list of ROI objects If <names> is given, only the ROIs with
        those names will be returned."""

        self.load()
        if names is None:
            if ignore_empty:
                return [roi for roi in self.rois if not roi.empty]
            return self.rois

        rois = []
        for name in names:
            roi = self.get_roi(name)
            if roi is not None:
                if not ignore_empty or not roi.empty:
                    rois.append(roi)
        return rois

    def get_rois_wildcard(self, wildcard):
        """Return list of ROIs matching a wildcard expression."""

        rois = []
        for roi in self.get_rois():
            if fnmatch.fnmatch(roi.name.lower(), wildcard.lower()):
                rois.append(roi)
        return rois

    def get_roi_names(self, original=False, ignore_empty=False):
        """
        Get list of names of ROIs in this structure set. If <original> is True,
        get the original names of the ROIs.  If <ignore_empty> is True,
        omit names of empty ROIs.
        """

        return [s.get_name(original) for s in self.get_rois()
                if not (ignore_empty and s.empty)]


    def get_roi_dict(self):
        """Get dict of ROI names and objects."""

        return {s.name: s for s in self.get_rois()}

    def get_roi(self, name):
        """Get an ROI with a specific name."""

        rois = self.get_roi_dict()
        if name not in rois:
            print(f"ROI {name} not found!")
            return
        return rois[name]

    def print_rois(self):

        self.load()
        print("\n".join(self.get_roi_names()))

    def get_name(self):
        '''
        Load self and get name.
        '''

        self.load()
        return (self.name)

    def get_translation_to_align(
            self, other, z_fraction1=None, z_fraction2=None):
        """
        Determine translation for aligning <self> to <other>.

        **Parameters:**

        other : ROI/StructureSet/list
            ROI with which alignment is to be performed.  This can be
            specified directly as a single ROI.  Alternatively, it can
            be a StructureSet, or a list or ROI/StructureSet objects,
            in which case the individual ROIs will be combined.

        z_fraction1 : float, default=None
            Position along z axis of slice through <self> on which
            to align.  If None, alignment is to the centroid of the
            whole ROI volume.  Otherwise, alignment is to the
            centroid of the slice at the specified distance
            from the ROI's most-inferior point: 0 corresponds to
            the most-inferior point (lowest z); 1 corresponds to the
            most-superior point (highest z).  Values for z_fraction
            outside the interval [0, 1] result in a RuntimeError.

        z_fraction2 : float, default=None
            Position along z axis of slice through <other> on which
            to align.  If None, alignment is to the centroid of the
            whole ROI volume.  Otherwise, alignment is to the
            centroid of the slice at the specified distance
            from the ROI's most-inferior point: 0 corresponds to
            the most-inferior point (lowest z); 1 corresponds to the
            most-superior point (highest z).  Values for z_fraction
            outside the interval [0, 1] result in a RuntimeError.
            """
        return get_translation_to_align(
                self, other, z_fraction1, z_fraction2)

    def get_conformity_index(self, names=None, **kwargs):
        """
        Get conformity index for ROIs of the StructureSet.

        **Parameters:**

        names : list, default=None
            List of strings corresponding to ROI names.  If non-null,
            The conformity index is calculated relative for the ROIs
            with the listed names only.  Otherwise, the conformity
            index is calculated relative to all ROIs in the structure
            set.

        kwargs
            Keyword arguments are passed to
            skrt.structures.get_conformity_index()..get_surface_distances().
            See documentation of this function for parameter details.
        """
        return get_conformity_index(self.get_rois(names), **kwargs)

    def get_geometry(
        self, 
        name_as_index=True, 
        html=False, 
        colored=False,
        greyed_out=None,
        roi_kwargs={},
        **kwargs):
        """Get pandas DataFrame of geometric properties for all ROIs.
        If no sl/idx/pos is given, the central slice of each ROI will be used.

        **Parameters**:

        name_as_index : bool, default=True
            If True, the ROI names will be used as row indices; otherwise, they
            will be given their own column.

        html : bool, default=False
            If True, the table will be converted to HTML text color-coded based
            on ROI colors.
        """

        if html:
            name_as_index = False
        if greyed_out is None:
            greyed_out = []

        rows = []
        for roi in self.get_rois(ignore_empty=True):

            # Get DataFrame for this ROI
            df_row = roi.get_geometry(name_as_index=name_as_index, **kwargs)

            # Replace all values with "--" if greying out this ROI
            if roi in greyed_out:
                for col in df_row.columns:
                    if col == "ROI": 
                        continue
                    df_row[col] = "--"

            # Set ROI name to have colored background if requested
            if colored:
                color = roi.get_color_from_kwargs(roi_kwargs)
                col_str = get_colored_roi_string(
                        roi, grey=(roi in greyed_out), color=color)
                if name_as_index:
                    df_row.rename({df_row.index[0]: col_str}, inplace=True)
                else:
                    df_row.iloc[0, 0] = col_str

            rows.append(df_row)

        df = pd.concat(rows)

        # Reset index if not using ROI names as index
        if not name_as_index:
            df = df.reset_index(drop=True)

        # Convert to HTML if needed
        if html:
            return df_to_html(df)
        return df

    def get_comparison(
        self, 
        other=None, 
        comp_type="auto", 
        consensus_type="majority", 
        **kwargs
    ):
        """Get pandas DataFrame of comparison metrics vs a single ROI or
        another StructureSet.

        **Parameters**:

        other : ROI/StructureSet, default=None
            Object to compare own ROIs with. Can either be a single ROI, which
            will be compared to every ROI in this structure set, or another
            structure set. If None, a comparison will be performed between the 
            ROIs of this structure set.

        comp_type : str, default="auto"
            Method for selecting which ROIs should be compared if other=None
            or other is a StructureSet. Options are:

                - "all" : compare every ROI in one structure set to every ROI
                in the other (removing comparisons of an ROI to itself).
                - "by_name" : compare every ROI in one structure set
                to every ROI in the other with the same name. Useful when
                comparing two structure sets containing ROIs with the same
                names.
                - "consensus" : if other=None, compare each ROI in this 
                structure set to the consensus of all other ROIs; otherwise,
                compare each ROI in this structure set to the consensus of
                all ROIs in the other structure set. Consensus type is set via
                the consensus_type argument.
                - "auto" : if other=None, use "all" comparison. Otherwise, 
                first look for any ROIs with matching names; if at least one 
                matching pair is found, compare via the "by_name" comparison 
                type. If no matches are found, compare via the "all" comparison 
                type.

        consensus_type : str, default="majority vote"
            Method for calculating consensus of ROIs if using the "consensus"
            comparison type. Options are:

                - "majority" : use majority vote of ROIs (i.e. voxels where
                at least half of the ROIs exist).
                - "overlap" : use overlap of ROIs.
                - "sum" : use sum of ROIs.
                - "staple" : use the STAPLE algorithm to calculate consensus.

        html : bool, default=False
            If True, the table will be converted to HTML text color-coded based
            on ROI colors.

        greyed_out : bool, default=None
            List of ROIs that should be greyed out (given a grey background
            to their text) if returning an HTML string.

        `**`kwargs : 
            Keyword args to pass to ROI.get_comparison(). See 
            ROI.get_comparison() documentation for details.
        """

        dfs = []

        if isinstance(other, ROI):
            pairs = [(roi, other) for roi in self.get_rois(ignore_empty=True)]
        elif isinstance(other, StructureSet) or other is None:
            pairs = self.get_comparison_pairs(other, comp_type, consensus_type)
        else:
            raise TypeError("<other> must be ROI or StructureSet!")

        return compare_roi_pairs(pairs, **kwargs)

    def get_comparison_pairs(self, other=None, comp_type="auto", 
                             consensus_type="majority", 
                             consensus_color="blue"):
        """Get list of ROIs to compare with one another."""

        # Check comp_type is valid
        valid_comp_types = ["all", "by_name", "consensus", "auto"]
        if comp_type not in valid_comp_types:
            raise ValueError(f"Unrecognised comparison type {comp_type}")

        # Consensus comparison
        if comp_type == "consensus":
            
            # If comparing to another StructureSet, take consensus of that 
            # entire StructureSet
            if other is not None:
                consensus = other.get_consensus(consensus_type, color=consensus_color)
                return [(roi, consensus) for roi in self.get_rois(ignore_empty=True)]

            # Otherwise, compare each ROI to consensus of others
            pairs = []
            for roi in self.get_rois(ignore_empty=True):
                pairs.append((roi, self.get_consensus(consensus_type, 
                                                      color=consensus_color,
                                                      exclude=roi.name)))
            return pairs

        # Set default behaviour to "all" if other is None
        if other is None:
            other = self
            if comp_type == "auto":
                comp_type = "all"

        # Check for ROIs with matching names if using "auto" or "by_name"
        matches = []
        if comp_type in ["auto", "by_name"]:
            matches = [
                s for s in self.get_roi_names(ignore_empty=True)
                if s in other.get_roi_names(ignore_empty=True)
                ]
            if len(matches) or comp_type == "by_name":
                return [
                    (self.get_roi(name), other.get_roi(name)) for name in matches
                ]

        # Otherwise, pair each ROI with every other (exlcuding pairs of the same
        # ROI)
        pairs = []
        for roi1 in self.get_rois(ignore_empty=True):
            for roi2 in other.get_rois(ignore_empty=True):
                if roi1 is not roi2 and (roi2, roi1) not in pairs:
                    pairs.append((roi1, roi2))

        return pairs

    def get_mid_idx(self, view="x-y"):
        """Return the array index of the slice that contains the most ROIs."""

        indices = []
        for roi in self.get_rois(ignore_empty=True):
            indices.extend(roi.get_indices(view))
        return np.bincount(indices).argmax()

    def plot_comparisons(
        self, other=None, comp_type="auto", outdir=None, legend=True, 
        names=None, legend_bbox_to_anchor=None, legend_loc="lower left",
        **kwargs
    ):
        """Plot comparison pairs."""

        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        for roi1, roi2 in self.get_comparison_pairs(other, comp_type):

            outname = None
            if outdir:
                comp_name = roi1.get_comparison_name(roi2, True)
                outname = os.path.join(outdir, f"{comp_name}.png")

            if names is None and roi1.name == roi2.name:
                names = [self.name, other.name]

            roi1.plot_comparison(
                roi2, legend=legend, save_as=outname, names=names,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_loc=legend_loc, **kwargs
            )

    def plot_surface_distances(
        self, other, outdir=None, signed=False, comp_type="auto", **kwargs
    ):
        """Plot surface distances for all ROI pairs."""

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for roi1, roi2 in self.get_comparison_pairs(other, comp_type):
            comp_name = roi1.get_comparison_name(roi2, True)
            if outdir:
                outname = os.path.join(outdir, f"{comp_name}.png")
            else:
                outname = None
            roi1.plot_surface_distances(roi2, signed=signed, save_as=outname, **kwargs)

    def write(self, outname=None, outdir=".", ext=None, overwrite=False,
            header_source=None, patient_id=None, modality=None,
            root_uid=None, verbose=True, header_extras={}, **kwargs):
        """Write to a dicom StructureSet file or directory of nifti files."""

        if ext is not None and not ext.startswith("."):
            ext = f".{ext}"

        # Check whether to write to dicom file
        if isinstance(outname, str) and outname.endswith(".dcm"):
            ext = ".dcm"

        if ext == ".dcm":
            dicom_writer = DicomWriter(
                outdir=outdir,
                data=self,
                affine=None,
                overwrite=overwrite,
                header_source=header_source,
                orientation=None,
                patient_id=patient_id,
                modality='RTSTRUCT',
                root_uid=root_uid,
                header_extras=header_extras,
                source_type=type(self).__name__,
                outname = outname,
            )
            self.dicom_dataset = dicom_writer.write()

            if verbose:
                print("Wrote dicom file to directory:", outdir)
            return

        # Otherwise, write to individual ROI files
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif overwrite:
            shutil.rmtree(outdir)
            os.mkdir(outdir)
        for s in self.get_rois():
            s.write(outdir=outdir, ext=ext, verbose=verbose, **kwargs)
            
    def plot(
        self,
        view="x-y",
        plot_type="contour",
        sl=None,
        idx=None,
        pos=None,
        ax=None,
        opacity=None,
        linewidth=None,
        include_image=True,
        centre_on_roi=None,
        show=True,
        save_as=None,
        legend=False,
        legend_bbox_to_anchor=None,
        legend_loc="lower left",
        consensus_type=None,
        exclude_from_consensus=None,
        consensus_color="blue",
        consensus_linewidth=None,
        **kwargs,
    ):
        """Plot the ROIs in this structure set.
       
        If consensus_type is set to any of 'majority', 'sum', 'overlap', or 'staple',
        the conensus contour will be plotted rather than individual ROIs. If
        'exclude_from_consensus' is set to the name of an ROI, that ROI will
        be excluded from the consensus calculation and plotted individually.
        """

        # If no sl/idx/pos given, use the slice with the most ROIs
        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)

        if plot_type is None:
            plot_type = kwargs.get("roi_plot_type", self.default_geom_method)
        kwargs["roi_plot_type"] = plot_type

        # Ensure that linewidth and opacity for ROI plotting are defined.
        roi_kwargs = {}
        if opacity is None:
            opacity = 0.3 if "filled" in plot_type else 1
        roi_kwargs["opacity"] = opacity
        if linewidth is None:
            linewidth = defaultParams["lines.linewidth"][0]
        roi_kwargs["linewidth"] = linewidth

        # Ensure that title is set to be a string.
        if kwargs.get("title", None) is None:
            kwargs["title"] = self.name or ""

        # Plot with image
        if include_image and self.image is not None:

            self.image.plot(
                view,
                sl=sl, 
                idx=idx,
                pos=pos,
                rois=self,
                roi_kwargs=roi_kwargs,
                centre_on_roi=centre_on_roi,
                show=show,
                save_as=save_as,
                legend=legend,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_loc=legend_loc,
                consensus_type=consensus_type,
                exclude_from_consensus=exclude_from_consensus,
                ax=ax,
                **kwargs
            )
            return

        # Plot consensus
        roi_handles = []
        if consensus_type is not None:

            # Plot consensus contour
            consensus = self.get_consensus(consensus_type, 
                                           color=consensus_color,
                                           exclude=exclude_from_consensus)
            consensus_kwargs = {} if exclude_from_consensus is not None \
                    else kwargs
            if consensus_linewidth is None:
                consensus_linewidth = defaultParams["lines.linewidth"][0] + 1

            consensus.plot(
                view, sl=sl, idx=idx, pos=pos, plot_type=plot_type, 
                color=consensus_color, linewidth=consensus_linewidth, 
                opacity=opacity, show=False, ax=ax, **consensus_kwargs)

            self.ax = consensus.ax
            self.fig = consensus.fig

            if legend:
                roi_handles.append(
                        consensus.get_patch(plot_type, consensus_color, opacity,
                            linewidth, consensus.name))

            # Plot excluded ROI on top
            if exclude_from_consensus is not None:
                kwargs["include_image"] = False
                excluded = self.get_roi(exclude_from_consensus)
                excluded.plot(view, sl=sl, idx=idx, pos=pos, plot_type=plot_type,
                              opacity=opacity, linewidth=linewidth, show=False,
                              ax=self.ax, **kwargs)
                if legend:
                    roi_handles.append(
                        excluded.get_patch(plot_type, excluded.color, opacity,
                            linewidth, excluded.name))

        # Otherwise, plot first ROI and get axes
        else:
            if centre_on_roi is not None:
                central = self.get_roi(centre_on_roi)
                idx = central.get_mid_idx(view)
                sl = None
                pos = None
                first_roi = central
            else:
                central = self.get_rois(ignore_empty=True)[0]

            central.plot(view, sl=sl, idx=idx, pos=pos, plot_type=plot_type,
                         opacity=opacity, linewidth=linewidth, show=False,
                         ax=ax, **kwargs)

            self.fig = central.fig
            self.ax = central.ax

            if legend:
                roi_handles.append(
                        central.get_patch(plot_type, central.color, opacity,
                            linewidth, central.name))

            # Plot other ROIs
            for i, roi in enumerate(self.get_rois(ignore_empty=True)):

                if roi is central:
                    continue

                plot_kwargs = {} if i < len(self.rois) - 1 else kwargs
                roi.plot(view, sl=sl, idx=idx, pos=pos, plot_type=plot_type,
                         opacity=opacity, linewidth=linewidth, show=False,
                         ax=self.ax, **kwargs)

                if legend:
                    if (idx is None and pos is None and sl is None) or \
                       roi.on_slice(view, sl=sl, idx=idx, pos=pos):
                        roi_handles.append(roi.get_patch(plot_type, roi.color,
                            opacity, linewidth, roi.name))

        # Draw legend
        if legend and len(roi_handles):
            self.ax.legend(handles=roi_handles,
                    bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc,
                    facecolor="white", framealpha=1
            )

        # Display image
        plt.tight_layout()
        if show:
            plt.show()

        # Save to file
        if save_as:
            self.fig.savefig(save_as)
            plt.close()

    def plot_consensus(self, consensus_type, view="x-y", sl=None, idx=None,
                       pos=None, rois_in_background=False, color=None, 
                       show=True, **kwargs):
        """Plot the consensus contour, with all ROIs in grey behind it if
        rois_in_background=True."""

        consensus = self.get_consensus(consensus_type)

        if idx is None and sl is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = consensus.get_idx(view, sl, idx, pos)
        ax = skrt.image._slice_axes[view]
        pos = consensus.idx_to_pos(idx, ax)

        if consensus.empty:
            print(f"{consensus_type} contour is empty")
            return
        consensus.plot(color=color, pos=pos, view=view, show=False, **kwargs)

        if rois_in_background:
            kwargs["ax"] = consensus.ax
            for roi in self.get_rois(ignore_empty=True):
                roi.plot(color="lightgrey", view=view, pos=pos, show=False, **kwargs)
            consensus.plot(color=color, view=view, pos=pos, show=False, **kwargs)

        consensus.ax.set_aspect("equal")

        if show:
            plt.show()

    def get_length(self, ax):
        """Get length covered by ROIs along an axis."""

        extent = self.get_extent(ax=ax)
        return abs(extent[1] - extent[0])

    def get_centre(self, **kwargs):
        """Get 3D centre of the area covered by ROIs."""

        kwargs.pop("ax", None)
        extents = [self.get_extent(ax=ax, **kwargs) for ax in skrt.image._axes]
        return [np.mean(ex) for ex in extents]

    def find_most_populated_slice(self, view="x-y"):
        """Find the index of the slice with the most ROIs on it."""

        indices = []
        for roi in self.get_rois(ignore_empty=True):
            indices.extend(roi.get_indices(view))
        vals, counts = np.unique(indices, return_counts=True)
        return vals[np.argmax(counts)]

    def view(self, include_image=True, rois=None, voxel_size=[1, 1], buffer=5, **kwargs):
        """View the StructureSet.

        **Parameters:**
        
        include_image : bool, default=True
            If True and this StructureSet has an associated image 
            (in self.image), the image will be displayed behind the ROIs.

        rois : StructureSet/list, default=None
            Any other ROIs or StructureSets to include in the plot.

        voxel_size : list, default=[1, 1]
            If the StructureSet does not have an associated image and has ROIs 
            described only by contours, this will be the voxel sizes used in 
            the x-y direction when converting ROIs to masks if a mask plot 
            type is selected.

        buffer : int, default=5
            If the StructureSet does not have an associated image and has ROIs
            described only by contours, this will be the number of buffer voxels 
            (i.e. whitespace) displayed around the extent of the ROIs.
        """

        from skrt.better_viewer import BetterViewer
        self.load()

        # Set initial zoom amount and centre
        self.rois[0].load()
        if self.image is not None and not self.get_rois(ignore_empty=True)[0].contours_only:
            view = kwargs.get("init_view", "x-y")
            axes = skrt.image._plot_axes[view]
            extents = [self.get_length(ax=ax) for ax in axes]
            im_lims = [self.image.get_length(ax=ax) for ax in axes]
            zoom = min([im_lims[i] / extents[i] for i in range(2)]) * 0.8
            zoom_centre = self.get_centre()
            if zoom > 1:
                kwargs.setdefault("zoom", zoom)
                kwargs.setdefault("zoom_centre", zoom_centre)

            # Set initial slice
            kwargs.setdefault(
                "init_slice", 
                self.image.idx_to_slice(self.find_most_populated_slice(view),
                                        skrt.image._slice_axes[view]))

        if "plot_type" in kwargs:
            kwargs["roi_plot_type"] = kwargs.pop("plot_type")

        # Ensure that title is set to be a string.
        if kwargs.get("title", None) is None:
            kwargs["title"] = self.name or ""

        # View with image
        if include_image and self.image is not None:
            if rois is None:
                rois = self
            else:
                if skrt.core.is_list(rois):
                    rois = [self] + rois
                else:
                    rois = [self, rois]
            bv = BetterViewer(self.image, rois=rois, **kwargs)

        # View without image
        else:

            structure_set_tmp = self

            # Make dummy image
            im = self.get_dummy_image(buffer=buffer, 
                                      voxel_size=voxel_size)
            structure_set_tmp = self.clone(
                copy_rois=True, 
                copy_roi_data=False,
            )
            structure_set_tmp.set_image(im)
            """
            if self.rois[0].contours_only or True:
                im = self.get_dummy_image(buffer=buffer, 
                                          voxel_size=voxel_size)
                structure_set_tmp = self.clone(
                    copy_rois=True, 
                    copy_roi_data=False
                )
                for roi in structure_set_tmp.rois:
                    if roi.contours_only:
                        roi.set_image(im)
            else:
                im = skrt.image.Image(
                    np.ones(self.rois[0].shape) * 1e4,
                    affine=self.rois[0].affine,
                )
            """

            if rois is None:
                rois = structure_set_tmp
            else:
                if skrt.core.is_list(rois):
                    rois = [structure_set_tmp] + rois
                else:
                    rois = [structure_set_tmp, rois]
            kwargs["show"] = False
            bv = BetterViewer(im, rois=rois, **kwargs)

            # Adjust UI
            no_ui = kwargs.get("no_ui", skrt.core.Defaults().no_ui)
            if not no_ui:
                bv.make_ui(no_intensity=True)
                bv.show()

        return bv

    def get_extent(self, **kwargs):
        """Get min and max extent of all ROIs in the StructureSet."""

        all_extents = []
        for roi in self.get_rois(ignore_empty=True):
            all_extents.extend(roi.get_extent(**kwargs))
        return min(all_extents), max(all_extents)

    def get_extents(self, roi_names=None, buffer=None, buffer_units="mm",
            method=None, origin=None):
        """
        Get minimum and maximum extent of StructureSet ROIs,
        in mm along all three axes, returned in order [x, y, z].
        Optionally apply a buffer to the extents such that they cover
        more than the region of the ROIs.

        **Parameters:**

        roi_names : list, default=None
            List of names of ROIs to be considered.  If None, all of the
            structure set's ROIs are considered.

        buffer : float, default=None
            Optional buffer to add to the extents. Units set by `buffer_units`.

        buffer_units : str, default="mm"
            Units for buffer, if using. Can be "mm", "voxels", or "frac" (which
            applies buffer as a fraction of total length in each dimension).

        method : str, default=None
            Method to use for extent calculation. Can be: 

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the 
                  binary mask.
                * None: use the method set in self.default_geom_method.

        origin : tuple, default=None
            Tuple specifying the (x, y, z) coordinates of the point
            with respect to which extents are to be determined.
            If None, then (0, 0, 0) is used.
        """
        return self.combine_rois(roi_names).get_extents(
                buffer, buffer_units, method, origin)

    def get_crop_limits(self, roi_names=None, crop_margins=None, method=None):
        """
        Get crop limits corresponding to StructureSet ROI extents plus margins.

        The tuples of limits returned, in the order (x, y, z) can
        be used, for example, as inputs to skrt.image.Image.crop().

        This method is similar to the method get_extents(), but allows
        different margins on each side of the StructureSet.

        **Parameters:**

        roi_names : list, default=None
            List of names of ROIs to be considered.  If None, all of the
            structure set's ROIs are considered.

        crop_margins : float/tuple, default=None
            Float or three-element tuple specifying the margins, in mm,
            to be added to StructureSet extents.  If a float, minus and plus the
            value specified are added to lower and upper extents respectively
            along each axis.  If a three-element tuple, elements are
            taken to specify margins in the order (x, y, z).  Elements
            can be either floats (minus and plus the value added respectively
            to lower and upper extents) or two-element tuples (elements 0 and 1
            added respectively to lower and upper extents).

        method : str, default=None
            Method to use for extent calculation. Can be:

                * "contour": get extent from min/max positions of contour(s).
                * "mask": get extent from min/max positions of voxels in the
                  binary mask.
                * None: use the method set in self.default_geom_method.
        """
        return self.combine_rois(roi_names).get_crop_limits(
                crop_margins, method)

    def get_bbox_centre_and_widths(self,
            roi_names=None, buffer=None, buffer_units="mm", method=None):
        """
        Get centre and widths in mm along all three axes of a bounding box
        enclosing StructureSet ROIs and optional buffer.  Centre
        and widths are returned as a tuple ([x, y, z], [dx, dy, dz]).

        Method parameters are passed to
        skrt.structures.StructureSet.get_extents() to obtain StructureSet
        extents.  For parameter explanations, see
        skrt.structures.StructureSet.get_extents() documentation.
        """
        extents = self.get_extents(roi_names, buffer, buffer_units, method) 
        centre = [0.5 * (extent[0] + extent[1]) for extent in extents]
        widths = [(extent[1] - extent[0]) for extent in extents]
        return (centre, widths)

    def get_dummy_image(self, **kwargs):
        """Make a dummy image that covers the area spanned by all ROIs in this
        StructureSet. Returns an Image object.

        **Parameters:**
        
        voxel_size : list, default=None
            Voxel size in mm in the dummy image in the x-y plane, given as 
            [vx, vy]. If <shape> and <voxel_size> are both None, voxel sizes of
            [1, 1] will be used by default. The voxel size in the z direction 
            is defined by <slice_thickness>.

        shape : list, default=None
            Number of voxels in the dummy image in the x-y plane, given as
            [nx, ny]. Only used if <voxel_size> is None. 
            The number of voxels in the z direction will be taken from the 
            number of slices in the x-y contours dictionary.

        fill_val : int/float, default=1e4
            Value with which the voxels in the dummy image should be filled. 

        buffer : int, default=1
            Number of empty buffer voxels to add outside the ROI in each
            direction.

        slice_thickness : float, default=None
            Voxel size in mm in the dummy image in the z direction.  If None,
            the value used is the minimum distance between slice positions
            in the x-y contours dictionary.
        """
        extents = [self.get_extent(ax=ax) for ax in skrt.image._axes]
        slice_thickness = kwargs.pop("slice_thickness", None)
        slice_thickness = (slice_thickness or
                self.get_rois(ignore_empty=True)[0]\
                        .get_slice_thickness_contours())
        return create_dummy_image(extents, slice_thickness, **kwargs)

    def _get_combination(self, name, force, exclude, combo_maker, **kwargs):
        """Either get an already calculated combination from the cache
        or create it, cache it, and return it."""

        # Return cached result if it exists and not forcing
        attr = f"_{name}"
        if exclude:
            attr += f"_excluded"
        if not force:
            if exclude is None and hasattr(self, attr):
                return getattr(self, attr)
            else:
                if hasattr(self, attr) and exclude in getattr(self, attr):
                    return getattr(self, attr)[exclude]

        # Get list of ROIs to include in this combination ROI
        if exclude is not None:

            if exclude not in self.get_roi_names():
                print(f"ROI to exclude {exclude} not found.")
                return

            rois_to_include = [roi for roi in self.rois if roi.name != exclude]

            if not hasattr(self, attr):
                setattr(self, attr, {})
        else:
            rois_to_include = self.rois

        # Make kwargs for ROI creation
        roi_kwargs = self.roi_kwargs.copy()
        roi_kwargs.update(kwargs)
        roi_kwargs["name"] = name
        roi_kwargs["image"] = self.image
        roi_kwargs["affine"] = rois_to_include[0].get_affine(standardise=True)

        # Create the ROI
        roi = combo_maker(rois_to_include, **roi_kwargs)

        # Add it to the cache and return
        if exclude is None:
            setattr(self, attr, roi)
        else:
            getattr(self, attr)[exclude] = roi
        return roi

    def get_consensus(self, consensus_type, color="blue", **kwargs):

        # Get consensus calculation function
        if consensus_type == "majority":
            consensus_func = StructureSet.get_majority_vote
        elif consensus_type == "sum":
            consensus_func = StructureSet.get_sum
        elif consensus_type == "overlap":
            consensus_func = StructureSet.get_overlap
        elif consensus_type == "staple":
            consensus_func = StructureSet.get_staple
        else:
            raise ValueError(f"Unrecognised consensus type: {consensus_type}")

        return consensus_func(self, color=color, **kwargs)

    def get_staple(self, force=False, exclude=None, **kwargs):
        """Get ROI object representing the STAPLE combination of ROIs in this 
        structure set. If <exclude> is set to a string, the ROI with that name 
        will be excluded from the calculation.

        **Parameters:**

        force : bool, default=False
            If False and STAPLE ROI has already been created, the 
            previously computed ROI will be returned. If Force=True, the 
            STAPLE ROI will be recreated.

        exclude : str, default=None
            If set to a string, the first ROI in self.rois with that name will 
            be excluded from the combination. This may be useful if comparison 
            of a single ROI with the consensus of all others is desired.

        `**`kwargs :
            Extra keyword arguments to pass to the creation of the ROI object
            representing the combination.
        """

        return self._get_combination("staple", force, exclude, create_staple,
                                     **kwargs)

    def get_majority_vote(self, force=False, exclude=None, **kwargs):
        """Get ROI object representing the majority vote combination of ROIs in 
        this structure set. If <exclude> is set to a string, the ROI with that 
        name will be excluded from the calculation.

        **Parameters:**

        force : bool, default=False
            If False and majority vote ROI has already been created, the 
            previously computed ROI will be returned. If Force=True, the 
            majority vote ROI will be recreated.

        exclude : str, default=None
            If set to a string, the first ROI in self.rois with that name will 
            be excluded from the combination. This may be useful if comparison 
            of a single ROI with the consensus of all others is desired.

        `**`kwargs :
            Extra keyword arguments to pass to the creation of the ROI object
            representing the combination.
        """

        return self._get_combination("majority_vote", force, exclude, 
                                     create_majority_vote, **kwargs)

    def get_sum(self, force=False, exclude=None, **kwargs):
        """Get ROI object representing the sum of ROIs in this structure set. 
        If <exclude> is set to a string, the ROI with that name will be 
        excluded from the calculation.

        **Parameters:**

        force : bool, default=False
            If False and sum ROI has already been created, the 
            previously computed ROI will be returned. If Force=True, the 
            sum ROI will be recreated.

        exclude : str, default=None
            If set to a string, the first ROI in self.rois with that name will 
            be excluded from the combination. This may be useful if comparison 
            of a single ROI with the consensus of all others is desired.

        `**`kwargs :
            Extra keyword arguments to pass to the creation of the ROI object
            representing the combination.
        """

        return self._get_combination("sum", force, exclude, 
                                     create_roi_sum, **kwargs)

    def get_overlap(self, force=False, exclude=None, **kwargs):
        """Get ROI object representing the overlap of ROIs in this 
        structure set. If <exclude> is set to a string, the ROI with that name 
        will be excluded from the calculation.

        **Parameters:**

        force : bool, default=False
            If False and overlap ROI has already been created, the 
            previously computed ROI will be returned. If Force=True, the 
            overlap ROI will be recreated.

        exclude : str, default=None
            If set to a string, the first ROI in self.rois with that name will 
            be excluded from the combination. This may be useful if comparison 
            of a single ROI with the consensus of all others is desired.

        `**`kwargs :
            Extra keyword arguments to pass to the creation of the ROI object
            representing the combination.
        """

        return self._get_combination("overlap", force, exclude, 
                                     create_roi_overlap, **kwargs)

    def transform(self, scale=1, translation=[0, 0, 0], rotation=[0, 0, 0],
            centre=[0, 0, 0], resample="fine", restore=True, 
            fill_value=None, force_contours=False, names=None):
        """
        Apply three-dimensional similarity transform to structure-set ROIs.

        If the transform corresponds to a translation and/or rotation
        about the z-axis, and either the roi source type is "dicom"
        or force-contours is True, the transform is applied to contour
        points and the roi mask is set as unloaded.  Otherwise the transform
        is applied to the mask and contours are set as unloaded.

        The transform is applied in the order: translation, scaling,
        rotation.  The latter two are about the centre coordinates.

        **Parameters:**
        
        force_contours : bool, default=False
            If True, and the transform corresponds to a translation
            and/or rotation about the z-axis, apply transform to contour
            points independently of the original data source.

        names : list/None, default=False
            List of ROIs to which transform is to be applied.  If None,
            transform is applied to all ROIs.

        For other parameters, see documentation for
        skrt.image.Image.transform().  Note that the ``order``
        parameter isn't available for structure-set transforms - a value of 0
        is used always.
        """

        for roi in self.get_rois(names):
            roi.transform(scale, translation, rotation, centre, resample,
                    restore, fill_value, force_contours)

        return None

    def add_plan(self, plan):
        """Add a Plan object to be associated with this structure set and
        its ROIs. This does not affect the structure set associated with
        the Plan object.

        **Parameters:**

        plan : skrt.dose.Plan
            A Plan object to assign to this structure set.
        """

        if not plan in self.plans:
            self.plans.append(plan)
            self.plans.sort()

            for roi in self.rois:
                roi.add_plan(plan)

    def clear_plans(self):
        """Clear all plan maps associated with this structure set."""

        self.plans = []

    def get_plans(self):
        """Return list of Plan objects associated with this structure set."""

        return self.plans

    def interpolate_points(self, n_point=None, dxy=None,
        smoothness_per_point=0):
        '''
        Return new StructureSet object, with interpolated contour points.

        Points are interpolated for each contour of each ROI.

        **Parameters:**

        n_point : int, default=None
            Number of points per contour, after interpolation.  This must
            be set to None for dxy to be considered.

        dxy : float, default=None
            Approximate distance required between contour points.  This is taken
            into account only if n_point is set to None.  For a contour of
            length contour_length, the number of contour points is then taken
            to be max(int(contour_length / dxy), 3).  

        smoothness_per_point : float, default=0
            Parameter determining the smoothness of the B-spline curve used
            in contour approximation for interpolation.  The product of
            smoothness_per_point and the number of contour points (specified
            directly via n_point, or indirectly via dxy) corresponds to the
            parameter s of scipy.interpolate.splprep - see documentation at:
        
            https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep

            A smoothness_per_point of 0 forces the B-spline to pass through
            all of the pre-interpolation contour points.
        '''

        # Interpolate points for the rois
        rois = []
        for roi in self.get_rois():
            rois.append(roi.interpolate_points(
                n_point, dxy, smoothness_per_point))

        # Clone self, then reset source and rois
        ss = StructureSet(self)
        ss.sources = rois
        ss.rois = rois

        return ss

    def order_rois(self, order='x+'):
        '''
        Order ROIs by one of their centroid coordinates.

        **Parameter:**

        order : str, default='x+'
            Specification of way it which ROIs are to be orderd:
            - 'x+' : in order of increasing x value.
            - 'x-' : in order of decreasing x value.
            - 'y+' : in order of increasing y value.
            - 'y-' : in order of decreasing y value.
            - 'z+' : in order of increasing z value.
            - 'z-' : in order of decreasing z value.
        '''

        if order not in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            return

        # Determine axis for ordering, and direction.
        axis = 'xyz'.find(order[0])
        reverse = ('-' in order)

        # Create dictionary where keys are centroid coordinates
        # and values are lists of ROI indicies at these coordinates.
        centroids = {}
        for i in range(len(self.rois)):
            xyz = self.rois[i].get_centroid()[axis]
            if xyz not in centroids:
                centroids[xyz] = []
            centroids[xyz].append(i)

        # Create list of ROIs ordered by centroid coordinate.
        rois = []
        for centroid in sorted(centroids, reverse=reverse):
            for i in centroids[centroid]:
                rois.append(self.rois[i])

        # Store the new order of ROIs.
        self.rois = rois

    def combine_rois(self, name=None, roi_names=None, image=None, method=None):
        '''
        Combine two or more ROIs as a single ROI.

        **Parameters:**

        name : str, default=None
            Name to be given to the composite ROI.  If None, the
            name assigned is formed by joining together the names
            of the original ROIs.

        roi_names : list, default=None
            List of names of ROIs to be combined.  If None, all of the
            structure set's ROIs are combined.

        image : skrt.image.Image, default=None
            Image to set for the composite ROI.  If None, use the image
            of the ROI with the first name in <roi_names>.

        method : str, default=None
            Method to use for combining ROIs.  Can be: 

                - "contour": take unary union of shapely polygons.
                - "mask": sum binary masks.
                - "auto": use the default_geom_method of the ROI with the
                          first name in <roi_names>.

            If None, "auto" is used.
        '''

        # If None values passed, set default behaviour.
        if roi_names is None:
            roi_names = self.get_roi_names()
        if name is None:
            name = '+'.join(roi_names)
        if method in [None, "auto"]:
            method = self[roi_names[0]].default_geom_method
        if image is None:
            image = self[roi_names[0]].image

        if "contour" == method:
            # Create a dictionary containing polygons for all ROIs.
            all_polygons = {}
            for roi in self.get_rois(roi_names):
                for key, polygons in roi.get_polygons().items():
                    if not key in all_polygons:
                        all_polygons[key] = []
                    all_polygons[key].extend(polygons)

            # Evaluate the unary union of polygons for each slice.
            all_polygons = {key: [ops.unary_union(all_polygons[key])]
                     for key in all_polygons}

            # Create the composite ROI.
            roi_new = ROI(source=all_polygons, image=image, name=name)

        else:
            # Use data from one of the ROIs as a starting point.
            roi0 = self.get_roi(roi_names[0])
            roi_new = ROI(source=roi0.get_mask().copy(), affine=roi0.affine,
                    name=name, image=image)

            # Combine with data from the other ROIs.
            for i in range(1, len(roi_names)):
                roi_new.mask.data |= self.get_roi(roi_names[i]).get_mask()

        return roi_new

    def get_mask_image(self, name=None):
        """
        Get image object representing mask for all ROIs of structure set.

        **Parameters:**

        name : str, default=None
            Name to be given to the mask image.  If None, the
            name assigned is the name of the structure set, with suffix
            "_mask" appended.
        """
        name = name or f"{self.name}_mask"
        roi_combined = self.combine_rois(name=name)
        mask_image = roi_combined.get_mask_image()
        return mask_image

    def crop(self, xlim=None, ylim=None, zlim=None):
        """
        Crop all ROIs of StructureSet to specified range in x, y, z.

        The parameters xlim, ylim, zlim are passed to each ROI's crop() method.
        For explanation of these parameters, see documentation for
        skrt.structures.ROI.crop().
        """
        for roi in self.get_rois():
            roi.crop(xlim, ylim, zlim)

class StructureSetIterator:

    def __init__(self, structure_set):
        self.idx = -1
        self.structure_set = structure_set

    def __next__(self):
        self.idx += 1
        if self.idx < len(self.structure_set.get_rois()):
            return self.structure_set.get_rois()[self.idx]
        raise StopIteration


def load_rois_dicom(path, names=None):
    """Load ROI(s) from a dicom structure set file. <name> can be a single
    name or list of names of ROIs to load."""

    # Load dicom object
    try:
        ds = pydicom.dcmread(path, force=True)
    except pydicom.errors.InvalidDicomError:
        return [], None
    if not hasattr(ds, "SOPClassUID"):
        return [], None
    if not (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"):
        print(f"Warning: {path} is not a DICOM structure set file!")
        return [], None

    # Get ROI names
    seq = get_dicom_sequence(ds, "StructureSetROI")
    rois = {}
    for roi in seq:
        rois[int(roi.ROINumber)] = {"name": roi.ROIName}

    # Find ROIs matching requested names
    names_to_load = None
    if isinstance(names, str):
        names_to_load = [names]
    elif skrt.core.is_list(names):
        names_to_load = names
    if names_to_load:
        rois = {
            i: s
            for i, s in rois.items()
            if any(
                [fnmatch.fnmatch(s["name"].lower(), n.lower()) for n in names_to_load]
            )
        }
        if not len(rois):
            print(f"Warning: no ROIs found matching name(s): {names}")
            return [], None

    # Get ROI details
    roi_seq = get_dicom_sequence(ds, "ROIContour")
    for roi in roi_seq:

        number = roi.ReferencedROINumber
        if number not in rois:
            continue
        data = {"contours": {}}

        # Get ROI colour
        if "ROIDisplayColor" in roi:
            data["color"] = [int(c) / 255 for c in list(roi.ROIDisplayColor)]
        else:
            data["color"] = None

        # Get contours
        contour_seq = get_dicom_sequence(roi, "Contour")
        if contour_seq:
            for c in contour_seq:
                plane_data = [
                    [float(p) for p in c.ContourData[i * 3 : i * 3 + 3]]
                    for i in range(c.NumberOfContourPoints)
                ]
                z = float(c.ContourData[2])
                if z not in data["contours"]:
                    data["contours"][z] = []
                data["contours"][z].append(np.array(plane_data))

        rois[number].update(data)

    return rois, ds


def get_dicom_sequence(ds=None, basename=""):

    sequence = []
    for suffix in ["Sequence", "s"]:
        attribute = f"{basename}{suffix}"
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break
    return sequence

def contour_to_polygon(contour):
    """Convert a list of contour points to a Shapely polygon, ensuring that 
    the polygon is valid."""

    # Disable shapely INFO messages issued when a polygon isn't valid:
    # "callback - INFO - Ring Self-intersection at or near point"
    logging.getLogger('shapely.geos').setLevel(
            skrt.core.Defaults().shapely_log_level)

    polygon = geometry.Polygon(contour)

    delta = 0.005
    if not polygon.is_valid:
        tmp = geometry.Polygon(polygon)
        buffer = 0.
        # The idea here is to increase the buffer distance until
        # a valid polygon is obtained.  This generally seems to work...
        while (isinstance(polygon, geometry.MultiPolygon)
                or not polygon.is_valid):
            buffer += delta
            polygon = tmp.buffer(buffer)
        points = []
        for x, y in polygon.exterior.coords:
            points.append((x, y))
        polygon = geometry.Polygon(points)

    return polygon

def polygon_to_contour(polygon):
    '''
    Convert a Shapely polygon to a list of contour points.

    **Parameter:**

    polygon: shapely.geometry.polygon
        Shapely polygon.

    z_polygon: z coordinate at which polygon is defined.
    '''
    contour_points = []
    for x, y in list(polygon.exterior.coords):
        contour_points.append([x, y])
    contour = np.array(contour_points)

    return contour

def interpolate_points_single_contour(source=None, n_point=None, dxy=None,
        smoothness_per_point=0):
    '''
    Interpolate points for a single contour.

    **Parameters:**

    source : shapely.polygon.Polygon/np.ndarray/list, default=None
        Contour representation as either a shapely Polygon, a numpy
        array of [x, y] pairs or a list of [x, y] pairs.  The
        returned contour has the same representation.

    n_point : int, default=None
        Number of points to define contour, after interpolation.  This must
        be set to None for dxy to be considered.

    dxy : float, default=None
        Approximate distance required between contour points.  This is taken
        into account only if n_point is set to None.  For a contour of
        length contour_length, the number of contour points is then taken
        to be max(int(contour_length / dxy), 3).  

    smoothness_per_point : float, default=0
        Parameter determining the smoothness of the B-spline curve used
        in contour approximation for interpolation.  The product of
        smoothness_per_point and the number of contour points (specified
        directly via n_point, or indirectly via dxy) corresponds to the
        parameter s of scipy.interpolate.splprep - see documentation at:
        
        https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep

        A smoothness_per_point of 0 forces the B-spline to pass through
        all of the pre-interpolation contour points.
    '''

    # Extract list of contour points from source.
    contour_length = None
    if isinstance(source, geometry.polygon.Polygon):
        points = list(polygon_to_contour(source))
        contour_length = source.length
    elif isinstance(source, np.ndarray) or isinstance(source, list):
        points = list(source)
    else:
        points = None

    if points is None:
        print('Unrecognised source passed to '
                '\'interpolate_points_single_contour()\'')
        print('Source must be shapely Polygon or contour')
        return None

    # If number of contour points not passed directly,
    # try to determine its value from the distance between contour points.
    if n_point is None and dxy:
        if contour_length is None:
            contour_length = contour_to_polygon(source).length
        n_point = max(int(contour_length / dxy), 3)
    if n_point is None:
        print('No interpolation performed')
        print('Number of contour points (n_point) or distance between points'
                'must be specified.')
        return None

    # Make last point the same as the first, to ensure closed curve.
    points.append(points[0])

    # Discard points that aren't separated from the preceeding point
    # by at least some minimum distance.
    dxy_min = 0.001
    x_last, y_last = points[0]
    x_values, y_values = ([x_last], [y_last])
    for i in range(1, len(points)):
        x, y = points[i]
        if (abs(x - x_last) > dxy_min) or (abs(y - y_last) > dxy_min):
            x_values.append(points[i][0])
            y_values.append(points[i][1])
        x_last, y_last = points[i]

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Interpolate contour points.
    smoothness = smoothness_per_point * len(x_values)
    try:
#        with warnings.catch_warnings():
#            warnings.filterwarnings("ignore", message="Setting x")
            tck, u = interpolate.splprep(
                    [x_values, y_values], s=smoothness, per=True)
    except TypeError as problem:
        print("WARNING: Problem in interpolate_contour_points():", problem)
        return None

    xi_values, yi_values = interpolate.splev(
        np.linspace(0., 1., n_point + 1), tck)

    # Store points in a form that may be converted to a numpy array.
    points_interpolated = list(zip(xi_values, yi_values))
    points_interpolated = [xy for xy in points_interpolated]

    # Return interpolated contour in the same representation as the source.
    if isinstance(source, geometry.polygon.Polygon):
        out_object = geometry.polygon.Polygon(points_interpolated)
    elif isinstance(source, np.ndarray):
        out_object = np.array(points_interpolated)
    else:
        out_object = points_interpolated

    return out_object

def write_structure_set_dicom(
    outname, 
    rois, 
    image=None, 
    affine=None, 
    shape=None,
    orientation=None, 
    header_source=None, 
    patient_id=None,
    modality=None, 
    root_uid=None
):

    # Check we have the relevant info
    if not image and (not affine or not shape):
        raise RuntimeError("Must provide either an image or an affine matrix "
                           "and shape!")

    # Try getting dicom dataset from image
    ds = None
    if image:
        if hasattr(image, "dicom_dataset"):
            ds = image.dicom_dataset
        else:
            ds = skrt.image.create_dicom()
            ds.set_geometry(
                image.get_affine(), image.get_data().shape,
                image.get_orientation_vector(image.get_affine, "dicom")
            )

    # Otherwise, create fresh dicom dataset
    else:
        ds = skrt.image.create_dicom(patient_id, modality, root_uid)
        ds.set_geometry(affine, shape, orientation)

    # Adjust dataset to be for StructureSet instead of image


def dicom_dataset_to_structure_set(ds):
    '''Convert an existing image dicom dataset to a StructureSet dataset.'''

    # Adjust class UIDs
    ds.file_meta.ImplementationClassUID = "9.9.9.100.0.0.1.0.9.6.0.0.1"
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"

    # Assign empty sequences
    ds.ReferencedFrameofReferenceSequence = Sequence()
    ds.StructureSetROISequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()
    ds.ROIContourSequence = Sequence()

    # Assign structure set properties
    ds.StructureSetLabel = ""
    ds.StructureSetDate = ds.InstanceCreationDate
    ds.StructureSetTime = ds.InstanceCreationTime

    # Assign frame of reference


def create_dummy_image(
    extents, 
    slice_thickness,
    shape=None, 
    voxel_size=None,
    fill_val=1e4,
    buffer=2
):
    """Make a dummy image that covers the area spanned by <extents> plus a 
    buffer.

    **Parameters:**
    
    extents : list
        List of [min, max] extents in mm along each of the three axes in 
        order [x, y, z].

    slice_thickness : float
        Slice thickness in mm.

    voxel_size : list, default=None
        Voxel size in mm in the dummy image in the x-y plane, given as 
        [vx, vy]. If <shape> and <voxel_size> are both None, voxel sizes of
        [1, 1] will be used by default. The voxel size in the z direction 
        will be taken from the minimum distance between slice positions in 
        the x-y contours dictionary.

    shape : list, default=None
        Number of voxels in the dummy image in the x-y plane, given as
        [nx, ny]. Only used if <voxel_size> is None. 
        The number of voxels in the z direction will be taken from the 
        number of slices in the x-y contours dictionary.

    fill_val : int/float, default=1e4
        Value with which the voxels in the dummy image should be filled. 

    buffer : float, default=2
        Buffer amount of whitespace in mm to add outside the ROI in each 
        direction.
    """

    # Assign default voxel size
    if shape is None and voxel_size is None:
        voxel_size = [1, 1]
    if not skrt.core.is_list(voxel_size):
        voxel_size = [voxel_size] * 2

    # Calculate voxel sizes from shape
    if shape is not None:
        shape = list(shape)
        if len(shape) != 2:
            raise TypeError("Shape should be a list of two values "
                            "specifying dummy image shape in the [x, y] "
                            "directions.")

        # Calculate voxel size if ROI extent plus buffer is divided by shape
        voxel_size = [(max(ex) - min(ex) + 2 * buffer) / shape[i]
                      for i, ex in enumerate(extents[:2])]

    # Otherwise, calculate shape from voxel sizes
    else:
        voxel_size = list(voxel_size)
        
        # Calculate number of voxels needed to cover ROI plus buffer each side
        shape = [(np.ceil((max(ex) - min(ex)) + 2 * buffer) / voxel_size[i]) 
                 for i, ex in enumerate(extents[:2])]

    # Get z voxel size and shape
    voxel_size.append(slice_thickness)
    shape_z = (max(extents[2]) - min(extents[2])) / slice_thickness

    # Add nearest integer number of buffer voxels
    n_buff_z = np.ceil(buffer / slice_thickness)
    shape.append(shape_z + 2 * n_buff_z)
    shape = [int(s) for s in shape]

    # Get origin position; ensure that origin points to centre of buffer voxel 
    # outside ROI
    origin = [min(ex) + abs(voxel_size[i]) * 0.5 - buffer
              for i, ex in enumerate(extents[:2])]
    origin.append(min(extents[2]) - abs(slice_thickness) * (n_buff_z - 0.5))

    # Create image
    return skrt.image.Image(
        np.ones((shape[1], shape[0], shape[2])) * fill_val,
        voxel_size=voxel_size,
        origin=origin
    )


def create_staple(rois, **kwargs):
    """Create STAPLE ROI from list of ROIs."""

    # Try import SimpleITK
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError:
        raise ModuleNotFoundError("SimpleITK is required to calculate STAPLE contour. "
                                  "Try installing via: pip install simpleitk")

    # Get STAPLE mask
    roi_arrays = []
    for roi in rois:
        roi_arrays.append(sitk.GetImageFromArray(roi.get_mask(
            standardise=True).astype(int)))
    probs = sitk.GetArrayFromImage(sitk.STAPLE(roi_arrays, 1))
    mask = probs > 0.95

    # Create STAPLE ROI object
    return ROI(mask, **kwargs)


def create_majority_vote(rois, **kwargs):
    """Create majority vote ROI from list of ROIs."""

    mask = rois[0].get_mask(standardise=True).astype(int)
    for roi in rois[1:]:
        mask += roi.get_mask(standardise=True).astype(int)
    mask = mask >= len(rois) / 2
    return ROI(mask, **kwargs)


def create_roi_sum(rois, **kwargs):
    """Create ROI from sum of list of ROIs."""

    mask = rois[0].get_mask(standardise=True).copy()
    for roi in rois[1:]:
        mask += roi.get_mask(standardise=True)
    return ROI(mask, **kwargs)


def create_roi_overlap(rois, **kwargs):
    """Create ROI from overlap of list of ROIs."""

    mask = rois[0].get_mask(standardise=True).copy()
    for roi in rois[1:]:
        mask *= roi.get_mask(standardise=True)
    return ROI(mask, **kwargs)


def df_to_html(df):
    """Convert a pandas DataFrame to html."""

    # Convert dataframe to HTML
    html = df.fillna('—').to_html(index=False)
    html = html.replace("^3", "<sup>3</sup>").replace("^2", "<sup>2</sup>")

    # Add header with style details
    header = """
        <head>
            <style>
                th, td {
                    padding: 2px 2px;
                }
                th {
                    background-color: transparent;
                    text-align: center;
                }
            </style>
        </head>
    """
    table_html = (
        (header + html)
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )
    return table_html


def best_text_color(red, green, blue):
    if (red * 0.299 + green * 0.587 + blue * 0.114) > 186:
        return "black"
    return "white"


def get_colored_roi_string(roi, grey=False, color=None):
    """Get ROI name in HTML with background color from roi.color. If grey=True,
    the background color will be grey."""

    if color is None:
        color = roi.color
    if grey:
        red, green, blue = 255, 255, 255
        red, green, blue = 128, 128, 128
        text_col = 200, 200, 200
        # Avoid setting colour for greyed out ROIs.
        # This gives good readability for both dark and light themes.
        return ('<p>&nbsp;{}&nbsp;</p>').format(roi.name)
    else:
        red, green, blue = [c * 255 for c in color[:3]]
        text_col = best_text_color(red, green, blue)
    return (
        '<p style="background-color: rgb({}, {}, {}); '
        'color: {};">&nbsp;{}&nbsp;</p>'
    ).format(red, green, blue, text_col, roi.name)

def compare_roi_pairs(
    pairs, 
    html=False,
    name_as_index=True,
    greyed_out=None,
    colored=False,
    roi_kwargs={},
    **kwargs
):
    if html:
        name_as_index = False
    if greyed_out is None:
        greyed_out = []
    dfs = []
    for roi1, roi2 in pairs:

        # Get DataFrame for this pair
        df_row = roi1.get_comparison(roi2, name_as_index=name_as_index,
                                     **kwargs)

        # Replace values with "--" if either ROI is greyed out
        grey = roi1 in greyed_out or roi2 in greyed_out
        if grey:
            for col in df_row.columns:
                if col == "ROI": 
                    continue
                df_row[col] = "--"

        # Adjust comparison name
        comp_name = roi1.get_comparison_name(roi2, colored=colored, grey=grey,
                roi_kwargs=roi_kwargs)
        if name_as_index:
            df_row.rename({df_row.index[0]: comp_name}, inplace=True)
        else:
            df_row.iloc[0, 0] = comp_name

        dfs.append(df_row)

    ignore_index = False if name_as_index else True
    df = pd.concat(dfs, ignore_index=ignore_index) if dfs else None
    if html:
        return df_to_html(df)
    return df

def get_conformity_index(
    rois,
    ci_type="gen",
    single_slice=False,
    view="x-y", 
    sl=None, 
    idx=None, 
    pos=None, 
    method=None,
    flatten=False,
    ):
    """
    Get conformity index for two or more ROIs.

    The conformity index may be calculated globally or for a single slice.

    This function returns either the value of one of the three types
    of conformity index ("gen" (for generalised), "pairs", "common")
    referred to in:
    https://doi.org/10.1088/0031-9155/54/9/018
    or a skrt.core.Data object with the values of all three.
    For the case of two ROIs, the three metrics are all equivalent
    to the Jaccard conformity index.

    **Parameters:**
        
    rois : list
        List of two or more skrt.structures.ROI objects for which conformity
        index is to be calculated.

    ci_type: str, default="gen"
        Type of conformity index to be returned, from "gen" (for generalised),
        "pairs", "common".  If set to "all", a skrt.core.Data object with
        values for the three types of conformity index is returned.

    single_slice : bool, default=False
        If False, the global 3D Dice score of the full ROIs will be returned;
        otherwise, the 2D Dice score on a single slice will be returned.

    view : str, default="x-y"
        Orientation of slice on which to get Dice score. Only used if 
        single_slice=True. If using, <ax> must be an axis that lies along
        the slice in this orientation.

    sl : int, default=None
        Slice number. If none of <sl>, <idx> or <pos> are supplied but
        <single_slice> is True, the central slice of this ROI will be used.

    idx : int, default=None
        Array index of slice. If none of <sl>, <idx> or <pos> are supplied but
        <single_slice> is True, the central slice of this ROI will be used.

    pos : float, default=None
        Slice position in mm. If none of <sl>, <idx> or <pos> are supplied but
        <single_slice> is True, the central slice of this ROI will be used.

    method : str, default=None
        Method to use for Dice score calculation. Can be: 
        - "contour": get intersections and areas of shapely contours.
        - "mask": count intersecting voxels in binary masks.
        - None: use the method set in self.default_geom_method.
        For more than two ROIs and <ci_type> "common", only the "mask"
        method is implemented.

    flatten : bool, default=False
        If True, all slices will be flattened in the given orientation and
        the Dice score of the flattened slices will be returned. Only 
        available if method="mask".
    """

    # Require valid inputs.
    if len(rois) < 2 or ci_type not in ["common", "gen", "pairs", "all"]:
        return

    # Initialise Data object for conformity indices.
    ci = skrt.core.Data({"common": 0, "gen":0, "pairs":0})

    # Deal with cases "gen" and "pairs".
    #
    # For "gen", the conformity index is calculated as the ratio of
    # (1) the sum of the intersections for all pairs to (2) the sum of
    # the unions for all pairs.
    #
    # For "pairs, the conformity index is calcualted as the mean of the
    # Jaccard conformity indices for all pairs.

    numerator = 0
    denominator = 0
    n_pair = 0
    for idx1 in range(len(rois) - 1):
        for idx2 in range(idx1 + 1, len(rois)):
            roi1 = rois[idx1]
            roi2 = rois[idx2]
            intersection, union, mean_size = (
                    roi1.get_intersection_union_size(roi2, single_slice,
                        view, sl, idx, pos, method, flatten))
            numerator += intersection
            denominator += union
            ci.pairs += intersection / union
            n_pair += 1

    ci.gen = numerator / denominator
    ci.pairs /= n_pair

    # Deal with case "common".
    #
    # The conformity index is calculated as the ratio of
    # (1) the common intersection of all ROIs to (2) the combined union
    # of all ROIs.
    if 2 == len(rois):
        intersection, union, mean_size = (
                rois[0].get_intersection_union_size(rois[1], single_slice,
                        view, sl, idx, pos, method, flatten))
        ci.common = intersection / union
    else:
        if not single_slice:
            mask_intersection = rois[0].clone().get_mask(
                    view, flatten, standardise=True)
            mask_union = rois[0].clone().get_mask(
                    view, flatten, standardise=True)
        else:
            mask_intersection = rois[0].clone().get_slice(
                    view, sl, idx, pos)
            mask_union = rois[0].clone().get_slice(view, sl, idx, pos)

        for roi in rois[1:]:
            if not single_slice:
                mask = roi.get_mask(view, flatten, standardise=True)
            else:
                mask = roi.get_slice(view, sl, idx, pos)

            mask_intersection = (mask_intersection & mask)
            mask_union = (mask_union | mask)

        ci.common = mask_intersection.sum() / mask_union.sum()

    return getattr(ci, ci_type, ci)

def get_roi_slice(roi, z_fraction=1, suffix=None):
    """
    Get ROI object corresponding to x-y slice through input ROI.

    **Parameters:**

    roi : skrt.structures.ROI
        ROI object for which slice is to be obtained.

    z_fraction : float, default=1
        Position along z axis at which to take slice through ROI.
        The position is specified as the fractional distance
        from the ROI's most-inferior point: 0 corresponds to
        the most-inferior point (lowest z); 1 corresponds to the
        most-superior point (highest z).  Values for z_fraction
        outside the interval [0, 1] result in a RuntimeError.

    suffix : str, default=None
        Suffix to append to name of input ROI, to form name
        of output ROI.  If None, append z_fraction, with value
        given to two decimal places.
    """
    # Check that z_fraction is in allowed interval.
    if z_fraction < 0 or z_fraction > 1:
        raise RuntimeError(f"z_fraction={z_fraction}"
                            " outside allowed interval [0, 1]")

    # Determine the index of the image slice at which to take ROI slice.
    pos1, pos2 = roi.get_extent()
    pos = pos1 + z_fraction * (pos2 - pos1)
    idx = roi.pos_to_idx(pos, "z")

    # Obtain contours for required slice, and convert to ROI object.
    contours = roi.get_contours(idx_as_key=True)[idx]
    roi_slice = ROI({roi.idx_to_pos(idx, "z"): contours})

    suffix = f"{z_fraction:.2f}" if suffix is None else suffix
    if suffix:
        roi_slice.name = f"{roi.name}_{suffix}"
    else:
        roi_slice.name = str(roi.name)

    return roi_slice

def get_metric_method(metric):
    """
    Map between metric identifier as listed in get_comparison_metrics()
    and name of ROI method for calculating the metric.
    
    This function is used in ROI.get_metric_by_slice() to determine
    the method to be called for each metric.
    """
    # Keys are metric identifiers; values are names of ROI methods
    # for metric calculation.
    # Metrics are included in the dictionary only if identifier and method
    # name are different.
    mappings = {
            "abs_centroid": "abs_centroid_distance",
            "centroid": "centroid_distance",
            "rel_area_diff": "relative_area_diff",
            }
    return mappings.get(metric, metric)

def get_comparison_metrics(centroid_components=False, slice_stats=None,
        default_by_slice=None, view="x-y"):
    """
    Get list of comparison metrics.

    All metrics listed here should be recognised by ROI.get_comparison(),
    and all metrics recognised by ROI.get_comparison() should be listed here.

    **Parameters:**

    centroid_components : bool, default=False
        If True, replace metrics "centroid" and "centroid_slice" by the
        metrics for their components.

    slice_stats : str/list/dict, default=None
        Specification of statistics to be calculated relative
        to slice-by-slice metric values.

        A statistic is specified by the name of the function for
        its calculation in the Python statistics module:

        https://docs.python.org/3/library/statistics.html

        For its use here, the relevant function must require
        a single-variable input, must not require any keyword
        arguments, and must return a single number.

        Available options include: "mean", "median", "mode",
        "stdev".

       Statistics to be calculated can be specified using any of
       the following:

       - String specifying single statistic to be calculated,
         with slices considered as given by <default_by_slice>,
         for example: "mean";
       - List specifying multiple statistics to be calculated,
         with slices considered as given by <default_by_slice>,
         for example: ["mean", "stdev"];
       - Dictionary where keys specify slices to be considered,
         and values specify statistics (string or list), for
         example: {"union": ["mean", "stdev"]}.  Valid slice
         specifications are as listed for <default_by_slice>.

    default_by_slice: str, default=None
        Default specification of slices to be considered when
        calculating slice-by-slice statistics for a metric comparing
        two ROIs, roi1 and roi2.  The valid specifications are:

        - "left": consider only slices containing roi1;
        - "right": consider only slices containing roi2;
        - "union": consider slices containing either of roi1 and roi2;
        - "intersection": consider slices containing both roi1 and roi2.

        If None, use skrt.core.Defaults().by_slice

        view : str, default="x-y"
            Orientation to consider when taking centroid components relative
            to a slice.  Can be "x-y", "y-z", or "x-z".
    """
    metrics = [
            "abs_centroid",
            "abs_centroid_flat",
            "abs_centroid_slice",
            "abs_centroid_slice_stats",
            "area_diff",
            "area_diff_flat",
            "area_diff_slice_stats",
            "area_ratio",
            "area_ratio_flat",
            "area_ratio_slice_stats",
            "centroid",
            "centroid_slice",
            "centroid_slice_stats",
            "dice",
            "dice_flat",
            "dice_slice",
            "dice_slice_stats",
            "hausdorff_distance",
            "hausdorff_distance_flat",
            "jaccard",
            "jaccard_flat",
            "jaccard_slice",
            "jaccard_slice_stats",
            "mean_distance_to_conformity",
            "mean_distance_to_conformity_flat",
            "mean_signed_surface_distance",
            "mean_signed_surface_distance_flat",
            "mean_surface_distance",
            "mean_surface_distance_flat",
            "mean_over_contouring",
            "mean_over_contouring_flat",
            "mean_under_contouring",
            "mean_under_contouring_flat",
            "rel_area_diff",
            "rel_area_diff_flat",
            "rel_area_diff_slice_stats",
            "rel_volume_diff",
            "rms_signed_surface_distance",
            "rms_signed_surface_distance_flat",
            "rms_surface_distance",
            "rms_surface_distance_flat",
            "volume_diff",
            "volume_ratio",
            ]

    # Replace centroid vectors by components.
    if centroid_components:
        centroids = {
                "centroid": ["centroid_x", "centroid_y", "centroid_z"],
                "centroid_slice": []
                }
        if slice_stats is not None:
            centroids["centroid_slice_stats"] = []

        for i_ax in skrt.image._plot_axes[view]:
            ax = skrt.image._axes[i_ax]
            centroids["centroid_slice"].append(f"centroid_slice_{ax}")

            if slice_stats is not None:
                centroids["centroid_slice_stats"].append(
                        f"centroid_{ax}_slice_stats")

        for vector, components in centroids.items():
            idx = metrics.index(vector)
            metrics[idx : idx + 1] = components

    # Remove metrics with suffix "slice_stats",
    # and replace with specific metrics relating to slice-by-slice statistics.
    if slice_stats is not None:
        slice_stats_metrics = []
        metrics2 = []
        for metric in metrics:
            if "slice_stats" in metric:
                slice_stats_metrics.append(metric.split("_slice_stats")[0])
            else:
                metrics2.append(metric)

        for by_slice, stats in (
                expand_slice_stats(slice_stats, default_by_slice).items()):
            for metric in slice_stats_metrics:
                for stat in stats:
                    metrics2.append(f"{metric}_slice_{by_slice}_{stat}")
        metrics = metrics2

    return sorted(metrics)

def get_consensus_types():
    """
    Get list of consensus types.

    All consensus types listed here should be recognised by
    StructureSet.get_consensus(), and all consensus types recognised
    by StructureSet.get_consensus should be listed here.
    """
    return [
            "majority",
            "overlap",
            "staple",
            "sum",
            ]

def get_by_slice_methods():
    """
    Get list of methods for defining slices to be considered
    for slice-by-slice ROI comparisons.
    """
    return ["left", "right", "union", "intersection"]

def get_all_rois(objs=None):
    """
    Create list of ROIs from arbitrary combination of ROIs and StructureSets.

    **Parameters:**

    objs : ROI/StructureSet/list, default=None
        Object(s) from which a list of ROIs is to be created.  The
        object(s) can be a single skrt.structures.ROI object, a single
        skrt.structures.StructureSet object, or a list containing any
        combination of ROI and StructureSet objects.
    """
    # Ensure that objs is a list
    if issubclass(type(objs), (ROI, StructureSet)):
        objs = [objs]
    elif objs is None:
        objs = []

    # Create a list containing all unique rois.
    all_rois = []
    for item in objs:
        if issubclass(type(item), ROI):
            candidate_rois = [item]
        elif issubclass(type(item), StructureSet):
            candidate_rois = item.get_rois()
        for roi in candidate_rois:
            if not roi in all_rois:
                all_rois.append(roi)

    return all_rois

def get_translation_to_align(roi1, roi2, z_fraction1=None, z_fraction2=None):
    """
    Determine translation for aligning <roi1> to <roi2>.

    **Parameters:**

    roi1 : ROI/StructureSet/list
        ROI that is to be translation to achieve the alignment.  This
        can be specified directly as a single ROI.  Alternatively,
        it can be a StructureSet, or a list or ROI/StructureSet objects,
        in which case the individual ROIs will be combined.

    roi2 : ROI/StructureSet/list
        ROI with which alignment is to be performed.  This can be
        specified directly as a single ROI.  Alternatively, it can
        be a StructureSet, or a list or ROI/StructureSet objects,
        in which case the individual ROIs will be combined.

    z_fraction1 : float, default=None
        Position along z axis of slice through <roi1> on which
        to align.  If None, alignment is to the centroid of the
        whole ROI volume.  Otherwise, alignment is to the
        centroid of the slice at the specified distance
        from the ROI's most-inferior point: 0 corresponds to
        the most-inferior point (lowest z); 1 corresponds to the
        most-superior point (highest z).  Values for z_fraction
        outside the interval [0, 1] result in a RuntimeError.

    z_fraction2 : float, default=None
        Position along z axis of slice through <roi2> on which
        to align.  If None, alignment is to the centroid of the
        whole ROI volume.  Otherwise, alignment is to the
        centroid of the slice at the specified distance
        from the ROI's most-inferior point: 0 corresponds to
        the most-inferior point (lowest z); 1 corresponds to the
        most-superior point (highest z).  Values for z_fraction
        outside the interval [0, 1] result in a RuntimeError.
        """
    # Create list of two single ROIs.
    rois = []
    for roi in [roi1, roi2]:
        if isinstance(roi, ROI):
            # Store single ROI directly.
            rois.append(roi)
        elif isinstance(roi, StructureSet):
            # Combine ROIs of StructureSet.
            rois.append(roi.combine_rois())
        else:
            # Combine ROIs from list.
            rois.append(StructureSet(get_all_rois(roi)).combine_rois())

    # Calculate alignment points.
    centroids = []
    for roi, z_fraction in zip(rois, [z_fraction1, z_fraction2]):
        # Calculate centroid for the whole ROI volume.
        if z_fraction is None:
            centroids.append(roi.get_centroid())
        # Calculate centroid for slice at specified position.
        else:
            centroids.append(roi.get_roi_slice(z_fraction).get_centroid())

    return tuple(centroids[1] - centroids[0])

def get_slice_positions(roi1, roi2=None, view="x-y", position_as_idx=False,
        method=None):
    """
    Get ordered list of slice positions for either or both of a pair of ROIs.

    **Parameters:**

    roi1 : skrt.structures.ROI
        ROI, or one of a pair of ROIs, for which slice information is to
        be obtained.

    roi2 : skrt.structures.ROI, default=None
        If not None, second in a pair of ROIs for which slice information
        is to be obtained.

    view : str, default="x-y"
        View in which to obtain slices.

    position_as_idx : bool, default=False
        If True, return positions as slice indices; if False,
        return positions as slice z-coordinates (mm).

    method: str, default=None
        String specifying slices for which positions are to be obtained,
        for ROIs roi1, roi2:

        - "left" (or roi2 is None): return positions of slices containing roi1;
        - "right": return positions of slices containing roi2;
        - "union": return positions of slices containing either roi1 or roi2;
        - "intersection": return positions of slices containing both
          roi1 and roi2.

        If None, value of skrt.core.Defaults().by_slice is used.
    """
    method = method or skrt.core.Defaults().by_slice
    if method not in get_by_slice_methods():
        raise RuntimeError(f"Method must be one of {get_by_slice_methods()}"
        " - not '{method}'")

    indices = None

    indices1 = roi1.get_indices(view=view, method="mask")
    if not issubclass(type(roi2), ROI) or "left" == method:
        indices = indices1

    if indices is None:
        indices2 = roi2.get_indices(view=view, method="mask")
        if "right" == method:
            indices = indices2
        
    if indices is None:
        if "union" == method:
            indices = list(set(indices1).union(set(indices2)))

    if indices is None:
        if "intersection" == method:
            indices = list(set(indices1).intersection(set(indices2)))

    if indices is not None:
        if position_as_idx:
            return sorted(indices)
        else:
            ax = skrt.image._slice_axes[view]
            return sorted([roi1.idx_to_pos(idx, ax) for idx in indices])

def expand_slice_stats(slice_stats=None, default_by_slice=None):
    """
    Expand specification of statistics to calculate for ROI comparison
    metrics evaluated slice by slice.

    A dictionary is returned, where keys give the slices to be considered
    for calculating statistics, and values are lists of metrics.

    **Parameters:**

    slice_stats : str/list/dict, default=None
        Specification of statistics to be calculated relative
        to slice-by-slice metric values.

        A statistic is specified by the name of the function for
        its calculation in the Python statistics module:

        https://docs.python.org/3/library/statistics.html

        For its use here, the relevant function must require
        a single-variable input, must not require any keyword
        arguments, and must return a single number.

        Available options include: "mean", "median", "mode",
        "stdev".

       Statistics to be calculated can be specified using any of
       the following:

       - String specifying single statistic to be calculated,
         with slices considered as given by <default_by_slice>,
         for example: "mean";
       - List specifying multiple statistics to be calculated,
         with slices considered as given by <default_by_slice>,
         for example: ["mean", "stdev"];
       - Dictionary where keys specify slices to be considered,
         and values specify statistics (string or list), for
         example: {"union": ["mean", "stdev"]}.  Valid slice
         specifications are as listed for <default_by_slice>.

    default_by_slice: str, default=None
        Default specification of slices to be considered when
        calculating slice-by-slice statistics for a metric comparing
        two ROIs, roi1 and roi2.  The valid specifications are:

        - "left": consider only slices containing roi1;
        - "right": consider only slices containing roi2;
        - "union": consider slices containing either of roi1 and roi2;
        - "intersection": consider slices containing both roi1 and roi2.

        If None, use skrt.core.Defaults().by_slice
    """
    if not slice_stats:
        return {}

    # Set default method for slice selection.
    default_by_slice = default_by_slice or skrt.core.Defaults().by_slice

    # Ensure that slice_stats is a dictionary.
    if isinstance(slice_stats, str):
        slice_stats = {default_by_slice: [slice_stats]}

    elif skrt.core.is_list(slice_stats):
        slice_stats = {default_by_slice: slice_stats}

    elif isinstance(slice_stats, dict):
        checked_slice_stats = {}
        for by_slice, stats in slice_stats.items():
            if isinstance(stats, str):
                checked_slice_stats[by_slice] = [stats]
            elif skrt.core.is_list(stats):
                checked_slice_stats[by_slice] = list(stats)
        slice_stats = checked_slice_stats

    if isinstance(slice_stats, dict):
        return slice_stats

    return {}
