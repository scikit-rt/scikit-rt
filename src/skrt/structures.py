"""Classes related to ROIs and structure sets."""

from scipy import ndimage
from skimage import draw
from shapely import affinity
from shapely import geometry
import fnmatch
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

import skrt.core
import skrt.image
from skrt.dicom_writer import DicomWriter

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
    """Single region of interest (ROI)."""

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
                (f) Dictionary of shapely polygons in the x-y orientation,
                    where the keys are z positions in  mm and values are lists
                    of polygons on each slice.

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
            shapely.draw.polygon2mask will be returned with no border checks, 
            which is faster than performing border checks.

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
                    else:
                        self.input_contours[z].append(contour)
        else:
            self.source = source

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
        self.mask_threshold = mask_threshold
        self.overlap_level = overlap_level
        self.contours = {}
        self.default_geom_method = default_geom_method
        self.kwargs = kwargs
        self.source_type = None
        self.dicom_dataset = None
        self.contours_only = False

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
                affine=self.source.get_affine()
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
                        contours[key] = []
                    contours[key].append([x_point, y_point])

                # Store the list of contours (one contour per slice per roi).
                self.input_contours = {}
                for key in sorted(contours):
                    self.input_contours[float(key)] = [np.array(contours[key])]
                rois.append(self.name)

            else:
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

    def reset_contours(self, contours=None):
        """Reset x-y contours to a given dict of slices and contour lists, 
        and ensure that mask and y-z/x-z contours will be recreated. If 
        contours is None, contours will be reset using own x-y contours."""

        # Check format is correct
        if contours is None:
            contours = self.get_contours()
        if not isinstance(contours, dict):
            raise TypeError("contours should be a dict")
        for z, c in contours.items():
            if not isinstance(c, list):
                raise TypeError(f"contours[{z}] should be a list of contours "
                                f"on slice {z}")

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
                origin=self.origin
            )

        elif mask is not None:
            raise TypeError("mask should be an Image or a numpy array")

        self.loaded_mask = True
        self.contours = {}
        self.empty = not np.any(self.mask.get_data())
        self.input_contours = None

    def get_contours(self, view="x-y", idx_as_key=False):
        """Get dict of contours in a given orientation."""

        self.load()
        if view not in self.contours:
            self.create_contours(view)
        contours = self.contours[view]

        # Convert keys to indices rather than positions
        if idx_as_key:
            contours = {self.pos_to_idx(key, skrt.image._slice_axes[view]): 
                        value for key, value in contours.items()}
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

    def get_voxel_size(self):
        """Load self and get voxel_size."""

        self.load()
        return self.voxel_size

    def get_origin(self):
        """Load self and get origin."""

        self.load()
        return self.origin

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
            pixel to be added to the mask. Only used if <check_borders> is 
            True. If None, the value of self.overlap_level will be used; otherwise,
            self.overlap_level will be overwritten with this value. If both are 
            None, the output of shapely.draw.polygon2mask will be returned 
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
                np.zeros((self.shape[1], self.shape[0], self.shape[2])),
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

                    # Check overlap of edge pixels
                    if self.overlap_level is not None:
                        conn = ndimage.morphology.generate_binary_structure(
                            2, 2)
                        edge = mask ^ ndimage.morphology.binary_dilation(
                            mask, conn)
                        if self.overlap_level >= 0.5:
                            edge += mask ^ ndimage.morphology.binary_erosion(
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
                    self.mask.data[:, :, iz] += mask.T

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

        idx = self.get_idx(view, sl, idx, pos)
        if idx is None:
            return False
        return idx in self.get_indices(view)

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
                centroid["voxels"] = np.array([
                    self.pos_to_idx(c, ax=i, return_int=False) 
                    for i, c in enumerate(centroid["mm"])
                ])
                centroid["slice"] = np.array([
                    self.pos_to_slice(c, ax=i, return_int=False) 
                    for i, c in enumerate(centroid["mm"])
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
            centroid["mm"] = np.array([
                self.idx_to_pos(c, ax=i) for i, c in 
                enumerate(centroid["voxels"])
            ])
            centroid["slice"] = np.array([
                self.idx_to_slice(c, ax=i) for i, c in 
                enumerate(centroid["voxels"])
            ])

        # Cache global centroid
        if not single_slice:
            self._centroid = centroid

        # Return centroid in requested units
        return centroid[units]

    def get_slice_thickness_contours(self):
        """Get z voxel size using positions of contours."""

        contours = self.get_contours("x-y")
        z_keys = sorted(contours.keys())
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

        # Get volume in ml
        self._volume["ml"] = self._volume["mm"] / 1000

        # Return volume in the requested units
        return self._volume[units]

    def get_area(
        self, 
        view="x-y", 
        sl=None, 
        idx=None, 
        pos=None, 
        units="mm", 
        method=None,
        flatten=False,
        **kwargs
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

    def get_extents(self, buffer=None, buffer_units="mm", method=None):
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
        """

        # Get extent in each direction
        extents = []
        for ax in skrt.image._axes:
            extents.append(self.get_extent(ax, method=method))

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

        # Calculate extent from contours
        if method == "contour":

            # Calculate full z extent from contour positions
            if ax == "z" and not single_slice:
                z_keys = list(self.get_contours("x-y").keys())
                vz = self.get_slice_thickness_contours()
                z_max = max(z_keys) +  vz / 2
                z_min = min(z_keys) - vz / 2
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
                return [min(points), max(points)]
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
        return [self.idx_to_pos(min_pos, ax), self.idx_to_pos(max_pos, ax)]

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
                - "centroid_x": 3D centre-of-mass of ROI along the x-axis.
                - "centroid_y": 3D centre-of-mass of ROI along the y-axis.
                - "centroid_z": 3D centre-of-mass of ROI along the z-axis.
                - "centroid_slice": 2D centre of mass on a single slice. Will
                  be split into two columns corresponding to the two axes on
                  the slice.

                - "length": lengths of the ROI in each direction; will be split
                  into three columns corresponding to the three axes.
                - "length_x": length of the ROI along the x-axis.
                - "length_y": length of the ROI along the y-axis.
                - "length_z": length of the ROI along the z-axis.
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
            "single_slice": True,
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

    def get_centroid_distance(self, roi, single_slice=False, **kwargs):
        """Get centroid displacement vector with respect to another ROI.

        **Parameters:**
        
        roi : ROI
            Other ROI with which to compare centroid.

        single_slice : bool, default=False
            If True, the centroid will be returned for a single slice.

        kwargs : dict
            Other kwargs to pass to ROI.get_centroid().
        """

        this_centroid = np.array(
            self.get_centroid(single_slice=single_slice, **kwargs)
        )
        other_centroid = np.array(
            roi.get_centroid(single_slice=single_slice, **kwargs)
        )
        if None in this_centroid or None in other_centroid:
            if single_slice:
                return np.array([None, None])
            else:
                return np.array([None, None, None])
        return other_centroid - this_centroid

    def get_abs_centroid_distance(
        self, 
        roi, 
        view="x-y", 
        single_slice=False,
        flatten=False, 
        **kwargs):
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

        kwargs : dict
            Other kwargs to pass to ROI.get_centroid().
        """

        # If flattening, need to get 3D centroid vector
        if flatten:
            single_slice = False

        # Get centroid vector
        centroid = self.get_centroid_distance(
            roi, 
            view=view, 
            single_slice=single_slice,
            **kwargs
        )

        # If centroid wasn't available, return None
        if None in centroid:
            return None

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
    ):
        """Get Dice score with respect to another ROI, either globally or on a 
        single slice.

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
        """

        self.load()

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

            # Get positions of slice(s) on which to compare areas and total areas
            # on those slice(s)
            if not single_slice:
                positions = list(self.get_contours("x-y").keys())
                areas1 = [self.get_area("x-y", pos=p) for p in positions]
                area1 = sum([a for a in areas1 if a is not None])
                areas2 = [self.get_area("x-y", pos=p) for p in positions]
                area2 = sum([a for a in areas2 if a is not None])
            else:
                positions = [
                    self.idx_to_pos(self.get_idx(view, sl, idx, pos),
                                    ax=skrt.image._slice_axes[view])
                ]
                area1 = self.get_area(view, sl, idx, pos)
                if area1 is None:
                    area1 = 0
                area2 = other.get_area(view, sl, idx, pos)
                if area2 is None:
                    area2 = 0
            
            # Compute intersecting area on slice(s)
            intersection = 0
            for p in positions:
                polygons1 = self.get_polygons_on_slice(view, pos=p)
                polygons2 = other.get_polygons_on_slice(view, pos=p)
                for p1 in polygons1:
                    for p2 in polygons2:
                        intersection += p1.intersection(p2).area

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

            intersection = (data1 & data2).sum()
            area1 = data1.sum()
            area2 = data2.sum()

        mean_area = np.mean([area1, area2])
        if mean_area == 0:
            return None
        return intersection / mean_area

    def get_volume_ratio(self, other, **kwargs):
        """Get ratio of another ROI's volume with respect to own volume."""

        own_volume = other.get_volume(**kwargs)
        other_volume = self.get_volume(**kwargs)
        if not other_volume or not own_volume:
            return None
        return own_volume / other_volume

    def get_area_ratio(self, other, **kwargs):
        """Get ratio of another ROI's area with respect to own area."""

        own_area = other.get_area(**kwargs)
        other_area = self.get_area(**kwargs)
        if not other_area or not own_area:
            return None
        return own_area / other_area

    def get_volume_diff(self, other, **kwargs):
        """Get own volume minus volume of other ROI."""

        own_volume = self.get_volume(**kwargs)
        other_volume = other.get_volume(**kwargs)
        if not own_volume or not other_volume:
            return None
        return (own_volume - other_volume) / own_volume

    def get_relative_volume_diff(self, other, **kwargs):
        """Get relative volume of another ROI with respect to own volume."""

        own_volume = self.get_volume(**kwargs)
        volume_diff = self.get_volume_diff(other, **kwargs)
        if volume_diff is None:
            return
        return volume_diff / own_volume

    def get_area_diff(self, other, **kwargs):
        """Get absolute area difference between two ROIs."""

        own_area = self.get_area(**kwargs)
        other_area = other.get_area(**kwargs)
        if not own_area or not other_area:
            return
        return own_area - other_area

    def get_relative_area_diff(self, other, **kwargs):
        """Get relative area of another ROI with respect to own area."""

        own_area = self.get_area(**kwargs)
        area_diff = self.get_area_diff(other, **kwargs)
        if area_diff is None:
            return 
        return area_diff / own_area

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
    ):
        """Get vector of surface distances between two ROIs."""

        # Ensure both ROIs are loaded
        self.load()
        other.load()

        # Check whether ROIs are empty
        if not np.any(self.get_mask()) or not np.any(other.get_mask()):
            return

        # Get binary masks and voxel sizes
        if single_slice:
            voxel_size = [self.voxel_size[i] for i in skrt.image._plot_axes[view]]
            mask1 = self.get_slice(view, sl=sl, idx=idx, pos=pos)
            mask2 = other.get_slice(view, sl=sl, idx=idx, pos=pos)
        else:
            if not flatten:
                vx, vy, vz = self.voxel_size
                voxel_size = [vy, vx, vz]
                mask1 = self.get_mask()
                mask2 = other.get_mask()
            else:
                voxel_size = [self.voxel_size[i] for i in skrt.image._plot_axes[view]]
                mask1 = self.get_mask(view, flatten=True)
                mask2 = other.get_mask(view, flatten=True)

        # Make structuring element
        conn2 = ndimage.morphology.generate_binary_structure(2, connectivity)
        if mask1.ndim == 2:
            conn = conn2
        else:
            conn = np.zeros((3, 3, 3), dtype=bool)
            conn[:, :, 1] = conn2

        # Get outer pixel of binary maps
        surf1 = mask1 ^ ndimage.morphology.binary_erosion(mask1, conn)
        surf2 = mask2 ^ ndimage.morphology.binary_erosion(mask2, conn)

        # Make arrays of distances to surface of each pixel
        dist1 = ndimage.morphology.distance_transform_edt(~surf1, voxel_size)
        dist2 = ndimage.morphology.distance_transform_edt(~surf2, voxel_size)

        # Get signed arrays
        if signed:
            dist1 = dist1 * ~mask1 - dist1 * mask1
            dist2 = dist2 * ~mask2 - dist2 * mask2

        # Make vector containing all distances
        sds = np.concatenate([np.ravel(dist1[surf2 != 0]), np.ravel(dist2[surf1 != 0])])
        return sds

    def get_mean_surface_distance(self, other, **kwargs):

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
        units_in_header=False,
        global_vs_slice_header=False,
        name_as_index=True,
        nice_columns=False,
        decimal_places=None,
        force=True
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

                * "centroid": 3D centroid distance vector.
                * "abs_centroid": magnitude of 3D centroid distance vector.
                * "abs_centroid_flat": magnitude of 3D centroid distance vector
                  projected into the plane specified in <view>.
                * "centroid_slice": 2D centroid distance vector on a single slice.
                * "abs_centroid_slice": magnitude of 2D centroid distance 
                  vector on a single slice.
                * "centroid_x": x component of 3D centroid distance vector.
                * "centroid_y": y component of 3D centroid distance vector.
                * "centroid_z": z component of 3D centroid distance vector.

                * "volume_diff": volume difference (own volume - other volume).
                * "rel_volume_diff": volume difference divided by own volume.
                * "volume ratio": volume ratio (own volume / other volume).

                * "area_diff": area difference (own area - other area) on a 
                  single slice.
                * "rel_area_diff": area difference divided by own area on a 
                  single slice.
                * "area ratio": area ratio (own area / other area).
                * "area_diff_flat": area difference of ROIs flattened in the
                  orientation specified in <view>.
                * "rel_area_diff_flat": relative area difference of ROIs
                  flattened in the orientation specified in <view>.
                * "area_ratio_flat": area ratio of ROIs flattened in the
                  orientation specified in <view>.

                * "mean_surface_distance": mean surface distance.
                * "mean_surface_distance_flat": mean surface distance of ROIs
                  flattened in the orientation specified in <view>.
                * "rms_surface_distance": RMS surface distance.
                * "rms_surface_distance_flat": RMS surface distance of ROIs
                  flattened in the orientation specified in <view>.
                * "hausdorff_distance": Hausdorff distance.
                * "hausdorff_distance_flat": Hausdorff distance of ROIs
                  flattened in the orientation specified in <view>.


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
        """

        # If no index given, use central slice
        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)

        # Default metrics
        if metrics is None:
            metrics = ["dice", "centroid"]

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

            # Dice score
            if m == "dice":
                comp[m] = self.get_dice(roi, method=method)
            elif m == "dice_flat":
                comp[m] = self.get_dice(
                    roi, 
                    view=view,
                    method=method,
                    flatten=True, 
                )
            elif m == "dice_slice":
                comp[m] = self.get_dice(roi, method=method, **slice_kwargs)

            # Centroid distances
            elif m == "centroid":
                centroid = self.get_centroid_distance(
                    roi, 
                    units=centroid_units, 
                    method=method,
                    force=force
                )
                for i, ax in enumerate(skrt.image._axes):
                    comp[f"centroid_{ax}"] = centroid[i]
            elif m == "abs_centroid":
                comp[m] = self.get_abs_centroid_distance(
                    roi, 
                    units=centroid_units,
                    method=method, 
                    force=force
                )
            elif m == "abs_centroid_flat":
                comp[m] = self.get_abs_centroid_distance(
                    roi,
                    view=view,
                    units=centroid_units,
                    method=method,
                    flatten=True,
                    force=force
                )
            elif m == "centroid_slice":
                centroid = self.get_centroid_distance(
                    roi,
                    units=centroid_units,
                    method=method,
                    **slice_kwargs
                )
                for i, i_ax in enumerate(skrt.image._plot_axes):
                    ax = skrt.image.axes[i_ax]
                    comp[f"centroid_slice_{ax}"] = centroid[i]
            elif m == "abs_centroid_slice":
                centroid = self.get_abs_centroid_distance(
                    roi,
                    units=centroid_units,
                    method=method,
                    **slice_kwargs
                )

            # Volume metrics
            elif m == "volume_diff":
                comp[m] = self.get_volume_diff(
                    roi,
                    units=vol_units,
                    method=method,
                    force=force
                )
            elif m == "rel_volume_diff":
                comp[m] = self.get_relative_volume_diff(
                    roi,
                    method=method,
                    force=force
                )
            elif m == "volume_ratio":
                comp[m] = self.get_volume_ratio(
                    roi,
                    method=method,
                    force=force
                )

            # Area metrics
            elif m == "area_diff":
                comp[m] = self.get_area_diff(
                    roi,
                    units=area_units,
                    method=method,
                    **slice_kwargs
                )
            elif m == "rel_area_diff":
                comp[m] = self.get_relative_area_diff(
                    roi,
                    method=method,
                    **slice_kwargs
                )
            elif m == "area_ratio":
                comp[m] = self.get_area_ratio(
                    roi,
                    method=method,
                    **slice_kwargs
                )
            elif m == "rel_area_diff_flat":
                comp[m] = self.get_relative_area_diff(
                    roi,
                    view=view,
                    method=method,
                    flatten=True,
                )
            elif m == "area_ratio_flat":
                comp[m] = self.get_area_ratio(
                    roi,
                    view=view,
                    method=method,
                    flatten=True,
                )

            # Surface distance metrics
            elif m == "mean_surface_distance":
                comp[m] = self.get_mean_surface_distance(roi)
            elif m == "mean_surface_distance_flat":
                comp[m] = self.get_mean_surface_distance(roi, flatten=True)
            elif m == "rms_surface_distance":
                comp[m] = self.get_rms_surface_distance(roi)
            elif m == "rms_surface_distance_flat":
                comp[m] = self.get_rms_surface_distance(roi, flatten=True)
            elif m == "hausdorff_distance":
                comp[m] = self.get_hausdorff_distance(roi)
            elif m == "hausdorff_distance_flat":
                comp[m] = self.get_hausdorff_distance(roi, flatten=True)

            else:
                found = False

                # Axis-specific metrics
                for i, ax in enumerate(skrt.image._axes):

                    # Global centroid position on a given axis
                    if m == f"centroid_{ax}":
                        comp[m] = self.get_abs_centroid_distance(
                            roi,
                            units=centroid_units,
                            method=method,
                            force=force
                        )[i]
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
        df = pd.DataFrame(comp_named, index=[self.name])

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
                            grey=False):
        """Get name of comparison between this ROI and another."""

        own_name = self.name
        other_name = roi.name

        if colored:
            own_name = get_colored_roi_string(self, grey)
            other_name = get_colored_roi_string(roi, grey)

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
        plot_type='contour',
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

        plot_type : str, default=None
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
            plot_type = self.default_geom_method

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

        # Set up axes
        self.set_ax(plot_type, include_image, view, ax, gs, figsize)

        # Adjust centroid marker size
        if contour_kwargs is None:
            contour_kwargs = {}
        if "centroid" in plot_type:
            if linewidth is None:
                linewidth = defaultParams["lines.linewidth"][0]
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
            will be taken from the minimum distance between slice positions in 
            the x-y contours dictionary.

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
        """

        extents = [self.get_extent(ax=ax) for ax in skrt.image._axes]
        slice_thickness  = self.get_slice_thickness_contours()
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

        self.image = im
        self.contours_only = False
        if self.loaded_mask and not im.has_same_geometry(self.mask):
            if not hasattr(self, "input_contours"):
                self.input_contours = self.get_contours("x-y")
            self.mask = None
            self.loaded_mask = False

        # Set geoemtric info
        data_shape = im.get_data().shape
        self.shape = [data_shape[1], data_shape[0], data_shape[2]]
        self.voxel_size = im.get_voxel_size()
        self.origin = im.get_origin()
        self.affine = im.get_affine()

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

        # View with image
        if include_image and self.image is not None:
            bv = BetterViewer(self.image, rois=self, **kwargs)

        # View without image
        else:

            roi_tmp = self

            # Make dummy image for background
            if self.contours_only:
                im = self.get_dummy_image(
                    voxel_size=voxel_size,
                    buffer=buffer
                )
                im.title = self.name
                roi_tmp = self.clone(copy_data=False)
                roi_tmp.image = im
                roi_tmp.load(force=True)
            else:
                im = skrt.image.Image(
                    np.ones(self.shape) * 1e4,
                    affine=self.affine,
                    title=self.name
                )

            # Create viewer
            bv = BetterViewer(im, rois=roi_tmp, **kwargs)

        # Adjust UI
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
            self.image.plot(view, idx=idx, ax=self.ax, show=False)

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
        if not self.on_slice(view, idx=idx) or self.empty:
            return

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs.setdefault("color", color)
        contour_kwargs.setdefault("linewidth", linewidth)

        # Plot underlying image
        if include_image and self.image is not None:
            self.image.plot(view, idx=idx, ax=self.ax, show=False)

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
            if names:
                roi1_name = names[0]
                roi2_name = names[1]
            else:
                roi1_name = self.name
                roi2_name = other.name
            handles = [
                mpatches.Patch(color=self.color, label=roi1_name),
                mpatches.Patch(color=roi2_color, label=roi2_name),
            ]
            self.ax.legend(
                handles=handles, framealpha=1, facecolor="white", 
                loc="lower left"
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
                for contour in contours:
                    for point in contour:
                        points.append(f"{point[0]:.3f} {point[1]:.3f} {z:.3f}")

            with open(outname, "w") as file:
                file.write("point\n")
                file.write(f"{len(points)}\n")
                file.write("\n".join(points))

            return

        # Write array to nifti or npy
        elif ext != ".dcm":
            self.create_mask()
            self.mask.write(outname, **kwargs)
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
                    polygon = contour_to_polygon(contour)
                    polygon = affinity.translate(polygon, *translation_2d)
                    polygon = affinity.rotate(polygon, angle, centre_2d)
                    polygon = affinity.scale(polygon, scale, scale, scale,
                            centre_2d)
                    new_contours[new_key].append(polygon_to_contour(polygon))

            self.reset_contours(new_contours)

        else:
            # Apply transform to roi mask
            self.create_mask()
            self.mask.transform(scale, translation, rotation,
                    centre, resample, restore, 0, fill_value)
            self.reset_mask()

    def crop_to_roi(self, roi, **kwargs):
        """
        Crop ROI mask to region covered by an ROI.
        """
        self.create_mask()
        self.mask.crop_to_roi(roi, **kwargs)
        self.shape = self.get_mask().shape


class StructureSet(skrt.core.Archive):
    """Structure set."""

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

        `**`kwargs :
            Additional keyword args to use when initialising new ROI objects.
        """

        # Clone from another StructureSet object
        if issubclass(type(path), StructureSet):
            path.clone_attrs(self)
            return

        self.name = name
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
        self.roi_kwargs = kwargs

        path = path if isinstance(path, str) else ""
        skrt.core.Archive.__init__(self, path)

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

            # Put affine matrix into kwargs
            if self.image is not None:
                self.roi_kwargs["affine"] = self.image.get_affine()

            # Load input array into image
            array = skrt.image.Image(sources).get_data(standardise=True).astype(int)
            n = array.max()
            i_name = 0
            for i in range(0, n):
                if self.names is not None and i_name < len(self.names):
                    name = self.names[i_name]
                    i_name += 1
                else:
                    name = f"ROI {i}"
                self.rois.append(ROI(
                    array == i + 1, 
                    image=self.image,
                    name=name,
                    **self.roi_kwargs
                ))
            sources = []
            self.loaded = True

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

        for source in sources_expanded:

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

        self.rename_rois(keep_renamed_only=self.keep_renamed_only)
        self.filter_rois()
        self.recolor_rois(self.colors)
        for roi in self.rois:
            roi.structure_set = self

        self.loaded = True

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

    def set_image(self, image):
        """Set image for self and all ROIs. Image.add_structure_set(self) will
        also be called."""

        # Convert to Image object if needed
        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)

        # Assign to self and all ROIs
        self.image = image
        for roi in self.rois:
            roi.set_image(image)

        # Assign self to the image
        if image is not None:
            image.add_structure_set(self)

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
                for i, s in enumerate(self.rois):

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

        ss = self.clone(copy_roi_data=copy_roi_data)
        if name is not None:
            ss.name = name
        ss.rename_rois(names, keep_renamed_only=keep_renamed_only)
        ss.filter_rois(to_keep, to_remove)
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
            if fnmatch.fnmatch(roi.name, wildcard):
                rois.append(roi)
        return rois

    def get_roi_names(self, original=False):
        """
        Get list of names of ROIs in this structure set. If <original> is True,
        get the original names of the ROIs.
        """

        if not original:
            return [s.name for s in self.get_rois()]
        else:
            return [s.original_name for s in self.get_rois()]

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

    def get_geometry(
        self, 
        name_as_index=True, 
        html=False, 
        colored=False,
        greyed_out=None,
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
                col_str = get_colored_roi_string(roi, grey=(roi in greyed_out))
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
                s for s in self.get_roi_names() if s in other.get_roi_names()
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
        names=None, **kwargs
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
                roi2, legend=legend, save_as=outname, names=names, **kwargs
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
            outname = os.path.join(outdir, outname)

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
            s.write(outdir=outdir, ext=ext, **kwargs)

    def plot(
        self,
        view="x-y",
        plot_type=None,
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

        # Plot with image
        if include_image and self.image is not None:

            roi_kwargs = {}
            if opacity is not None:
                roi_kwargs["opacity"] = opacity
            if linewidth is not None:
                roi_kwargs["linewidth"] = opacity

            self.image.plot(
                view,
                sl=sl, 
                idx=idx,
                pos=pos,
                rois=self,
                roi_plot_type=plot_type,
                roi_kwargs=roi_kwargs,
                centre_on_roi=centre_on_roi,
                show=show,
                save_as=save_as,
                legend=legend,
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
                    mpatches.Patch(color=consensus_color, label=consensus.name))

            # Plot excluded ROI on top
            if exclude_from_consensus is not None:
                kwargs["include_image"] = False
                excluded = self.get_roi(exclude_from_consensus)
                excluded.plot(view, sl=sl, idx=idx, pos=pos, plot_type=plot_type,
                              opacity=opacity, linewidth=linewidth, show=False,
                              ax=self.ax, **kwargs)
                if legend:
                    roi_handles.append(
                        mpatches.Patch(color=excluded.color, label=excluded.name))

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
                         ax=ax)

            self.fig = central.fig
            self.ax = central.ax

            if legend:
                roi_handles.append(
                    mpatches.Patch(color=central.color, label=central.name))

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
                        roi_handles.append(mpatches.Patch(
                            color=roi.color,
                            label=roi.name))

        # Draw legend
        if legend and len(roi_handles):
            self.ax.legend(
                handles=roi_handles, loc=legend_loc, facecolor="white",
                framealpha=1
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

    def get_centre(self):
        """Get 3D centre of the area covered by ROIs."""

        extents = [self.get_extent(ax=ax) for ax in skrt.image._axes]
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
            if self.rois[0].contours_only:
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

            if rois is None:
                rois = structure_set_tmp
            else:
                if skrt.core.is_list(rois):
                    rois = [structure_set_tmp] + rois
                else:
                    rois = [structure_set_tmp, rois]
            kwargs["show"] = False
            bv = BetterViewer(im, rois=rois, **kwargs)
            bv.make_ui(no_intensity=True)
            bv.show()

        return bv

    def get_extent(self, **kwargs):
        """Get min and max extent of all ROIs in the StructureSet."""

        all_extents = []
        for roi in self.get_rois(ignore_empty=True):
            all_extents.extend(roi.get_extent(**kwargs))
        return min(all_extents), max(all_extents)

    def get_dummy_image(self, **kwargs):
        """Make a dummy image that covers the area spanned by all ROIs in this
        StructureSet. Returns an Image object.

        **Parameters:**
        
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

        buffer : int, default=1
            Number of empty buffer voxels to add outside the ROI in each
            direction.
        """

        extents = [self.get_extent(ax=ax) for ax in skrt.image._axes]
        slice_thickness = self.get_rois(ignore_empty=True)[0].get_slice_thickness_contours()
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

    polygon = geometry.Polygon(contour)

    delta = 0.001
    if not polygon.is_valid:
        tmp = geometry.Polygon(polygon)
        buffer = 0.
        while not polygon.is_valid:
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
    buffer=1
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

    buffer : float, default=1
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
        mask += roi.get_mask(standardise=True)
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
                    padding: 2px 10px;
                }
                th {
                    background-color: rgb(225, 225, 225);
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


def get_colored_roi_string(roi, grey=False):
    """Get ROI name in HTML with background color from roi.color. If grey=True,
    the background color will be grey."""

    if grey:
        red, green, blue = 255, 255, 255
        text_col = 200, 200, 200
    else:
        red, green, blue = [c * 255 for c in roi.color[:3]]
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
        comp_name = roi1.get_comparison_name(roi2, colored=colored, grey=grey)
        if name_as_index:
            df_row.rename({df_row.index[0]: comp_name}, inplace=True)
        else:
            df_row.iloc[0, 0] = comp_name

        dfs.append(df_row)

    df = pd.concat(dfs)
    if html:
        return df_to_html(df)
    return df
