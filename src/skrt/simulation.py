"""Classes for creating synthetic images with simple geometries."""

import os
import pathlib
import shutil

import numpy as np

from skrt.core import is_list, to_list
from skrt.image import Image, _axes
from skrt.patient import Patient
from skrt.structures import StructureSet, ROI


class SyntheticImage(Image):
    """Class for creating synthetic image data with simple geometric shapes."""

    def __init__(
        self,
        shape,
        filename=None,
        origin=(0, 0, 0),
        voxel_size=(1, 1, 1),
        intensity=-1000,
        noise_std=None,
        auto_timestamp=False,
    ):
        """Create an initially blank synthetic image to which geometric
        ROIs can be added.

        **Parameters:**

        shape : int/tuple
            Dimensions of the image array to create in order (x, y, z).
            If an int is given, the image will be created with dimensions
            (shape, shape, shape).

        filename : str, default=None
            Name of output file or directory. If given, the file will
            automatically be written; otherwise, no file will be written until
            the 'write' method is called.

        origin : tuple, default=(0, 0, 0)
            Origin in mm for the image in order (x, y, z).

        voxel_size : tuple, default=(1, 1, 1)
            Voxel sizes in mm for the image in order (x, y, z).

        intensity : float, default=-1000
            Intensity in HU for the background of the image.

        noise_std : float, default=None
            Standard deviation of Gaussian noise to apply to the image.
            If None, no noise will be applied.

        auto_timestamp : bool default=False
            If true and no valid timestamp is found within the path string,
            timestamp generated from current date and time.
        """

        # Assign properties
        shape = to_list(shape)
        self.shape = [shape[1], shape[0], shape[2]]
        self.n_voxels = shape
        self.voxel_size = to_list(voxel_size)
        self.origin = origin
        self.max_hu = 0 if noise_std is None else noise_std * 3
        self.min_hu = -self.max_hu if self.max_hu != 0 else -20
        self.noise_std = noise_std
        self.bg_intensity = intensity
        self.shapes = []
        self.roi_shapes = {}
        self.rois = {}
        self.groups = {}
        self.shape_count = {}
        self.translation = None
        self.rotation = None

        # Initialise as Image
        super().__init__(
            self.get_background(),
            voxel_size=self.voxel_size,
            origin=self.origin,
            auto_timestamp=auto_timestamp,
        )

        # Write to file if a filename is given
        if filename is not None:
            self.filename = os.path.expanduser(filename)
            self.write()

    def update(self):
        """Update self.data so that it contains all current shapes."""

        # Get background array
        data = self.get_background(with_noise=False).copy()

        # Add shapes
        for shape in self.shapes:
            data[shape.get_data(self.get_coords())] = shape.intensity

        # Apply noise
        data = self.apply_noise(data)
        self.data = data

        # Assign structure set
        self.clear_structure_sets()
        self.update_rois()
        self._image = Image(self)
        self._structure_set = StructureSet(
            list(self.rois.values()), image=self._image, auto_timestamp=True
        )
        self.structure_sets = [self._structure_set]

    def get_image(self):
        """Get self as an Image object."""

        self.update()
        return self._image

    def get_data(self, *args, **kwargs):
        """Get Image data."""

        self.update()
        return Image.get_data(self, **kwargs)

    def plot(self, **kwargs):
        """Plot the current image with ROIs overlaid."""

        self.update()
        self.update_rois()
        kwargs.setdefault("rois", -1)
        Image.plot(self, **kwargs)

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
        kwargs.setdefault("rois", self.get_structure_set())

        return BetterViewer(ims, **kwargs)

    def get_roi_data(self):
        """Get dict of ROIs and names with any transformations applied."""

        roi_data = {}
        for name, shape in self.roi_shapes.items():
            data = shape.get_data(self.get_coords())
            roi_data[name] = data
        return roi_data

    def get_structure_set(self):
        """Return StructureSet containing own structures."""

        self.update()
        return self._structure_set

    def update_roi(self, name):
        """Update an ROI to ensure it has the correct data."""

        self.rois[name].data = self.roi_shapes[name].get_data(self.get_coords())

    def update_rois(self):
        """Update all ROIs to have the correct data."""

        for name in self.rois:
            self.update_roi(name)

    def get_roi(self, name):
        """Get a named ROI as an ROI object."""

        if name not in self.rois:
            print("ROI", name, "not found!")
            return None

        self.update_roi(name)
        return self.rois[name]

    def get_rois(self):
        """Get list of all owned ROI objects."""

        self.update_rois()
        return list(self.rois.values())

    def write(self, outname=None, overwrite_roi_dir=False):
        """Write image data to an output file."""

        # Check filename
        if outname is None:
            if hasattr(self, "filename"):
                outname = self.filename
            else:
                raise RuntimeError(
                    "Filename must be specified in __init__() or write()!"
                )

        # Write image data
        self.update()
        Image.write(self, outname)

        # Write ROIs
        structure_set = self.get_structure_set()
        exts = [".nii", ".nii.gz", ".npy"]
        outdir = outname
        ext_to_use = None
        for ext in exts:
            if outname.endswith(ext):
                ext_to_use = ext
                outdir = outname.replace(ext, "")
        structure_set.write(
            outdir=outdir, ext=ext_to_use, overwrite=overwrite_roi_dir
        )

    def write_dicom_dataset(
        self,
        patient_id="001",
        outdir="synthetic_dicom",
        modality="CT",
        series_number=1,
        structure_set_label="Geometric structures",
        root_uid=None,
        verbose=False,
    ):
        """
        Write image and associated structure set as a patient DICOM dataset.

        **Parameters:**

        patient_id: str, default="001"
            Patient identifier to be assigned to the dataset.

        outdir: str/pathlib.Pat, default="synthetic_dicom"
            Path to directory where dataset is to be written.

        modality: str, default="CT"
            Modality to be assigned to synthetic image.

        series_number: int, default=1
            Series number to be assigned to synthetic image.

        structure_set_label: str, default="Geometric structures"
            Label to be assigned to structure set, if present.

        root_uid : str, default=None
            Root to be used in Globally Unique Identifiers (GUIDs).  This
            should uniquely identify the institution or group generating
            the GUID.  If None, the value of pydicom.uid.PYDICOM_ROOT_UID
            is used.

        verbose : bool, default=False
            If True, print information about progress of dataset writing.
        """
        # Define permanent and temporary output directories.
        patient_dir = pathlib.Path(outdir) / patient_id
        if patient_dir.exists():
            shutil.rmtree(patient_dir)
        tmp_dir = pathlib.Path(outdir) / "tmp" / patient_id

        # Obtain reference to structure set, and write associated image.
        # (The original image is a simulation.SyntheticImage object,
        # but the associated # image is a skrt.patient.Image object.)
        ss = self.get_structure_set()
        ss.get_image().write(
            tmp_dir,
            patient_id=patient_id,
            modality=modality,
            root_uid=root_uid,
            header_extras={"SeriesNumber": series_number},
            verbose=verbose,
        )

        # Write structure set, if non-empty,
        # after associating to it the DICOM image,
        if ss.get_roi_names():
            im = Image(tmp_dir)
            ds = im.get_dicom_dataset()
            header_extras = {
                "SeriesNumber": ds.SeriesNumber,
                "StudyInstanceUID": ds.StudyInstanceUID,
                "StudyDate": ds.StudyDate,
                "StudyTime": ds.StudyTime,
                "StructureSetLabel": structure_set_label,
            }
            ss.set_image(im)
            ss.write(
                outdir=tmp_dir,
                ext=".dcm",
                patient_id=patient_id,
                modality=modality,
                root_uid=root_uid,
                header_extras=header_extras,
                verbose=verbose,
            )

        # Load the DICOM data, then rewrite as a sorted dataset.
        p = Patient(tmp_dir, unsorted_dicom=True)
        p.copy_dicom(patient_dir.parent)
        shutil.rmtree(tmp_dir.parent)

        return patient_dir

    def get_background(self, with_noise=True):
        """Make blank image array or noisy array."""

        bkg = np.ones(self.shape) * self.bg_intensity
        if with_noise:
            bkg = self.apply_noise(bkg)
        return bkg

    def apply_noise(self, array):
        """Apply background noise to an array."""

        if self.noise_std is not None:
            array += np.random.normal(0, self.noise_std, array.shape)
        return array

    def set_noise_std(self, std):
        """Set noise standard deviation."""

        self.noise_std = std
        self.update()

    def reset(self):
        """Remove all shapes."""

        self.shapes = []
        self.roi_shapes = {}
        self.groups = {}
        self.shape_count = {}
        self.translation = None
        self.rotation = None
        self.update()

    def add_shape(self, shape, shape_type, is_roi, above, group):
        if above:
            self.shapes.append(shape)
        else:
            self.shapes.insert(0, shape)

        # Automatically treat as ROI if given a group or name
        if is_roi is None and (group is not None or shape.name is not None):
            is_roi = True

        if is_roi:
            if group is not None:
                if group not in self.groups:
                    self.groups[group] = ShapeGroup([shape], name=group)
                    self.roi_shapes[group] = self.groups[group]
                else:
                    self.groups[group].add_shape(shape)
                self.rois[group] = ROI(
                    self.groups[group].get_data(self.get_coords()),
                    name=group,
                    affine=self.get_affine(),
                    image=self,
                )
            else:
                if shape_type not in self.shape_count:
                    self.shape_count[shape_type] = 1
                else:
                    self.shape_count[shape_type] += 1

                if shape.name is None:
                    shape.name = f"{shape_type}{self.shape_count[shape_type]}"

                self.roi_shapes[shape.name] = shape
                self.rois[shape.name] = ROI(
                    shape.get_data(self.get_coords()),
                    name=shape.name,
                    affine=self.get_affine(),
                    image=self,
                )

        self.min_hu = min([shape.intensity, self.min_hu])
        self.max_hu = max([shape.intensity, self.max_hu])

    def add_sphere(
        self,
        radius,
        centre=None,
        intensity=0,
        is_roi=None,
        name=None,
        above=True,
        group=None,
    ):
        if centre is None:
            centre = self.get_centre()
        sphere = Sphere(radius, centre, intensity, name)
        self.add_shape(sphere, "sphere", is_roi, above, group)

    def add_cylinder(
        self,
        radius,
        length,
        axis="z",
        centre=None,
        intensity=0,
        is_roi=None,
        name=None,
        above=True,
        group=None,
    ):
        if centre is None:
            centre = self.get_centre()
        cylinder = Cylinder(radius, length, axis, centre, intensity, name)
        self.add_shape(cylinder, "cylinder", is_roi, above, group)

    def add_cube(
        self,
        side_length,
        centre=None,
        intensity=0,
        is_roi=None,
        name=None,
        above=True,
        group=None,
    ):
        self.add_cuboid(
            side_length, centre, intensity, is_roi, name, above, group=group
        )

    def add_cuboid(
        self,
        side_length,
        centre=None,
        intensity=0,
        is_roi=None,
        name=None,
        above=True,
        group=None,
    ):
        if centre is None:
            centre = self.get_centre()
        side_length = to_list(side_length)

        cuboid = Cuboid(side_length, centre, intensity, name)
        self.add_shape(cuboid, "cuboid", is_roi, above, group)

    def add_grid(
        self,
        spacing,
        thickness=1,
        intensity=0,
        axis=None,
        name=None,
        above=True,
    ):
        grid = Grid(self.shape, spacing, thickness, intensity, axis, name)
        self.add_shape(grid, "grid", False, above, group=None)

    def reset_transforms(self):
        """Remove any rotations or translations."""

        self.translation = None
        self.rotation = None
        for shape in self.shapes:
            shape.translation = None
            shape.rotation = None

    def translate(self, dx=0, dy=0, dz=0):
        """Set a translation to apply to the final image."""

        self.translation = (dy, dx, dz)

    def rotate(self, yaw=0, pitch=0, roll=0):
        """Set a rotation to apply to the final image."""

        self.rotation = (yaw, pitch, roll)

    def rescale(self, v_min=0.0, v_max=1.0, constant=0.5):
        """
        Linearly rescale image greyscale values,
        so that they span a specified range.

        Rescaling is applied also to greyscale values for shapes.

        **Parameters:**

        v_min: float, default=0.0
            Minimum greyscale value after rescaling.

        v_max: float, default=1.0
            Maximum greyscale value after rescaling.

        constant: float, default=0.5
            Greyscale value to assign after rescaling if all values
            in the original image are the same.  If None,
            original value is kept.
        """
        # Perform rescaling.
        u_min = self.get_min(force=True)
        u_max = self.get_max(force=True)
        du = u_max - u_min
        dv = v_max - v_min
        if du:
            self.data = v_min + (
                (self.data.astype(np.float32) - u_min) * (dv / du)
            )
            for shape in self.shapes:
                shape.intensity = v_min + (shape.intensity - u_min) * (dv / du)
        elif constant is not None:
            self.data.fill(constant)
            for shape in self.shapes:
                shape.intensity = constant
        else:
            return None

        # Remove any prior standardised data.
        self._sdata = None
        self._saffine = None

        # Remove any cached values for maxium and minimum.
        self._max = None
        self._min = None

        return None


class ShapeGroup:
    """Class for grouping multiple shapes, to be represented by a single ROI."""
    def __init__(self, shapes, name):
        self.name = name
        self.shapes = shapes

    def add_shape(self, shape):
        self.shapes.append(shape)

    def get_data(self, coords):
        data = self.shapes[0].get_data(coords)
        for shape in self.shapes[1:]:
            data += shape.get_data(coords)
        return data


class Sphere:
    """Class representing a sphere."""
    def __init__(self, radius, centre, intensity, name=None):
        self.name = name
        self.radius = radius
        self.centre = centre
        self.intensity = intensity

    def get_data(self, coords):
        distance_to_centre = np.sqrt(
            (coords[1] - self.centre[1]) ** 2
            + (coords[0] - self.centre[0]) ** 2
            + (coords[2] - self.centre[2]) ** 2
        )
        return distance_to_centre <= self.radius


class Cuboid:
    """Class representing a cuboid."""
    def __init__(self, side_length, centre, intensity, name=None):
        self.name = name
        self.side_length = to_list(side_length)
        self.centre = centre
        self.intensity = intensity

    def get_data(self, coords):
        try:
            data = (
                (
                    np.absolute(coords[1] - self.centre[1])
                    <= self.side_length[1] / 2
                )
                & (
                    np.absolute(coords[0] - self.centre[0])
                    <= self.side_length[0] / 2
                )
                & (
                    np.absolute(coords[2] - self.centre[2])
                    <= self.side_length[2] / 2
                )
            )
            return data
        except TypeError:
            print("centre:", self.centre)
            print("side length:", self.side_length)
            return None

class Cylinder:
    """Class representing a cylinder."""
    def __init__(self, radius, length, axis, centre, intensity, name=None):
        self.radius = radius
        self.length = length
        self.centre = centre
        self.axis = axis
        self.intensity = intensity
        self.name = name

    def get_data(self, coords):
        # Get coordinates in each direction
        axis_idx = _axes.index(self.axis)
        circle_idx = [i for i in range(3) if i != axis_idx]
        coords_c1 = coords[circle_idx[1]]
        coords_c2 = coords[circle_idx[0]]
        coords_length = coords[axis_idx]

        # Get centre in each direction
        centre = [self.centre[1], self.centre[0], self.centre[2]]
        centre_c1 = centre[circle_idx[1]]
        centre_c2 = centre[circle_idx[0]]
        centre_length = centre[axis_idx]

        # Make cylinder array
        data = (
            np.sqrt((coords_c1 - centre_c1) ** 2 + (coords_c2 - centre_c2) ** 2)
            <= self.radius
        ) & (np.absolute(coords_length - centre_length) <= self.length / 2)
        return data


class Grid:
    """Class representing a grid."""
    def __init__(
        self, shape, spacing, thickness, intensity, axis=None, name=None
    ):
        self.name = name
        self.spacing = to_list(spacing)
        self.thickness = to_list(thickness)
        self.intensity = intensity
        self.axis = axis
        self.shape = shape

    def get_data(self, _):
        coords = np.meshgrid(
            np.arange(0, self.shape[1]),
            np.arange(0, self.shape[0]),
            np.arange(0, self.shape[2]),
        )
        if self.axis is not None:
            axis = _axes.index(self.axis)
            ax1, ax2 = [i for i in [0, 1, 2] if i != axis]
            return (coords[ax1] % self.spacing[ax1] < self.thickness[ax1]) | (
                coords[ax2] % self.spacing[ax2] < self.thickness[ax2]
            )
        return (
            (coords[1] % self.spacing[1] < self.thickness[1])
            | (coords[0] % self.spacing[0] < self.thickness[0])
            | (coords[2] % self.spacing[2] < self.thickness[2])
        )


def make_grid(
    image,
    spacing=(30, 30, 30),
    thickness=(2, 2, 2),
    background=-1000,
    foreground=1000,
    voxel_units=False,
):
    """
    Create a synthetic image of a grid pattern for a specified image.

    **Parameters:**

    image : skrt.image.Image
        Reference image, for which grid pattern is to be created.

    spacing : tuple, default=(30, 30, 30)
        Spacing along (x, y, z) directions of grid lines.  If
        voxel_units is True, values are taken to be in numbers
        of voxels.  Otherwise, values are taken to be in the
        same units as the voxel dimensions of the reference image.

    thickness: tuple, default=(2, 2, 2)
        Thickness along (x, y, z) directions of grid lines.  If
        voxel_units is True, values are taken to be in numbers
        of voxels.  Otherwise, values are taken to be in the
        same units as the voxel dimensions of the reference image.

    background: int/float, default=-1000
        Intensity value to be assigned to voxels not on grid lines.

    foreground: int/float, default=1000
        Intensity value to be assigned to voxels on grid lines.

    voxel_units: bool, default=False
        If True, values for spacing and thickness are taken to be
        in numbers of voxels.  If False, values for spacing and
        thickness are taken to be in the same units as the
        voxel dimensions of the reference image.
    """
    image.load()

    if not voxel_units:
        spacing = [
            max(round(spacing[i] / abs(image.get_voxel_size()[i])), 1)
            for i in range(len(spacing))
        ]
        thickness = [
            max(round(thickness[i] / abs(image.get_voxel_size()[i])), 1)
            for i in range(len(thickness))
        ]

    grid_image = SyntheticImage(
        shape=image.get_n_voxels(),
        origin=image.get_origin(),
        voxel_size=image.get_voxel_size(),
        intensity=background,
    )
    grid_image.add_grid(spacing, thickness, foreground)

    return grid_image


def make_head(
    shape=(256, 256, (50, 80)),
    origin=None,
    voxel_size=(1, 1, 3),
    noise_std=10,
    intensity=-1000,
):
    """
    Create an image featuring a loose approximation of a head.

    In the parameters described below, each value may be a single
    number or a two-element tuple.  In the latter case, the number used
    will be drawn at random from a uniform distribution over the range
    from the first element to the second.

    **Parameters:**
    shape : int/tuple
        Dimensions of the image array to create in order (x, y, z).
        If an int is given, the image will be created with dimensions
        (shape, shape, shape).

    origin : tuple, default=None
        Origin in mm for the image in order (x, y, z).  If None, set
        the origin so that that the image centre is at (0, 0, 0).

    voxel_size : tuple, default=(1, 1, 3)
        Voxel sizes in mm for the image in order (x, y, z).

    noise_std : float, default=None
        Standard deviation of Gaussian noise to apply to the image.
        If None, no noise will be applied.

        intensity : float, default=-1000
            Intensity in HU for the background of the image.
    """
    ##################
    # Image settings #
    ##################
    shape = [round(value) for value in get_values(shape) or []]
    origin = get_values(origin) or [-shape[0] / 2, -shape[1] / 2, 0]
    voxel_size = get_values(voxel_size)
    noise_std = get_value(noise_std)

    ####################################
    # Randomised anatomical parameters #
    ####################################
    # Head
    head_height = np.random.uniform(100, 170)
    head_radius = np.random.uniform(40, 80)

    # Ears
    ear_offset_x = np.random.uniform(-5, 5)
    ear_offset_y = np.random.uniform(-5, 5)
    ear_offset_z = np.random.uniform(-5, 10)
    ear_size_x = np.random.uniform(20, 60)
    ear_size_y = np.random.uniform(5, 20)
    ear_size_z = np.random.uniform(40, 60)

    # Eyes
    eye_angle = np.random.uniform(15, 40)
    eye_radius = np.random.uniform(5, 15)
    eye_offset_z = np.random.uniform(-20, -30)

    # Teeth
    teeth_bottom_row_from_chin = np.random.uniform(10, 20)
    teeth_row_spacing = np.random.uniform(5, 15)
    teeth_angle_spacing = np.random.uniform(5, 10)
    teeth_size = np.random.uniform(5, 10)
    teeth_radius_frac = np.random.uniform(0.7, 0.9)

    ##############
    # Make image #
    ##############
    # Background and head
    head_image = SyntheticImage(
        shape, origin=origin, voxel_size=voxel_size, intensity=intensity,
        noise_std=noise_std
        )
    centre = head_image.get_centre()
    head_image.add_cylinder(
        radius=head_radius, length=head_height, group="head"
    )

    # Add ears
    for i in [-1, 1]:
        head_image.add_cuboid(
            side_length=[ear_size_x, ear_size_y, ear_size_z],
            centre=[
                centre[0] + i * (head_radius + ear_offset_x),
                centre[1] + ear_offset_y,
                centre[2] + ear_offset_z,
            ],
            group="head",
        )

    # Add eyes
    for i, name in zip([-1, 1], ["left", "right"]):
        head_image.add_sphere(
            radius=eye_radius,
            centre=[
                centre[0] - head_radius * np.sin(np.radians(eye_angle)) * i,
                centre[1] - head_radius * np.cos(np.radians(eye_angle)),
                centre[2] - eye_offset_z,
            ],
            intensity=40,
            name="eye_" + name,
        )

    # Add teeth
    for i in [-2, -1, 1, 2]:
        radius = head_radius * teeth_radius_frac
        angle = np.sign(i) * np.radians((abs(i) - 0.5) * teeth_angle_spacing)
        z = centre[2] - head_height / 2 + teeth_bottom_row_from_chin
        head_image.add_cube(
            side_length=teeth_size,
            centre=[
                centre[0] + np.sin(angle) * radius,
                centre[1] - np.cos(angle) * radius,
                z,
            ],
            intensity=100,
            group="teeth",
        )
        head_image.add_cube(
            side_length=teeth_size,
            centre=[
                centre[0] + np.sin(angle) * radius,
                centre[1] - np.cos(angle) * radius,
                z + teeth_row_spacing,
            ],
            intensity=100,
            group="teeth",
        )

    return head_image


def get_value(val):
    return np.random.uniform(*val) if is_list(val) else val


def get_values(vals):
    return [get_value(val) for val in vals] if is_list(vals) else None
