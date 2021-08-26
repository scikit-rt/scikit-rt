'''Classes for loading and comparing medical images.'''

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pydicom.dataset import FileDataset, FileMetaDataset
from scipy import interpolate
import datetime
import glob
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pydicom
import tempfile
import time
import uuid

import skrt.core


_axes = ['x', 'y', 'z']
_slice_axes = {'x-y': 2, 'y-z': 0, 'x-z': 1}
_plot_axes = {'x-y': [0, 1], 'y-z': [2, 1], 'x-z': [2, 0]}
_default_figsize = 6
_default_stations = {'0210167': 'LA3', '0210292': 'LA4'}


class Image(skrt.core.Archive):
    '''Loads and stores a medical image and its geometrical properties, either
    from a dicom/nifti file or a numpy array.'''

    def __init__(
        self,
        path,
        load=True,
        title=None,
        affine=None,
        voxel_size=(1, 1, 1),
        origin=(0, 0, 0),
        nifti_array=False,
        downsample=None,
    ):
        '''
        Initialise from a medical image source.

        Parameters
        ----------
        path : str/array/Nifti1Image
            Source of image data. Can be either:
                (a) A string containing the path to a dicom or nifti file;
                (b) A string containing the path to a numpy file containing a
                    2D or 3D array;
                (c) A 2D or 3D numpy array;
                (d) A nibabel.nifti1.Nifti1Image object.

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
        '''

        self.data = None
        self.title = title
        self.source = path
        self.source_type = None
        self.voxel_size = voxel_size
        self.origin = origin
        self.affine = affine
        self.downsampling = downsample
        self.nifti_array = nifti_array
        self.structs = []

        path = self.source if isinstance(self.source, str) else ''
        skrt.core.Archive.__init__(self, path)

        if load:
            self.load_data()

    def __repr__(self):

        self.load_data()
        out_str = 'Image\n{'
        attrs_to_print = sorted(
            [
                'date',
                'path',
                'subdir',
                'source_type',
                'affine',
                'timestamp',
                'title',
                'downsampling',
            ]
        )
        for attr in attrs_to_print:
            out_str += f'\n  {attr} : {getattr(self, attr)}'
        if len(self.structs):
            out_str += (
                f'\n  structs: [{len(self.structs)} '
                f'* {type(self.structs[0])}]'
            )
        else:
            out_str += '  structs: []'
        out_str += '\n}'
        return out_str

    def get_data(self, standardise=False):
        '''Return image array.'''

        if self.data is None:
            self.load_data()
        if standardise:
            return self.get_standardised_data()
        return self.data

    def get_voxel_size(self):
        '''Return voxel sizes in mm in order (x, y, z).'''

        self.load_data()
        return self.voxel_size

    def get_origin(self):
        '''Return origin position in mm in order (x, y, z).'''

        self.load_data()
        return self.origin

    def get_n_voxels(self):
        '''Return number of voxels in order (x, y, z).'''

        self.load_data()
        return self.n_voxels

    def get_affine(self):
        '''Return affine matrix.'''

        self.load_data()
        return self.affine

    def load_data(self, force=False):
        '''Load pixel array from image source.'''

        if self.data is not None and not force:
            return

        window_width = None
        window_centre = None

        # Load image array from source
        # Numpy array
        if isinstance(self.source, np.ndarray):
            self.data = self.source
            self.source_type = 'nifti array' if self.nifti_array else 'array'

        # Try loading from nifti file
        else:
            if not os.path.exists(self.source):
                raise RuntimeError(
                    f'Image input {self.source} does not exist!')
            if os.path.isfile(self.source):
                self.data, affine = load_nifti(self.source)
                self.source_type = 'nifti'
            if self.data is not None:
                self.affine = affine

        # Try loading from dicom file
        if self.data is None:
            self.data, affine, window_centre, window_width = load_dicom(
                self.source)
            self.source_type = 'dicom'
            if self.data is not None:
                self.affine = affine

        # Try loading from numpy file
        if self.data is None:
            self.data = load_npy(self.source)
            self.source_type = 'nifti array' if self.nifti_array else 'array'

        # If still None, raise exception
        if self.data is None:
            raise RuntimeError(f'{self.source} not a valid image source!')

        # Ensure array is 3D
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]

        # Apply downsampling
        if self.downsampling:
            self.downsample(self.downsampling)
        else:
            self.set_geometry(force=force)

        # Set default grayscale range
        if window_width and window_centre:
            self.default_window = [
                window_centre - window_width / 2,
                window_centre + window_width / 2,
            ]
        else:
            self.default_window = [-300, 200]

        # Set title from filename
        if self.title is None:
            if isinstance(self.source, str) and os.path.exists(self.source):
                self.title = os.path.basename(self.source)

    def get_standardised_data(self):
        '''Return standardised image array.'''

        self.standardise_data()
        return self.sdata

    def standardise_data(self, force=False):
        '''Manipulate data array and affine matrix into a standard
        configuration.'''

        if hasattr(self, 'sdata') and not force:
            return

        data = self.data
        affine = self.affine

        # Adjust dicom
        if self.source_type == 'dicom':

            # Transform array to be in order (row, col, slice) = (x, y, z)
            orient = np.array(self.get_orientation_vector()).reshape(2, 3)
            axes = self.get_axes()
            axes_colrow = self.get_axes(col_first=True)
            transpose = [axes_colrow.index(i) for i in (1, 0, 2)]
            data = np.transpose(self.data, transpose).copy()

            # Adjust affine matrix
            affine = self.affine.copy()
            for i in range(3):

                # Voxel sizes
                if i != axes.index(i):
                    voxel_size = affine[i, axes.index(i)].copy()
                    affine[i, i] = voxel_size
                    affine[i, axes.index(i)] = 0

                # Invert axis direction if negative
                if axes.index(i) < 2 and orient[axes.index(i), i] < 0:
                    affine[i, i] *= -1
                    to_flip = [1, 0, 2][i]
                    data = np.flip(data, axis=to_flip)
                    n_voxels = data.shape[to_flip]
                    affine[i, 3] = affine[i, 3] - (n_voxels - 1) * affine[i, i]

        # Adjust nifti
        elif 'nifti' in self.source_type:

            init_dtype = self.get_data().dtype
            nii = nibabel.as_closest_canonical(
                nibabel.Nifti1Image(self.data.astype(np.float64), self.affine)
            )
            data = nii.get_fdata().transpose(1, 0, 2)[::-1, ::-1, :].astype(
                init_dtype)
            affine = nii.affine

            # Reverse x and y directions
            affine[0, 3] = -(affine[0, 3] + (data.shape[1] - 1) * affine[0, 0])
            affine[1, 3] = -(affine[1, 3] + (data.shape[0] - 1) * affine[1, 1])

        # Assign standardised image array and affine matrix
        self.sdata = data
        self.saffine = affine

        # Get standard voxel sizes and origin
        self.svoxel_size = list(np.diag(self.saffine))[:-1]
        self.sorigin = list(self.saffine[:-1, -1])

    def resample(self, voxel_size):
        '''Resample image to have particular voxel sizes.'''

        # Parse input voxel sizes
        self.load_data()
        parsed = []
        for i, vx in enumerate(voxel_size):
            if vx is None:
                parsed.append(self.voxel_size[i])
            elif self.voxel_size[i] * vx < 0:
                parsed.append(-vx)
            else:
                parsed.append(vx)
        voxel_size = parsed

        # Make interpolant
        old_coords = [
            np.linspace(
                self.origin[i],
                self.origin[i] + (self.n_voxels[i] - 1) * self.voxel_size[i],
                self.n_voxels[i],
            )
            for i in range(3)
        ]
        for i in range(len(old_coords)):
            if old_coords[i][0] > old_coords[i][-1]:
                old_coords[i] = old_coords[i][::-1]
        interpolant = interpolate.RegularGridInterpolator(
            old_coords,
            self.get_data(),
            method='linear',
            bounds_error=False,
            fill_value=self.get_min(),
        )

        # Calculate new number of voxels
        n_voxels = [
            int(np.round(abs(self.get_image_length(i) / voxel_size[i])))
            for i in range(3)
        ]
        voxel_size = [
            np.sign(voxel_size[i]) * self.get_image_length(i) / n_voxels[i]
            for i in range(3)
        ]
        shape = [n_voxels[1], n_voxels[0], n_voxels[2]]
        origin = [
            self.origin[i] - self.voxel_size[i] / 2 + voxel_size[i] / 2
            for i in range(3)
        ]

        # Interpolate to new coordinates
        new_coords = [
            np.linspace(
                origin[i], origin[i] + (n_voxels[i] - 1) * voxel_size[i],
                n_voxels[i]
            )
            for i in range(3)
        ]
        stack = np.vstack(np.meshgrid(*new_coords, indexing='ij'))
        points = stack.reshape(3, -1).T.reshape(*shape, 3)
        self.data = interpolant(points)[::-1, :, :]

        # Reset properties
        self.origin = [
            self.origin[i] - self.voxel_size[i] / 2 + voxel_size[i] / 2
            for i in range(3)
        ]
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.affine = None
        self.set_geometry(force=True)

    def match_voxel_size(self, image, method='self'):
        '''Resample to match z-axis voxel size with that of another Image
        object.

        Available methods:
            - self: match own z voxel size to that of <image>.
            - coarse: resample the image with the smaller z voxels to match
            that of the image with larger z voxels.
            - fine: resample the image with the larger z voxels to match
            that of the image with smaller z voxels.
        '''

        # Determine which image should be resampled
        if method == 'self':
            to_resample = self
            match_to = image
        else:
            own_vz = self.get_voxel_size()[2]
            other_vz = image.get_voxel_size()[2]
            if own_vz == other_vz:
                print('Voxel sizes already match! No resampling applied.')
                return
            if method == 'coarse':
                to_resample = self if own_vz < other_vz else image
                match_to = self if own_vz > other_vz else image
            elif method == 'fine':
                to_resample = self if own_vz > other_vz else image
                match_to = self if own_vz < other_vz else image
            else:
                raise RuntimeError(f'Unrecognised resampling option: {method}')

        # Perform resampling
        voxel_size = [None, None, match_to.get_voxel_size()[2]]
        init_vz = to_resample.voxel_size[2]
        init_nz = int(to_resample.n_voxels[2])
        to_resample.resample(voxel_size)
        print(
            f'Resampled z axis from {init_nz} x {init_vz:.3f} mm -> '
            f'{int(to_resample.n_voxels[2])} x {to_resample.voxel_size[2]:.3f}'
            'mm'
        )

    def get_min(self):
        '''Get minimum value of data array.'''

        if not hasattr(self, 'min_val'):
            self.load_data()
            self.min_val = self.data.min()
        return self.min_val

    def get_orientation_codes(self, affine=None, source_type=None):
        '''Get image orientation codes in order (row, column, slice) for
        dicom or (column, row, slice) for nifti.

        L = Left (x axis)
        R = Right (x axis)
        P = Posterior (y axis)
        A = Anterior (y axis)
        I = Inferior (z axis)
        S = Superior (z axis)
        '''

        if affine is None:
            self.load_data()
            affine = self.affine
        codes = list(nibabel.aff2axcodes(affine))

        if source_type is None:
            source_type = self.source_type

        # Reverse codes for row and column of a dicom
        pairs = [('L', 'R'), ('P', 'A'), ('I', 'S')]
        if 'nifti' not in source_type:
            for i in range(2):
                switched = False
                for p in pairs:
                    for j in range(2):
                        if codes[i] == p[j] and not switched:
                            codes[i] = p[1 - j]
                            switched = True

        return codes

    def get_orientation_vector(self, affine=None, source_type=None):
        '''Get image orientation as a row and column vector.'''

        if source_type is None:
            source_type = self.source_type
        if affine is None:
            affine = self.affine

        codes = self.get_orientation_codes(affine, source_type)
        if 'nifti' in source_type:
            codes = [codes[1], codes[0], codes[2]]
        vecs = {
            'L': [1, 0, 0],
            'R': [-1, 0, 0],
            'P': [0, 1, 0],
            'A': [0, -1, 0],
            'S': [0, 0, 1],
            'I': [0, 0, -1],
        }
        return vecs[codes[0]] + vecs[codes[1]]

    def get_axes(self, col_first=False):
        '''Return list of axis numbers in order (column, row, slice) if
        col_first is True, otherwise in order (row, column, slice). The axis
        numbers 0, 1, and 2 correspond to x, y, and z, respectively.'''

        orient = np.array(self.get_orientation_vector()).reshape(2, 3)
        axes = [sum([abs(int(orient[i, j] * j)) for j in range(3)])
                for i in range(2)]
        axes.append(3 - sum(axes))
        if not col_first:
            return axes
        else:
            return [axes[1], axes[0], axes[2]]

    def get_machine(self, stations=_default_stations):

        machine = None
        if self.files:
            ds = pydicom.read_file(self.files[0].path, force=True)
            try:
                station = ds.StationName
            except BaseException:
                station = None
            if station in stations:
                machine = stations[station]
        return machine

    def set_geometry(self, force=False):
        '''Set geometric properties.'''

        # Set affine matrix, voxel sizes, and origin
        if self.affine is None:
            self.affine = np.array(
                [
                    [self.voxel_size[0], 0, 0, self.origin[0]],
                    [0, self.voxel_size[1], 0, self.origin[1]],
                    [0, 0, self.voxel_size[2], self.origin[2]],
                    [0, 0, 0, 1],
                ]
            )
            if 'nifti' in self.source_type:
                self.affine[0, :] = -self.affine[0, :]
                self.affine[1, 3] = -(
                    self.affine[1, 3] + (self.data.shape[1] - 1)
                    * self.voxel_size[1]
                )
        else:
            self.voxel_size = list(np.diag(self.affine))[:-1]
            self.origin = list(self.affine[:-1, -1])

        # Set number of voxels
        self.n_voxels = [self.data.shape[1], self.data.shape[0],
                         self.data.shape[2]]

        # Set axis limits for standardised plotting
        self.standardise_data(force=force)
        self.lims = [
            (
                self.sorigin[i],
                self.sorigin[i] + (self.n_voxels[i] - 1) * self.svoxel_size[i],
            )
            for i in range(3)
        ]
        self.image_extent = [
            (
                self.lims[i][0] - self.svoxel_size[i] / 2,
                self.lims[i][1] + self.svoxel_size[i] / 2,
            )
            for i in range(3)
        ]
        self.plot_extent = {
            view: self.image_extent[x_ax] + self.image_extent[y_ax][::-1]
            for view, (x_ax, y_ax) in _plot_axes.items()
        }

    def get_idx(self, view, sl=None, idx=None, pos=None):
        '''Get an array index from either a slice number, index, or
        position.'''

        if sl is not None:
            idx = self.slice_to_idx(sl, _slice_axes[view])
        elif idx is None:
            if pos is not None:
                idx = self.pos_to_idx(pos, _slice_axes[view])
            else:
                centre_pos = self.get_image_centre()[_slice_axes[view]]
                idx = self.pos_to_idx(centre_pos, _slice_axes[view])
        return idx

    def get_slice(
        self, view='x-y', sl=None, idx=None, pos=None, flatten=False, **kwargs
    ):
        '''Get a slice of the data in the correct orientation for plotting.'''

        # Get image slice
        idx = self.get_idx(view, sl, idx, pos)
        transposes = {'x-y': (0, 1, 2), 'y-z': (0, 2, 1), 'x-z': (1, 2, 0)}
        transpose = transposes[view]
        list(_plot_axes[view]) + [_slice_axes[view]]
        data = np.transpose(self.get_standardised_data(), transpose)
        if flatten:
            return np.sum(data, axis=2)
        else:
            return data[:, :, idx]

    def set_ax(
        self,
        view=None,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        zoom=None,
        colorbar=False,
        **kwargs,
    ):
        '''Set up axes for plotting this image, either from a given exes or
        gridspec, or by creating new axes.'''

        # Set up figure/axes
        if ax is None and gs is not None:
            ax = plt.gcf().add_subplot(gs)
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            if figsize is None:
                figsize = _default_figsize
            if skrt.core.is_list(figsize):
                fig_tuple = figsize
            else:
                figsize = to_inches(figsize)
                aspect = self.get_plot_aspect_ratio(view, zoom, colorbar,
                                                    figsize)
                fig_tuple = (figsize * aspect, figsize)
            self.fig = plt.figure(figsize=fig_tuple)
            self.ax = self.fig.add_subplot()

    def add_structs(self, structure_set):
        '''Add a structure set.'''

        self.structs.append(structure_set)

    def clear_structs(self):
        '''Clear all structures.'''

        self.structs = []

    def get_mpl_kwargs(self, view, mpl_kwargs=None, scale_in_mm=True):
        '''Get matplotlib kwargs dict including defaults.'''

        if mpl_kwargs is None:
            mpl_kwargs = {}

        # Set colormap
        if 'cmap' not in mpl_kwargs:
            mpl_kwargs['cmap'] = 'gray'

        # Set colour range
        for i, name in enumerate(['vmin', 'vmax']):
            if name not in mpl_kwargs:
                mpl_kwargs[name] = self.default_window[i]

        # Set image extent and aspect ratio
        extent = self.plot_extent[view]
        mpl_kwargs['aspect'] = 1
        x_ax, y_ax = _plot_axes[view]
        if not scale_in_mm:
            extent = [
                self.pos_to_slice(extent[0], x_ax, False),
                self.pos_to_slice(extent[1], x_ax, False),
                self.pos_to_slice(extent[2], y_ax, False),
                self.pos_to_slice(extent[3], y_ax, False),
            ]
            mpl_kwargs['aspect'] = abs(self.voxel_size[y_ax]
                                       / self.voxel_size[x_ax])
        mpl_kwargs['extent'] = extent

        return mpl_kwargs

    def view(self, **kwargs):
        '''View self with QuickViewer.'''

        from skrt.viewer import QuickViewer

        QuickViewer(self.source, **kwargs)

    def plot(
        self,
        view='x-y',
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
        hu=None,
        mpl_kwargs=None,
        show=True,
        colorbar=False,
        colorbar_label='HU',
        no_title=False,
        no_ylabel=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        structure_set=None,
        struct_plot_type='contour',
        struct_legend=False,
        struct_kwargs={},
        centre_on_struct=None,
        legend_loc='lower left',
        flatten=False,
    ):
        '''Plot a 2D slice of the image.

        Parameters
        ----------
        view : str, default='x-y'
            Orientation in which to plot the image. Can be any of 'x-y',
            'y-z', and 'x-z'.

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

        colorbar : bool, default=True
            If True, a colorbar will be drawn alongside the plot.

        colorbar_label : str, default='HU'
            Label for the colorbar, if drawn.

        hu : list, default=None
            Two-item list containing min and max HU for plotting. Supercedes
            'vmin' and 'vmax' in <mpl_kwargs>.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        show : bool, default=True
            If True, the plotted figure will be shown via
            matplotlib.pyplot.show().

        no_title : bool, default=False
            If True, the plot will not be given a title.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        annotate_slice : bool/str, default=False
            Color for annotation of slice number. If False, no annotation will
            be added. If True, the default color (white) will be used.

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

        structure_set : int/str, default=None
            Option for which structure set should be plotted (if the Image
            owns any structure sets). Can be:
                - None: no structures will be plotted.
                - The index in self.structs of the structure set (e.g. to plot
                the newest structure set, use structure_set=-1)
                - 'all': all structure sets will be plotted.

        struct_plot_type : str, default='contour'
            Structure plotting type (see ROI.plot() for options).

        struct_legend : bool, default=False
            If True, a legend will be drawn containing structure names.

        struct_kwargs : dict, default=None
            Extra arguments to provide to structure plotting.

        centre_on_struct : str, default=None
            Name of struct on which to centre, if no idx/sl/pos is given.
            If <zoom> is given but no <zoom_centre>, the zoom will also centre
            on this struct.

        legend_loc : str, default='lower left'
            Legend location for structure legend.
        '''

        self.load_data()

        # Set up axes
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        self.ax.clear()
        self.load_data()

        # Get list of structures to plot
        to_plot = []
        if structure_set is not None:

            # Get list of structure sets to plot
            if isinstance(structure_set, int):
                try:
                    to_plot = [self.structs[structure_set]]
                except IndexError:
                    print(
                        f'Warning: structure set {structure_set} not found! '
                        f'Image only has {len(self.structs)} structure sets.'
                    )
            elif structure_set == 'all':
                to_plot = self.structs
            elif skrt.core.is_list(structure_set):
                to_plot = [self.structs[i] for i in structure_set]
            else:
                print(
                    f'Warning: structure set option {structure_set} not '
                    'recognised! Must be an int, None, or "all".'
                )

            # If centering on structure, find central slice
            if all([i is None for i in [idx, pos, sl]]) and centre_on_struct:
                central_struct = to_plot[0].get_struct(centre_on_struct)
                idx = central_struct.get_mid_idx(view)
                if zoom and zoom_centre is None:
                    zoom_centre = central_struct.get_zoom_centre(view)

        # Get image slice
        idx = self.get_idx(view, sl, idx, pos)
        image_slice = self.get_slice(view, sl=sl, idx=idx, pos=pos,
                                     flatten=flatten)

        # Apply HU window if given
        if mpl_kwargs is None:
            mpl_kwargs = {}
        if hu is not None:
            mpl_kwargs['vmin'] = hu[0]
            mpl_kwargs['vmax'] = hu[1]

        # Plot the slice
        mesh = self.ax.imshow(
            image_slice, **self.get_mpl_kwargs(view, mpl_kwargs, scale_in_mm)
        )

        # Plot structure sets
        if len(to_plot):
            handles = []
            for ss in to_plot:
                for s in ss.get_structs():
                    name = s.name
                    if len(to_plot) > 1:
                        name += f' ({ss.name})'
                    s.plot(
                        view,
                        idx=idx,
                        ax=self.ax,
                        plot_type=struct_plot_type,
                        include_image=False,
                        **struct_kwargs,
                    )
                    if s.on_slice(view, idx=idx) and struct_legend:
                        handles.append(mpatches.Patch(color=s.color,
                                                      label=name))

            # Draw structure legend
            if struct_legend and len(handles):
                self.ax.legend(
                    handles=handles, loc=legend_loc, facecolor='white',
                    framealpha=1
                )

        # Label axes
        self.label_ax(
            view,
            idx,
            scale_in_mm,
            no_title,
            no_ylabel,
            annotate_slice,
            major_ticks,
            minor_ticks,
            ticks_all_sides,
        )
        self.zoom_ax(view, zoom, zoom_centre)

        # Add colorbar
        if colorbar and mpl_kwargs.get('alpha', 1) > 0:
            clb = self.fig.colorbar(mesh, ax=self.ax, label=colorbar_label)
            clb.solids.set_edgecolor('face')

        # Display image
        if show:
            plt.tight_layout()
            plt.show()

        # Save to file
        if save_as:
            self.fig.savefig(save_as)
            plt.close()

    def label_ax(
        self,
        view,
        idx,
        scale_in_mm=True,
        no_title=False,
        no_ylabel=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        **kwargs,
    ):

        x_ax, y_ax = _plot_axes[view]

        # Set title
        if self.title and not no_title:
            self.ax.set_title(self.title, pad=8)

        # Set axis labels
        units = ' (mm)' if scale_in_mm else ''
        self.ax.set_xlabel(_axes[x_ax] + units, labelpad=0)
        if not no_ylabel:
            self.ax.set_ylabel(_axes[y_ax] + units)
        else:
            self.ax.set_yticks([])

        # Annotate with slice position
        if annotate_slice:
            z_ax = _axes[_slice_axes[view]]
            if scale_in_mm:
                z_str = '{} = {:.1f} mm'.format(
                    z_ax, self.idx_to_pos(idx, z_ax))
            else:
                z_str = '{} = {}'.format(z_ax, self.idx_to_slice(idx, z_ax))
            if matplotlib.colors.is_color_like(annotate_slice):
                color = annotate_slice
            else:
                color = 'white'
            self.ax.annotate(
                z_str,
                xy=(0.05, 0.93),
                xycoords='axes fraction',
                color=color,
                fontsize='large',
            )

        # Adjust tick marks
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
                    which='minor', bottom=True, top=True, left=True, right=True
                )

    def zoom_ax(self, view, zoom=None, zoom_centre=None):
        '''Zoom in on axes if needed.'''

        if not zoom or isinstance(zoom, str):
            return
        zoom = skrt.core.to_three(zoom)
        x_ax, y_ax = _plot_axes[view]
        if zoom_centre is None:
            im_centre = self.get_image_centre()
            mid_x = im_centre[x_ax]
            mid_y = im_centre[y_ax]
        else:
            mid_x, mid_y = zoom_centre[x_ax], zoom_centre[y_ax]

        # Calculate new axis limits
        init_xlim = self.plot_extent[view][:2]
        init_ylim = self.plot_extent[view][2:]
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
        '''Convert an array index to a position in mm along a given axis.'''

        self.load_data()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if standardise:
            origin = self.sorigin
            voxel_size = self.svoxel_size
        else:
            origin = self.origin
            voxel_size = self.voxel_size
        return origin[i_ax] + idx * voxel_size[i_ax]

    def pos_to_idx(self, pos, ax, return_int=True, standardise=True):
        '''Convert a position in mm to an array index along a given axis.'''

        self.load_data()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if standardise:
            origin = self.sorigin
            voxel_size = self.svoxel_size
        else:
            origin = self.origin
            voxel_size = self.voxel_size
        idx = (pos - origin[i_ax]) / voxel_size[i_ax]
        if return_int:
            return round(idx)
        else:
            return idx

    def idx_to_slice(self, idx, ax):
        '''Convert an array index to a slice number along a given axis.'''

        self.load_data()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if i_ax == 2:
            return self.n_voxels[i_ax] - idx
        else:
            return idx + 1

    def slice_to_idx(self, sl, ax):
        '''Convert a slice number to an array index along a given axis.'''

        self.load_data()
        i_ax = _axes.index(ax) if ax in _axes else ax
        if i_ax == 2:
            return self.n_voxels[i_ax] - sl
        else:
            return sl - 1

    def pos_to_slice(self, pos, ax, return_int=True, standardise=True):
        '''Convert a position in mm to a slice number along a given axis.'''

        sl = self.idx_to_slice(
            self.pos_to_idx(pos, ax, return_int, standardise), ax)
        if return_int:
            return round(sl)
        else:
            return sl

    def slice_to_pos(self, sl, ax, standardise=True):
        '''Convert a slice number to a position in mm along a given axis.'''

        return self.idx_to_pos(self.slice_to_idx(sl, ax), ax, standardise)

    def get_image_centre(self):
        '''Get position in mm of the centre of the image.'''

        self.load_data()
        return [np.mean(self.lims[i]) for i in range(3)]

    def get_range(self, ax='z'):
        '''Get range of the scan in mm along a given axis.'''

        i_ax = _axes.index(ax) if ax in _axes else ax
        origin = self.get_origin()[i_ax]
        return [origin, origin + (self.n_voxels[i_ax] - 1)
                * self.voxel_size[i_ax]]

    def get_image_length(self, ax='z'):
        '''Get total length of image.'''

        i_ax = _axes.index(ax) if ax in _axes else ax
        return abs(self.n_voxels[i_ax] * self.voxel_size[i_ax])

    def get_voxel_coords(self):
        '''Get arrays of voxel coordinates in each direction.'''

        return

    def get_plot_aspect_ratio(
        self, view, zoom=None, n_colorbars=0, figsize=_default_figsize
    ):
        '''Estimate the ideal width/height ratio for a plot of this image
        in a given orientation.

        view : str
            Orienation ('x-y', 'y-z', or 'x-z')

        zoom : float/list, default=None
            Zoom factors; either a single value for all axes, or three values
            in order (x, y, z).

        n_colorbars : int, default=0
            Number of colorbars to make space for.
        '''

        # Get length of the image in the plot axis directions
        x_ax, y_ax = _plot_axes[view]
        x_len = abs(self.lims[x_ax][1] - self.lims[x_ax][0])
        y_len = abs(self.lims[y_ax][1] - self.lims[y_ax][0])

        # Add padding for axis labels and title
        font = mpl.rcParams['font.size'] / 72
        y_pad = 2 * font
        if self.title:
            y_pad += 1.5 * font
        max_y_digits = np.floor(np.log10(max([abs(lim) for lim in
                                              self.lims[y_ax]])))
        minus_sign = any([lim < 0 for lim in self.lims[y_ax]])
        x_pad = (0.7 * max_y_digits + 1.2 * minus_sign + 1) * font

        # Account for zoom
        if zoom:
            zoom = skrt.core.to_three(zoom)
            x_len /= zoom[x_ax]
            y_len /= zoom[y_ax]

        # Add padding for colorbar(s)
        colorbar_frac = 0.4 * 5 / figsize
        x_len *= 1 + (n_colorbars * colorbar_frac)

        # Return estimated width ratio
        total_y = figsize + y_pad
        total_x = figsize * x_len / y_len + x_pad
        return total_x / total_y

    def downsample(self, downsampling):
        '''Apply downsampling to the image array. Can be either a single
        value (to downsampling equally in all directions) or a list of 3
        values.'''

        # Get downsampling in each direction
        if skrt.core.is_list(downsampling):
            if len(downsampling) != 3:
                raise TypeError('<downsample> must contain 3 elements!')
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
        '''Get image array and affine matrix in canonical nifti
        configuration.'''

        # Convert dicom-style array to nifti
        if 'nifti' not in self.source_type:
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
            nii = nibabel.as_closest_canonical(
                nibabel.Nifti1Image(data, affine))
            return nii.get_fdata(), nii.affine
        else:
            return data, affine

    def get_dicom_array_and_affine(self, standardise=False):
        '''Get image array and affine matrix in dicom configuration.'''

        # Return standardised dicom array
        if standardise:
            return self.get_standardised_data(), self.saffine

        # Convert nifti-style array to dicom
        if 'nifti' in self.source_type:
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
        header_source=None,
        patient_id=None,
        modality=None,
        root_uid=None,
    ):
        '''Write image data to a file. The filetype will automatically be
        set based on the extension of <outname>:

            (a) *.nii or *.nii.gz: Will write to a nifti file with
            canonical nifti array and affine matrix.

            (b) *.npy: Will write the dicom-style numpy array to a binary filem
            unless <nifti_array> is True, in which case the canonical
            nifti-style array will be written. If <write_geometry> is True,
            a text file containing the voxel sizes and origin position will
            also be written in the same directory.

            (c) *.dcm: Will write to dicom file(s) (1 file per x-y slice) in
            the directory of the filename given, named by slice number.

            (d) No extension: Will create a directory at <outname> and write
            to dicom file(s) in that directory (1 file per x-y slice), named
            by slice number.

        If (c) or (d) (i.e. writing to dicom), the header data will be set in
        one of three ways:
            - If the input source was not a dicom, <dicom_for_header> is None,
            a brand new dicom with freshly generated UIDs will be created.
            - If <dicom_for_header> is set to the path to a dicom file, that
            dicom file will be used as the header.
            - Otherwise, if the input source was a dicom or directory
            containing dicoms, the header information will be taken from the
            input dicom file.
        '''

        outname = os.path.expanduser(outname)
        self.load_data()

        # Write to nifti file
        if outname.endswith('.nii') or outname.endswith('.nii.gz'):
            data, affine = self.get_nifti_array_and_affine(standardise)
            if data.dtype == 'bool':
                data = data.copy().astype(int)
            write_nifti(outname, data, affine)
            print('Wrote to NIfTI file:', outname)

        # Write to numpy file
        elif outname.endswith('.npy'):
            if nifti_array:
                data, affine = self.get_nifti_array_and_affine(standardise)
            else:
                data, affine = self.get_dicom_array_and_affine(standardise)
            if not write_geometry:
                affine = None
            write_npy(outname, data, affine)
            print('Wrote to numpy file:', outname)

        # Write to dicom
        else:

            # Get name of dicom directory
            if outname.endswith('.dcm'):
                outdir = os.path.dirname(outname)
            else:
                outdir = outname

            # Get header source
            if header_source is None and self.source_type == 'dicom':
                header_source = self.source
            data, affine = self.get_dicom_array_and_affine(standardise)
            orientation = self.get_orientation_vector(affine, 'dicom')
            write_dicom(
                outdir,
                data,
                affine,
                header_source,
                orientation,
                patient_id,
                modality,
                root_uid,
            )
            print('Wrote dicom file(s) to directory:', outdir)

    def get_coords(self):
        '''Get grids of x, y, and z coordinates for each voxel in the image.'''

        if not hasattr(self, 'coords'):

            # Make coordinates
            coords_1d = [
                np.arange(
                    self.origin[i],
                    self.origin[i] + self.n_voxels[i] * self.voxel_size[i],
                    self.voxel_size[i],
                )
                for i in range(3)
            ]
            X, Y, Z = np.meshgrid(*coords_1d)

            # Set coords
            self.coords = (X, Y, Z)

        # Apply transformations
        return self.coords


class ImageComparison:
    '''Plot comparisons of two images and calculate comparison metrics.'''

    def __init__(self, im1, im2, **kwargs):

        # Load images
        self.ims = []
        for im in [im1, im2]:
            if issubclass(type(im), Image):
                self.ims.append(im)
            else:
                self.ims.append(Image(im, **kwargs))

    def plot_chequerboard(
        self,
        view='x-y',
        invert=False,
        n_splits=2,
        mpl_kwargs=None,
        save_as=None,
        **kwargs,
    ):

        # Get indices of images to plot
        i1 = int(invert)
        i2 = 1 - i1

        # Plot background image
        self.ims[i1].plot(view=view, mpl_kwargs=mpl_kwargs, show=False,
                          **kwargs)

        # Create mask for second image
        im2_slice = self.ims[i2].get_slice(view=view, **kwargs)
        nx = int(np.ceil(im2_slice.shape[0] / n_splits))
        ny = int(np.ceil(im2_slice.shape[1] / n_splits))
        mask = np.kron(
            [[1, 0] * n_splits, [0, 1] * n_splits] * n_splits,
            np.ones((nx, ny))
        )
        mask = mask[: im2_slice.shape[0], : im2_slice.shape[1]]

        # Plot second image
        self.ims[i1].ax.imshow(
            np.ma.masked_where(mask < 0.5, im2_slice),
            **self.ims[i2].get_mpl_kwargs(view, mpl_kwargs),
        )

        if save_as:
            self.ims[0].fig.savefig(save_as)
            plt.close()

    def plot_overlay(
        self,
        view='x-y',
        invert=False,
        opacity=0.5,
        mpl_kwargs=None,
        save_as=None,
        show=True,
        **kwargs,
    ):

        i1 = int(invert)
        i2 = 1 - i1
        cmaps = ['Reds', 'Blues']
        alphas = [1, opacity]

        # Set axes
        self.ims[0].set_ax(view=view, **kwargs)
        ax = self.ims[0].ax
        ax.set_facecolor('w')

        # Plot images
        for n, i in enumerate([i1, i2]):
            im_slice = self.ims[i].get_slice(view=view, **kwargs)
            mpl_kwargs = self.ims[i].get_mpl_kwargs(view, mpl_kwargs)
            mpl_kwargs['cmap'] = cmaps[n]
            mpl_kwargs['alpha'] = alphas[n]
            ax.imshow(im_slice, **mpl_kwargs)

        if save_as:
            self.ims[0].fig.savefig(save_as)
            plt.close()


def load_nifti(path):
    '''Load an image from a nifti file.'''

    try:
        nii = nibabel.load(path)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine

    except FileNotFoundError:
        print(f'Warning: file {path} not found! Could not load nifti.')
        return None, None

    except nibabel.filebasedimages.ImageFileError:
        return None, None


def load_dicom(path):
    '''Load a dicom image from one or more dicom files.'''

    # Try loading single dicom file
    paths = []
    if os.path.isfile(path):
        ds = pydicom.read_file(path, force=True)

        # Discard if not a valid dicom file
        if not hasattr(ds, 'SOPClassUID'):
            return None, None, None, None

        # Assign TransferSyntaxUID if missing
        if not hasattr(ds, 'TransferSyntaxUID'):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        if ds.get('ImagesInAcquisition', None) == 1:
            paths = [path]

    # Case where there are multiple dicom files for this image
    if not paths:
        if os.path.isdir(path):
            dirname = path
        else:
            dirname = os.path.dirname(path)
        paths = sorted([os.path.join(dirname, p) for p in os.listdir(dirname)])

        # Ensure user-specified file is loaded first
        if path in paths:
            paths.insert(0, paths.pop(paths.index(path)))

    # Load image arrays from all files
    study_uid = None
    series_num = None
    modality = None
    orientation = None
    slice_thickness = None
    pixel_size = None
    rescale_slope = None
    rescale_intercept = None
    window_centre = None
    window_width = None
    data_slices = {}
    image_position = {}
    for dcm in paths:
        try:

            # Load file and check it matches the others
            ds = pydicom.read_file(dcm, force=True)
            if study_uid is None:
                study_uid = ds.StudyInstanceUID
            if series_num is None:
                series_num = ds.SeriesNumber
            if modality is None:
                modality = ds.Modality
            if orientation is None:
                orientation = ds.ImageOrientationPatient
                orient = np.array(orientation).reshape(2, 3)
                axes = [
                    sum([abs(int(orient[i, j] * j)) for j in range(3)])
                    for i in range(2)
                ]
                axes.append(3 - sum(axes))
            if (
                ds.StudyInstanceUID != study_uid
                or ds.SeriesNumber != series_num
                or ds.Modality != modality
                or ds.ImageOrientationPatient != orientation
            ):
                continue
            if not hasattr(ds, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = \
                        pydicom.uid.ImplicitVRLittleEndian

            # Get data
            pos = getattr(ds, 'ImagePositionPatient', [0, 0, 0])
            z = pos[axes[2]]
            data_slices[z] = ds.pixel_array
            image_position[z] = pos

            # Get voxel spacings
            if pixel_size is None:
                for attr in ['PixelSpacing', 'ImagerPixelSpacing']:
                    pixel_size = getattr(ds, attr, None)
                    if pixel_size:
                        break
            if slice_thickness is None:
                slice_thickness = getattr(ds, 'SliceThickness', None)

            # Get rescale settings
            if rescale_slope is None:
                rescale_slope = getattr(ds, 'RescaleSlope', None)
            if rescale_intercept is None:
                rescale_intercept = getattr(ds, 'RescaleIntercept', None)

            # Get HU window defaults
            if window_centre is None:
                window_centre = getattr(ds, 'WindowCenter', None)
                if isinstance(window_centre, pydicom.multival.MultiValue):
                    window_centre = window_centre[0]
            if window_width is None:
                window_width = getattr(ds, 'WindowWidth', None)
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]

        # Skip any invalid dicom files
        except pydicom.errors.InvalidDicomError:
            continue

    # Case where no data was found
    if not data_slices:
        print(f'Warning: no valid dicom files found in {path}')
        return None, None, None, None

    # Case with single image array
    if len(data_slices) == 1:
        data = list(data_slices.values())[0]

    # Combine arrays
    else:

        # Sort by slice position
        sorted_slices = sorted(list(data_slices.keys()))
        sorted_data = [data_slices[z] for z in sorted_slices]
        data = np.stack(sorted_data, axis=-1)

        # Recalculate slice thickness from spacing
        slice_thickness = (sorted_slices[-1] - sorted_slices[0]) / (
            len(sorted_slices) - 1
        )

    # Apply rescaling
    if rescale_slope:
        data = data * rescale_slope
    if rescale_intercept:
        data = data + rescale_intercept

    # Make affine matrix
    zmin = sorted_slices[0]
    zmax = sorted_slices[-1]
    n = len(sorted_slices)
    affine = np.array(
        [
            [
                orient[0, 0] * pixel_size[0],
                orient[1, 0] * pixel_size[1],
                (image_position[zmax][0] - image_position[zmin][0]) / (n - 1),
                image_position[zmin][0],
            ],
            [
                orient[0, 1] * pixel_size[0],
                orient[1, 1] * pixel_size[1],
                (image_position[zmax][1] - image_position[zmin][1]) / (n - 1),
                image_position[zmin][1],
            ],
            [
                orient[0, 2] * pixel_size[0],
                orient[1, 2] * pixel_size[1],
                (image_position[zmax][2] - image_position[zmin][2]) / (n - 1),
                image_position[zmin][2],
            ],
            [0, 0, 0, 1],
        ]
    )

    return data, affine, window_centre, window_width


def load_npy(path):
    '''Load a numpy array from a .npy file.'''

    try:
        data = np.load(path)
        return data

    except (IOError, ValueError):
        return


def downsample(data, dx=None, dy=None, dz=None):
    '''Downsample an array by the factors specified in <dx>, <dy>, and <dz>.'''

    if dx is None:
        dx = 1
    if dy is None:
        dy = 1
    if dx is None:
        dz = 1

    return data[:: round(dy), :: round(dx), :: round(dz)]


def to_inches(size):
    '''Convert a size string to a size in inches. If a float is given, it will
    be returned. If a string is given, the last two characters will be used to
    determine the units:
        - 'in': inches
        - 'cm': cm
        - 'mm': mm
        - 'px': pixels
    '''

    if not isinstance(size, str):
        return size

    val = float(size[:-2])
    units = size[-2:]
    inches_per_cm = 0.394
    if units == 'in':
        return val
    elif units == 'cm':
        return inches_per_cm * val
    elif units == 'mm':
        return inches_per_cm * val / 10
    elif units == 'px':
        return val / mpl.rcParams['figure.dpi']


def write_nifti(outname, data, affine):
    '''Create a nifti file at <outname> containing <data> and <affine>.'''

    nii = nibabel.Nifti1Image(data, affine)
    nii.to_filename(outname)


def write_npy(outname, data, affine=None):
    '''Create numpy file containing data. If <affine> is not None, voxel
    sizes and origin will be written to a text file.'''

    np.save(outname, data)
    if affine is not None:
        voxel_size = np.diag(affine)[:-1]
        origin = affine[:-1, -1]
        geom_file = outname.replace('.npy', '.txt')
        with open(geom_file, 'w') as f:
            f.write('voxel_size')
            for vx in voxel_size:
                f.write(' ' + str(vx))
            f.write('\norigin')
            for p in origin:
                f.write(' ' + str(p))
            f.write('\n')


def write_dicom(
    outdir,
    data,
    affine,
    header_source=None,
    orientation=None,
    patient_id=None,
    modality=None,
    root_uid=None,
):
    '''Write image data to dicom file(s) inside <outdir>. <header_source> can
    be:
        (a) A path to a dicom file, which will be used as the header;
        (b) A path to a directory containing dicom files; the first file
        alphabetically will be used as the header;
        (c) A pydicom FileDataset object;
        (d) None, in which case a brand new dicom file with new UIDs will be
        created.
    '''

    # Create directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        # Clear out any dicoms in the directory
        dcms = glob.glob(f'{outdir}/*.dcm')
        for dcm in dcms:
            os.remove(dcm)

    # Try loading from header
    ds = None
    if header_source:
        if isinstance(header_source, FileDataset):
            ds = header_source
        else:
            dcm_path = None
            if os.path.isfile(header_source):
                dcm_path = header_source
            elif os.path.isdir(header_source):
                dcms = glob.glob(f'{header_source}/*.dcm')
                if dcms:
                    dcm_path = dcms[0]
            if dcm_path:
                try:
                    ds = pydicom.read_file(dcm_path, force=True)
                except pydicom.errors.InvalidDicomError:
                    pass

    # Make fresh header if needed
    fresh_header = ds is None
    if fresh_header:
        ds = create_dicom(orientation, patient_id, modality, root_uid)

    # Set voxel sizes etc from affine matrix
    if orientation is None:
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    else:
        ds.ImageOrientationPatient = orientation
    ds.PixelSpacing = [affine[0, 0], affine[1, 1]]
    ds.SliceThickness = affine[2, 2]
    ds.ImagePositionPatient = list(affine[:-1, 3])
    ds.Columns = data.shape[1]
    ds.Rows = data.shape[0]

    # Rescale data
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    if fresh_header:
        intercept = np.min(data)
        ds.RescaleIntercept = intercept

    # Save each x-y slice to a dicom file
    for i in range(data.shape[2]):
        sl = data.shape[2] - i
        pos = affine[2, 3] + i * affine[2, 2]
        xy_slice = data[:, :, i].copy()
        xy_slice = (xy_slice - intercept) / slope
        xy_slice = xy_slice.astype(np.uint16)
        ds.PixelData = xy_slice.tobytes()
        ds.SliceLocation = pos
        ds.ImagePositionPatient[2] = pos
        outname = f'{outdir}/{sl}.dcm'
        ds.save_as(outname)


def create_dicom(
    orientation=None,
    patient_id=None,
    modality=None,
    root_uid=None
):
    '''Create a fresh dicom dataset. Taken from https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_write_dicom.html#sphx-glr-auto-examples-input-output-plot-write-dicom-py.'''

    # Create some temporary filenames
    suffix = '.dcm'
    filename = tempfile.NamedTemporaryFile(suffix=suffix).name

    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Create the FileDataset instance
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b'\x00' * 128)

    # Add data elements
    ds.PatientID = patient_id if patient_id is not None else '123456'
    ds.PatientName = ds.PatientID
    ds.Modality = modality if modality is not None else 'CT'
    ds.SeriesInstanceUID = get_new_uid(root_uid)
    ds.StudyInstanceUID = get_new_uid(root_uid)
    ds.SeriesNumber = '123456'
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelRepresentation = 0

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    # Data storage
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15

    return ds


def get_new_uid(root=None):
    '''Generate a globally unique identifier (GUID). Credit: Karl Harrison.

    <root> should uniquely identify the group generating the GUID. A unique
    root identifier can be obtained free of charge from Medical Connections:
        https://www.medicalconnections.co.uk/FreeUID/
    '''

    if root is None:
        print(
            'Warning: using generic root UID 1.2.3.4. You should use a root '
            'UID unique to your institution. A unique root ID can be '
            'obtained free of charge from: '
            'https://www.medicalconnections.co.uk/FreeUID/'
        )
        root = '1.2.3.4'

    id1 = uuid.uuid1()
    id2 = uuid.uuid4().int & (1 << 24) - 1
    date = time.strftime('%Y%m%d')
    new_id = f'{root}.{date}.{id1.time_low}.{id2}'

    if not len(new_id) % 2:
        new_id = '.'.join([new_id, str(np.random.randint(1, 9))])
    else:
        new_id = '.'.join([new_id, str(np.random.randint(10, 99))])
    return new_id
