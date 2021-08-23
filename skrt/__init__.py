'''Prototype classes for core data functionality.'''

from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from scipy.ndimage import morphology
from scipy import interpolate
from shapely import geometry
import copy
import datetime
import fnmatch
import functools
import glob
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pandas as pd
import pydicom
import re
import shutil
import skimage.measure
import tempfile
import time
import uuid


default_stations = {'0210167': 'LA3', '0210292': 'LA4'}

default_opts = {}
default_opts['print_depth'] = 0


class Defaults:
    '''
    Singleton class for storing default values of parameters
    that may be used in object initialisation.

    Implementation of the singleton design pattern is based on:
    https://python-3-patterns-idioms-test.readthedocs.io
           /en/latest/Singleton.html
    '''

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __Defaults:

        # Define instance attributes based on opts dictionary
        def __init__(self, opts={}):
            for key, value in opts.items():
                setattr(self, key, value)

        # Allow for printing instance attributes
        def __repr__(self):
            out_list = []
            for key, value in sorted(self.__dict__.items()):
                out_list.append(f'{key}: {value}')
            out_string = '\n'.join(out_list)
            return out_string

    def __init__(self, opts={}, reset=False):
        '''
        Constructor of Defaults singleton class.

        Parameters
        ----------
        opts : dict, default={}
            Dictionary of attribute-value pairs.

        reset : bool, default=False
            If True, delete all pre-existing instance attributes before
            adding attributes and values from opts dictionary.
            If False, don't delete pre-existing instance attributes,
            but add to them, or modify values, from opts dictionary.
        '''

        if not Defaults.instance:
            Defaults.instance = Defaults.__Defaults(opts)
        else:
            if reset:
                Defaults.instance.__dict__ = {}
            for key, value in opts.items():
                setattr(Defaults.instance, key, value)

    # Allow for getting instance attributes
    def __getattr__(self, name):
        return getattr(self.instance, name)

    # Allow for setting instance attributes
    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)

    # Allow for printing instance attributes
    def __repr__(self):
        return self.instance.__repr__()


Defaults(default_opts)


class DataObject:
    '''
    Base class for objects serving as data containers.
    An object has user-defined data attributes, which may include
    other DataObject objects and lists of DataObject objects.

    The class provides for printing attribute values recursively, to
    a chosen depth, and for obtaining nested dictionaries of
    attributes and values.
    '''

    def __init__(self, opts={}, **kwargs):
        '''
        Constructor of DataObject class, allowing initialisation of an 
        arbitrary set of attributes.

        Parameters
        ----------
        opts : dict, default={}
            Dictionary to be used in setting instance attributes
            (dictionary keys) and their initial values.

        **kwargs
            Keyword-value pairs to be used in setting instance attributes
            and their initial values.
        '''

        for key, value in opts.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None

    def __repr__(self, depth=None):
        '''
        Create string recursively listing attributes and values.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, depth is set to the value
            of the object's print_depth property, if defined,
            or otherwise to the value of Defaults().print_depth.
        '''

        if depth is None:
            depth = self.get_print_depth()

        out_list = [f'\n{self.__class__.__name__}', '{']

        # Loop over attributes, with different treatment
        # depending on whether attribute value is a list.
        # Where an attribute value of list item is
        # an instance of DataObject or a subclass
        # it's string representation is obtained by calling
        # the instance's __repr__() method with depth decreased
        # by 1, or (depth less than 1) is the class representation.
        for key in sorted(self.__dict__):
            item = self.__dict__[key]
            if isinstance(item, list):
                items = item
                n = len(items)
                if n:
                    if depth > 0:
                        value_string = '['
                        for i, item in enumerate(items):
                            item_string = item.__repr__(depth=(depth - 1))
                            comma = ',' if (i + 1 < n) else ''
                            value_string = \
                                f'{value_string} {item_string}{comma}'
                        value_string = f'{value_string}]'
                    else:
                        value_string = f'[{n} * {item[0].__class__}]'
                else:
                    value_string = '[]'
            else:
                if issubclass(item.__class__, DataObject):
                    if depth > 0:
                        value_string = item.__repr__(depth=(depth - 1))
                    else:
                        value_string = f'{item.__class__}'
                else:
                    value_string = item.__repr__()
            out_list.append(f'  {key} : {value_string} ')
        out_list.append('}')
        out_string = '\n'.join(out_list)
        return out_string

    def get_dict(self):
        '''
        Return a nested dictionary of object attributes (dictionary keys)
        and their values.
        '''

        objects = {}
        for key in self.__dict__:
            try:
                objects[key] = self.__dict__[key].get_dict()
            except AttributeError:
                objects[key] = self.__dict__[key]

        return objects

    def get_print_depth(self):
        '''
        Retrieve the value of the object's print depth,
        setting an initial value if not previously defined.
        '''

        if not hasattr(self, 'print_depth'):
            self.set_print_depth()
        return self.print_depth

    def print(self, depth=None):
        '''
        Convenience method for recursively printing
        object attributes and values, with recursion
        to a specified depth.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, depth is set in the
            __repr__() method.
        '''

        print(self.__repr__(depth))
        return None

    def set_print_depth(self, depth=None):
        '''
        Set the object's print depth.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, the object's print depth is
            set to the value of Defaults().print_depth.
        '''

        if depth is None:
            depth = Defaults().print_depth
        self.print_depth = depth
        return None


class PathObject(DataObject):
    '''DataObject with and associated directory; has the ability to 
    extract a list of dated objects from within this directory.'''

    def __init__(self, path=''):
        self.path = fullpath(path)
        self.subdir = ''

    def get_dated_objects(self, dtype='DatedObject', subdir='', **kwargs):
        '''Create list of objects of a given type, <dtype>, inside own 
        directory, or inside own directory + <subdir> if given.'''

        # Create object for each file in the subdir
        objs = []
        path = os.path.join(self.path, subdir)
        if os.path.isdir(path):
            filenames = os.listdir(path)
            for filename in filenames:
                if is_timestamp(filename):
                    filepath = os.path.join(path, filename)
                    try:
                        objs.append(globals()[dtype](path=filepath, **kwargs))
                    except RuntimeError:
                        pass

        # Sort and assign subdir to the created objects
        objs.sort()
        if subdir:
            for obj in objs:
                obj.subdir = subdir

        return objs


@functools.total_ordering
class DatedObject(PathObject):
    '''PathObject with an associated date and time, which can be used for 
    sorting multiple DatedObjects.'''

    def __init__(self, path=''):

        PathObject.__init__(self, path)

        # Assign date and time
        timestamp = os.path.basename(self.path)
        self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(os.path.dirname(self.path))
            self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(self.path)
            try:
                self.date, self.time = timestamp.split('_')
            except ValueError:
                self.date, self.time = (None, None)

        self.timestamp = f'{self.date}_{self.time}'

    def in_date_interval(self, min_date=None, max_date=None):
        '''Check whether own date falls within an interval.'''

        if min_date:
            if self.date < min_date:
                return False
        if max_date:
            if self.date > max_date:
                return False
        return True

    def __eq__(self, other):
        return self.date == other.date and self.time == other.time

    def __ne__(self, other):
        return self.date == other.date or self.time == other.time

    def __lt__(self, other):
        if self.date == other.date:
            return self.time < other.time
        return self.date < other.date

    def __gt__(self, other):
        if self.date == other.date:
            return self.time > other.time
        return self.date > other.date

    def __le__(self, other):
        return self
        if self.date == other.date:
            return self.time < other.time
        return self.date < other.date


class MachineObject(DatedObject):
    '''Dated object with an associated machine name.'''

    def __init__(self, path=''):
        DatedObject.__init__(self, path)
        self.machine = os.path.basename(os.path.dirname(path))


class ArchiveObject(DatedObject):
    '''Dated object associated with multiple files.'''

    def __init__(self, path='', allow_dirs=False):

        DatedObject.__init__(self, path)
        self.files = []
        try:
            filenames = os.listdir(self.path)
        except OSError:
            filenames = []
        for filename in filenames:

            # Disregard hidden files
            if not filename.startswith('.'):
                filepath = os.path.join(self.path, filename)
                if not os.path.isdir(filepath) or allow_dirs:
                    self.files.append(File(path=filepath))

        self.files.sort()


class File(DatedObject):
    '''File with an associated date. Files can be sorted based on their 
    filenames.'''

    def __init__(self, path=''):
        DatedObject.__init__(self, path)

    def __cmp__(self, other):

        result = DatedObject.__cmp__(self, other)
        if not result:
            self_basename = os.path.basename(self.path)
            other_basename = os.path.basename(other.path)
            basenames = [self_basename, other_basename]
            basenames.sort(key=alphanumeric)
            if basenames[0] == self_basename:
                result = -1
            else:
                result = 1
        return result

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return self.path != other.path

    def __lt__(self, other):

        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = eval(self_name) < eval(other_name)
        except (NameError, SyntaxError):
            result = self.path < other.path
        return result

    def __gt__(self, other):

        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = eval(self_name) > eval(other_name)
        except (NameError, SyntaxError):
            result = self.path > other.path
        return result


def alphanumeric(in_str=''):
    '''Function that can be passed as value for list sort() method
    to have alphanumeric (natural) sorting'''

    import re

    elements = []
    for substr in re.split('(-*[0-9]+)', in_str):
        try:
            element = int(substr)
        except BaseException:
            element = substr
        elements.append(element)
    return elements


def fullpath(path=''):
    '''Evaluate full path, expanding '~', environment variables, and 
    symbolic links.'''

    expanded = ''
    if path:
        tmp = os.path.expandvars(path.strip())
        tmp = os.path.abspath(os.path.expanduser(tmp))
        expanded = os.path.realpath(tmp)
    return expanded


def get_time_and_date(timestamp=''):

    timeAndDate = (None, None)
    if is_timestamp(timestamp):
        valueList = os.path.splitext(timestamp)[0].split('_')
        valueList = [value.strip() for value in valueList]
        if valueList[0].isalpha():
            timeAndDate = tuple(valueList[1:3])
        else:
            timeAndDate = tuple(valueList[0:2])
    else:
        i1 = timestamp.find('_')
        i2 = timestamp.rfind('.')
        if (-1 != i1) and (-1 != i2):
            bitstamp = timestamp[i1 + 1 : i2]
            if is_timestamp(bitstamp):
                timeAndDate = tuple(bitstamp.split('_'))

    return timeAndDate


def is_timestamp(testString=''):

    timestamp = True
    valueList = os.path.splitext(testString)[0].split('_')
    valueList = [value.strip() for value in valueList]
    if len(valueList) > 2:
        if valueList[0].isalpha() and valueList[1].isdigit() \
           and valueList[2].isdigit():
            valueList = valueList[1:3]
        elif valueList[0].isdigit() and valueList[1].isdigit():
            valueList = valueList[:2]
        elif valueList[0].isdigit() and valueList[1].isdigit():
            valueList = valueList[:2]
    if len(valueList) != 2:
        timestamp = False
    else:
        for value in valueList:
            if not value.isdigit():
                timestamp = False
                break
    return timestamp


_axes = ['x', 'y', 'z']
_slice_axes = {
    'x-y': 2,
    'y-z': 0,
    'x-z': 1
}
_plot_axes = {
    'x-y': [0, 1],
    'y-z': [2, 1],
    'x-z': [2, 0]
}
_default_figsize = 6


class Image(ArchiveObject):
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
        downsample=None
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
        ArchiveObject.__init__(self, path)

        if load:
            self.load_data()

    def __repr__(self):

        self.load_data()
        out_str = 'Image\n{'
        attrs_to_print = sorted([
            'date',
            'path',
            'subdir',
            'source_type',
            'affine',
            'timestamp',
            'title',
            'downsampling'
        ])
        for attr in attrs_to_print:
            out_str += f'\n  {attr} : {getattr(self, attr)}'
        if len(self.structs):
            out_str += f'\n  structs: [{len(self.structs)} * {type(self.structs[0])}]'
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
                raise RuntimeError(f'Image input {self.source} does not exist!')
            if os.path.isfile(self.source):
                self.data, affine = load_nifti(self.source)
                self.source_type = 'nifti'
            if self.data is not None:
                self.affine = affine

        # Try loading from dicom file
        if self.data is None:
            self.data, affine, window_centre, window_width \
                    = load_dicom(self.source)
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
                window_centre + window_width / 2
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
                    affine[i, 3] = affine[i, 3] - (n_voxels - 1) \
                            * affine[i, i]

        # Adjust nifti
        elif 'nifti' in self.source_type:

            init_dtype = self.get_data().dtype
            nii = nibabel.as_closest_canonical(nibabel.Nifti1Image(
                self.data.astype(np.float64), self.affine))
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
            np.linspace(self.origin[i], 
                        self.origin[i] 
                        + (self.n_voxels[i] - 1) * self.voxel_size[i],
                        self.n_voxels[i]
                       ) 
            for i in range(3)
        ]
        for i in range(len(old_coords)):
            if old_coords[i][0] > old_coords[i][-1]:
                old_coords[i] = old_coords[i][::-1]
        interpolant = interpolate.RegularGridInterpolator(
            old_coords, self.get_data(), method='linear', bounds_error=False,
            fill_value=self.get_min())
        
        # Calculate new number of voxels
        n_voxels = [
            int(np.round(abs(self.get_image_length(i) / voxel_size[i])))
            for i in range(3)
        ]
        voxel_size = [np.sign(voxel_size[i]) * self.get_image_length(i) / 
                      n_voxels[i] for i in range(3)]
        shape = [n_voxels[1], n_voxels[0], n_voxels[2]]
        origin = [
            self.origin[i] - self.voxel_size[i] / 2 + voxel_size[i] / 2
            for i in range(3)
        ]

        # Interpolate to new coordinates
        new_coords = [
            np.linspace(
                origin[i], 
                origin[i] + (n_voxels[i] - 1) * voxel_size[i],
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
        print(f'Resampled z axis from {init_nz} x {init_vz:.3f} mm -> '
              f'{int(to_resample.n_voxels[2])} x {to_resample.voxel_size[2]:.3f} mm')

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
        pairs = [
            ('L', 'R'),
            ('P', 'A'),
            ('I', 'S')
        ]
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
            'I': [0, 0, -1]
        }
        return vecs[codes[0]] + vecs[codes[1]]

    def get_axes(self, col_first=False):
        '''Return list of axis numbers in order (column, row, slice) if 
        col_first is True, otherwise in order (row, column, slice). The axis 
        numbers 0, 1, and 2 correspond to x, y, and z, respectively.'''

        orient = np.array(self.get_orientation_vector()).reshape(2, 3)
        axes = [
            sum([abs(int(orient[i, j] * j)) for j in range(3)]) 
            for i in range(2)
        ]
        axes.append(3 - sum(axes))
        if not col_first:
            return axes
        else:
            return [axes[1], axes[0], axes[2]]

    def get_machine(self, stations=default_stations):

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
            self.affine = np.array([
                [self.voxel_size[0], 0, 0, self.origin[0]],
                [0, self.voxel_size[1], 0, self.origin[1]],
                [0, 0, self.voxel_size[2], self.origin[2]],
                [0, 0, 0, 1]
            ])
            if 'nifti' in self.source_type:
                self.affine[0, :] = -self.affine[0, :]
                self.affine[1, 3] = -(self.affine[1, 3] 
                                      + (self.data.shape[1] - 1)
                                      * self.voxel_size[1])
        else:
            self.voxel_size = list(np.diag(self.affine))[:-1]
            self.origin = list(self.affine[:-1, -1])

        # Set number of voxels
        self.n_voxels = [
            self.data.shape[1],
            self.data.shape[0],
            self.data.shape[2]
        ]

        # Set axis limits for standardised plotting
        self.standardise_data(force=force)
        self.lims = [
            (self.sorigin[i], 
             self.sorigin[i] + (self.n_voxels[i] - 1) * self.svoxel_size[i])
            for i in range(3)
        ]
        self.image_extent = [
            (self.lims[i][0] - self.svoxel_size[i] / 2,
             self.lims[i][1] + self.svoxel_size[i] / 2)
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

    def get_slice(self, view='x-y', sl=None, idx=None, pos=None, flatten=False, 
                  **kwargs):
        '''Get a slice of the data in the correct orientation for plotting.'''

        # Get image slice
        idx = self.get_idx(view, sl, idx, pos)
        transpose = {
            'x-y': (0, 1, 2),
            'y-z': (0, 2, 1),
            'x-z': (1, 2, 0)
        }[view]
        list(_plot_axes[view]) + [_slice_axes[view]]
        data = np.transpose(self.get_standardised_data(), transpose)
        if flatten:
            return np.sum(data, axis=2)
        else:
            return data[:, :, idx]

    def set_ax(self, view=None, ax=None, gs=None, figsize=_default_figsize,
               zoom=None, colorbar=False, **kwargs):
        '''Set up axes for plotting this image, either from a given exes or
        gridspec, or by creating new axes.'''

        # Set up figure/axes
        if ax is None and  gs is not None:
            ax = plt.gcf().add_subplot(gs)
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            if figsize is None:
                figsize = _default_figsize
            if is_list(figsize):
                fig_tuple = figsize
            else:
                figsize = to_inches(figsize)
                aspect = self.get_plot_aspect_ratio(
                    view, zoom, colorbar, figsize
                )
                fig_tuple = (figsize * aspect, figsize)
            self.fig = plt.figure(figsize=fig_tuple)
            self.ax = self.fig.add_subplot()

    def add_structs(self, structure_set):
        '''Add a structure set.'''

        if not isinstance(structure_set, RtStruct):
            raise TypeError('<structure_set> must be an RtStruct!')
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
                self.pos_to_slice(extent[3], y_ax, False)
            ]
            mpl_kwargs['aspect'] = abs(self.voxel_size[y_ax] 
                                       / self.voxel_size[x_ax])
        mpl_kwargs['extent'] = extent

        return mpl_kwargs

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
        flatten=False
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
                    print(f'Warning: structure set {structure_set} not found! '
                          f'Image only has {len(self.structs)} structure sets.'
                         )
            elif structure_set == 'all':
                to_plot = self.structs
            elif is_list(structure_set):
                to_plot = [self.structs[i] for i in structure_set]
            else:
                print(f'Warning: structure set option {structure_set} not '
                      'recognised! Must be an int, None, or \'all\'.')

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

        # Plot the slice
        mesh = self.ax.imshow(
            image_slice, 
            **self.get_mpl_kwargs(view, mpl_kwargs, scale_in_mm)
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
                        **struct_kwargs
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
        self.label_ax(view, idx, scale_in_mm, no_title, no_ylabel,
                      annotate_slice, major_ticks, minor_ticks, 
                      ticks_all_sides)
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

    def label_ax(self, view, idx, scale_in_mm=True, no_title=False,
                 no_ylabel=False, annotate_slice=False, major_ticks=None,
                 minor_ticks=None, ticks_all_sides=False, **kwargs):

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
                z_str = '{} = {:.1f} mm'.format(z_ax, self.idx_to_pos(idx, z_ax))
            else:
                z_str = '{} = {}'.format(z_ax, self.idx_to_slice(idx, z_ax))
            if matplotlib.colors.is_color_like(annotate_slice):
                color = annotate_slice
            else:
                color = 'white'
            self.ax.annotate(z_str, xy=(0.05, 0.93), xycoords='axes fraction',
                             color=color, fontsize='large')

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
        zoom = to_three(zoom)
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
            mid_x + (init_xlim[1] - init_xlim[0]) / (2 * zoom[x_ax])
        ]
        ylim = [
            mid_y - (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax]),
            mid_y + (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax])
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
            self.pos_to_idx(pos, ax, return_int, standardise), 
            ax)
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

    def get_plot_aspect_ratio(self, view, zoom=None, n_colorbars=0,
                              figsize=_default_figsize):
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
        max_y_digits = np.floor(np.log10(
            max([abs(lim) for lim in self.lims[y_ax]])
        ))
        minus_sign = any([lim < 0 for lim in self.lims[y_ax]])
        x_pad = (0.7 * max_y_digits + 1.2 * minus_sign + 1) * font

        # Account for zoom
        if zoom:
            zoom = to_three(zoom)
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
        if is_list(downsampling):
            if len(downsampling) != 3:
                raise TypeError('<downsample> must contain 3 elements!')
            dx, dy, dz = downsampling
        else:
            dx = dy = dz = downsampling

        # Apply to image array
        self.data = downsample(self.data, dx, dy, dz)

        # Adjust voxel sizes
        self.voxel_size = [
            v * d for v, d in zip(self.voxel_size, [dx, dy, dz])
        ]
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
            nii = nibabel.as_closest_canonical(nibabel.Nifti1Image(data, affine))
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
            affine[1, 3] = -(affine[1, 3] + (data.shape[0] - 1) *
                             self.voxel_size[1])
            if standardise:
                nii = nibabel.as_closest_canonical(nibabel.Nifti1Image(
                    data, affine))
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
        root_uid=None
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
            write_dicom(outdir, data, affine, header_source, orientation,
                        patient_id, modality, root_uid)
            print('Wrote dicom file(s) to directory:', outdir)

    def get_coords(self):
        '''Get grids of x, y, and z coordinates for each voxel in the image.'''

        if not hasattr(self, 'coords'):

            # Make coordinates
            coords_1d = [
                np.arange(
                    self.origin[i], 
                    self.origin[i] + self.n_voxels[i] * self.voxel_size[i],
                    self.voxel_size[i]
                )
                for i in range(3)
            ]
            X, Y, Z = np.meshgrid(*coords_1d)

            # Set coords
            self.coords = (X, Y, Z)

        # Apply transformations
        return self.coords


class StructDefaults:
    '''Singleton class for assigning default structure names and colours.'''

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __StructDefaults:

        def __init__(self):

            self.n_structs = 0
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
        '''Constructor of StructDefaults singleton class.'''

        if not StructDefaults.instance:
            StructDefaults.instance = StructDefaults.__StructDefaults()
        elif reset:
            StructDefaults.instance.__init__()

    def get_default_struct_name(self):
        '''Get a default name for a structure.'''

        StructDefaults.instance.n_structs += 1
        return f'ROI {StructDefaults.instance.n_structs}'

    def get_default_struct_color(self):
        '''Get a default structure color.'''

        if StructDefaults.instance.n_colors_used \
           >= len(StructDefaults.instance.colors):
            return np.random.rand(3)
        color = StructDefaults.instance.colors[
            StructDefaults.instance.n_colors_used]
        StructDefaults.instance.n_colors_used += 1
        return color

    def get_default_structure_set_name(self):
        '''Get a default name for a structure set.'''

        StructDefaults.instance.n_structure_sets += 1
        return f'RtStruct {StructDefaults.instance.n_structure_sets}'


StructDefaults()


class ROI(Image):
    '''Single structure.'''

    def __init__(
        self,
        source=None,
        name=None,
        color=None,
        load=None,
        image=None,
        shape=None,
        mask_level=0.25,
        **kwargs
    ):

        '''Load structure from mask or contour.

        Parameters
        ----------
        source : str/array/nifti, default=None
            Source of image data to load. Can be either:
                (a) The path to a nifti file containing a binary mask;
                (b) A numpy array containing a binary mask;
                (c) The path to a file containing a numpy array;
                (d) The path to a dicom structure set file.
                (e) Dictionary of contours in the x-y orienation, where the 
                keys are z positions in mm and values are lists of lists of 
                3D contour points in order (x, y, z), with one list per contour
                on that slice. These contours will be used to generate a 
                binary mask.

            If <source> is not given, <contours> and <shape> must be given in
            order to load a structure directly from a contour.

        name : str, default=None
            Name of the structure. If <source> is a file and no name is given,
            the name will be inferred from the filename.

        color : matplotlib color, default=None
            Color in which this structure will be plotted. If None, a color
            will be assigned.

        load : bool, default=True
            If True, contours/mask will be created during initialisation; 
            otherwise they will be created on-demand.

        image : Image/str, default=None
            Associated image from which to extract shape and affine matrix.

        shape : list, default=None
            Number of voxels in the image to which the structure belongs, in 
            order (x, y, z). Needed if <contours> is used instead of <source>.

        kwargs : dict, default=None
            Extra arguments to pass to the initialisation of the parent
            Image object (e.g. affine matrix if loading from a numpy array).

        '''

        # Assign properties
        if isinstance(source, dict):
            self.source = None
            self.input_contours = source
        else:
            self.source = source
            self.input_contours = None
        self.custom_color = color is not None
        self.set_color(color)
        self.image = image
        if image and not isinstance(image, Image):
            self.image = Image(image)
        self.shape = shape
        self.mask_level = mask_level
        self.kwargs = kwargs

        # Create name
        self.name = name
        if self.name is None:
            if isinstance(self.source, str):
                basename = os.path.basename(self.source).split('.')[0]
                name = re.sub(r'RTSTRUCT_[MVCT]+_\d+_\d+_\d+', '', basename)
                self.name = name[0].upper() + name[1:]
            else:
                self.name = StructDefaults().get_default_struct_name()

        # Load structure data
        self.loaded = False
        self.loaded_contours = False
        self.loaded_mask = False
        if load:
            self.load()

    def __repr__(self):

        out_str = 'ROI\n{'
        attrs_to_print = sorted([
            'name',
            'color',
            'loaded_contours',
            'loaded_mask'
        ])
        for attr in attrs_to_print:
            if hasattr(self, attr):
                out_str += f'\n {attr}: {getattr(self, attr)}'
        out_str += '\n}'
        return out_str

    def load(self):
        '''Load structure from file.'''

        if self.loaded:
            return

        if self.image:
            self.image.load_data()

        # Try loading from dicom structure set
        structs = []
        self.source_type = None
        if isinstance(self.source, str):
        
            structs = load_structs_dicom(self.source, names=self.name)
            if len(structs):

                # Check a shape or image was given
                if self.shape is None and self.image is None:
                    raise RuntimeError('Must provide an associated image or '
                                       'image shape if loading a structure '
                                       'from dicom!')

                # Get structure info
                struct = structs[list(structs.keys())[0]]
                self.name = struct['name']
                self.input_contours = struct['contours']
                if not self.custom_color:
                    self.set_color(struct['color'])
                self.source_type = 'dicom'

        # Load structure mask
        if not len(structs) and self.source is not None:
            Image.__init__(self, self.source, **self.kwargs)
            self.loaded = True
            self.create_mask()

        # Deal with input from dicom
        if self.input_contours is not None:

            # Create Image object
            if self.image is not None:
                self.kwargs['voxel_size'] = self.image.voxel_size
                self.kwargs['origin'] = self.image.origin
                self.shape = self.image.data.shape
                Image.__init__(self, np.zeros(self.shape), **self.kwargs)

            # Set x-y contours with z indices as keys
            self.contours = {'x-y': {}}
            for z, contours in self.input_contours.items():
                iz = self.pos_to_idx(z, 'z')
                self.contours['x-y'][iz] = [
                    [tuple(p[:2]) for p in points] for points in contours
                ]
            self.loaded = True

    def get_contours(self, view='x-y'):
        '''Get dict of contours in a given orientation.'''

        self.create_contours()
        return self.contours[view]

    def get_mask(self, view='x-y', flatten=False):
        '''Get binary mask.'''

        self.load()
        self.create_mask()
        if not flatten:
            return self.data
        return np.sum(self.get_standardised_data(), 
                      axis=_slice_axes[view]).astype(bool)

    def create_contours(self):
        '''Create contours in all orientations.'''
        
        if self.loaded_contours:
            return
        if not self.loaded:
            self.load()

        if not hasattr(self, 'contours'):
            self.contours = {}
        self.create_mask()

        # Create contours in every orientation
        for view, z_ax in _slice_axes.items():
            if view in self.contours:
                continue

            # Make new contours from mask
            self.contours[view] = {}
            for iz in range(self.n_voxels[z_ax]):

                # Get slice of mask array
                mask_slice = self.get_slice(view, idx=iz).T
                if mask_slice.max() < 0.5:
                    continue 

                points = self.mask_to_contours(mask_slice, view)
                if points:
                    self.contours[view][iz] = points

        self.loaded_contours = True

    def mask_to_contours(self, mask, view, invert=False):
        '''Create contours from a mask.'''

        contours = skimage.measure.find_contours(
            mask, 0.5, 'low', 'low')

        # Convert indices to positions in mm
        x_ax, y_ax = _plot_axes[view]
        points = []
        for contour in contours:
            contour_points = []
            for ix, iy in contour:
                px = self.idx_to_pos(ix, x_ax)
                py = self.idx_to_pos(iy, y_ax)
                if invert:
                    px, py = py, px
                contour_points.append((px, py))
            points.append(contour_points)

        return points

    def create_mask(self):
        '''Create binary mask.'''

        if self.loaded_mask:
            return
        if not self.loaded:
            self.load()

        # Create mask from x-y contours if needed
        if self.input_contours:

            # Check an image or shape was given
            if self.image is None and self.shape is None:
                raise RuntimeError('Must set structure.image or structure.shape'
                                   ' before creating mask!')
            if self.image is None:
                Image.__init__(self, np.zeros(self.shape), **self.kwargs)

            # Create mask on each z layer
            for z, contours in self.input_contours.items():

                # Convert z position to index
                iz = self.pos_to_idx(z, 'z')

                # Loop over each contour on the z slice
                pos_to_idx_vec = np.vectorize(self.pos_to_idx)
                for points in contours:

                    # Convert (x, y) positions to array indices
                    points_idx = np.zeros((points.shape[0], 2))
                    for i in range(2):
                        points_idx[:, i] = pos_to_idx_vec(points[:, i], i,
                                                          return_int=False)

                    # Create polygon
                    polygon = geometry.Polygon(np.unique(points_idx, axis=0))

                    # Get polygon's bounding box
                    ix1, iy1, ix2, iy2 = [int(xy) for xy in polygon.bounds]
                    ix1 = max(0, ix1)
                    ix2 = min(ix2 + 1, self.shape[1])
                    iy1 = max(0, iy1)
                    iy2 = min(iy2 + 1, self.shape[0])

                    # Loop over pixels
                    for ix in range(ix1, ix2):
                        for iy in range(iy1, iy2):

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
                            self.data[iy, ix, int(iz)] += overlap
                            
            self.data = self.data > self.mask_level

        # Convert to boolean mask
        if hasattr(self, 'data'):
            if not self.data.dtype == 'bool':
                self.data = self.data > 0.5
            if not hasattr(self, 'empty'):
                self.empty = not np.any(self.data)
            self.loaded_mask = True

    def resample(self, *args, **kwargs):
        self.create_mask()
        Image.resample(self, *args, **kwargs)

    def match_voxel_size(self, other, *args, **kwargs):
        
        if isinstance(other, ROI):
            other.create_mask()
        self.create_mask()
        Image.match_voxel_size(self, other, *args, **kwargs)

    def get_slice(self, *args, **kwargs):

        self.create_mask()
        return Image.get_slice(self, *args, **kwargs).astype(bool)

    def get_indices(self, view='x-y', slice_num=False):
        '''Get list of slice indices on which this structure exists. If 
        <slice_num> is True, slice numbers will be returned instead of 
        indices.'''

        if not hasattr(self, 'contours') or view not in self.contours:
            self.create_contours()
        indices = list(self.contours[view].keys())
        if slice_num:
            z_ax = _slice_axes[view]
            return [self.idx_to_slice(i, z_ax) for i in indices]
        else:
            return indices

    def get_mid_idx(self, view='x-y', slice_num=False):
        '''Get central slice index of this structure in a given orientation.'''

        indices = self.get_indices(view, slice_num=slice_num) 
        if not len(indices):
            return None
        return round(np.mean(indices))

    def on_slice(self, view, sl=None, idx=None, pos=None):
        '''Check whether this structure exists on a given slice.'''

        self.create_mask()
        idx = self.get_idx(view, sl, idx, pos)
        return idx in self.get_indices(view)

    def get_centroid(self, view=None, sl=None, idx=None, pos=None, units='mm',
                     standardise=True, flatten=False):
        '''Get centroid position in 2D or 3D.'''
        
        # Get 2D or 3D data from which to calculate centroid
        if view or sl or idx or pos:
            if sl is None and idx is None and pos is None:
                idx = self.get_mid_idx(view)
            if view is None:
                view = 'x-y'
            if not self.on_slice(view, sl, idx, pos):
                return [None, None]
            data = self.get_slice(view, sl, idx, pos)
            axes = _plot_axes[view]
        else:
            if flatten:
                if view is None:
                    view = 'x-y'
                data = self.get_mask(view, flatten=True)
            else:
                self.create_mask()
                data = self.get_data(standardise)
            axes = _axes

        # Compute centroid
        non_zero = np.argwhere(data)
        if not len(non_zero):
            if data.ndim == 2:
                return None, None
            else:
                return None, None, None
        centroid_rowcol = list(non_zero.mean(0))
        centroid = [centroid_rowcol[1], centroid_rowcol[0]] \
                + centroid_rowcol[2:] 
        
        # Convert to mm
        if units == 'mm':
            centroid = [self.idx_to_pos(c, axes[i]) for i, c in 
                        enumerate(centroid)]
        return centroid

    def get_centre(self, view=None, sl=None, idx=None, pos=None, units='mm',
                   standardise=True):
        '''Get centre position in 2D or 3D.'''

        # Get 2D or 3D data for which to calculate centre
        if view is None:
            data = self.get_data(standardise)
            axes = _axes
        else:
            if sl is None and idx is None and pos is None:
                idx = self.get_mid_idx(view)
            data = self.get_slice(view, sl, idx, pos)
            axes = _plot_axes[view]

        # Calculate mean of min and max positions
        non_zero = np.argwhere(data)
        if not len(non_zero):
            return [0 for i in axes]
        centre_rowcol = list((non_zero.max(0) + non_zero.min(0)) / 2)
        centre = [centre_rowcol[1], centre_rowcol[0]] + centre_rowcol[2:]
        
        # Convert to mm
        if units == 'mm':
            centre = [self.idx_to_pos(c, axes[i]) for i, c in enumerate(centre)]
        return centre

    def get_volume(self, units='mm'):
        '''Get structure volume.'''
        
        if hasattr(self, 'volume'):
            return self.volume[units]

        self.create_mask()
        self.volume = {}
        self.volume['voxels'] = self.data.astype(bool).sum()
        self.volume['mm'] = self.volume['voxels'] * abs(np.prod(self.voxel_size))
        self.volume['ml'] = self.volume['mm'] * (0.1 ** 3)
        return self.volume[units]

    def get_area(self, view='x-y', sl=None, idx=None, pos=None, units='mm',
                 flatten=False):
        '''Get the area of the structure on a given slice.'''

        if view is None:
            view = 'x-y'
        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        im_slice = self.get_slice(view, sl, idx, pos, flatten=flatten)
        area = im_slice.astype(bool).sum()
        if units == 'mm':
            x_ax, y_ax = _plot_axes[view]
            area *= abs(self.voxel_size[x_ax] * self.voxel_size[y_ax])
        return area

    def get_length(self, units='mm', ax='z'):
        '''Get total length of the structure along a given axis.'''
        
        if hasattr(self, 'length') and ax in self.length:
            return self.length[ax][units]
        
        self.create_mask()
        if not hasattr(self, 'length'):
            self.length = {}
        self.length[ax] = {}

        nonzero = np.argwhere(self.data)
        vals = nonzero[:, _axes.index(ax)]
        if len(vals):
            self.length[ax]['voxels'] = max(vals) + 1 - min(vals)
            self.length[ax]['mm'] = self.length[ax]['voxels'] \
                    * abs(self.voxel_size[_axes.index(ax)])
        else:
            self.length[ax] = {'voxels': 0, 'mm': 0}

        return self.length[ax][units]

    def get_geometry(
        self, 
        metrics=['volume', 'area', 'centroid', 'x_length', 'y_length', 'z_length'],
        vol_units='mm',
        area_units='mm',
        length_units='mm',
        centroid_units='mm',
        view=None,
        sl=None,
        pos=None,
        idx=None
    ):
        '''Get a pandas DataFrame of the geometric properties listed in 
        <metrics>.

        Possible metrics:
            - 'volume': volume of entire structure.
            - 'area': area either of the central x-y slice, or of a given 
            view/slice if either sl/pos/idx are given.
            - 'centroid': centre-of-mass either of the entire structure, or a 
            given view/slice if either sl/pos/idx are given.
            - 'centroid_global': centre-of-mass of the entire structure (note 
            that this is the same as 'centroid' if sl/pos/idx are all None)
            - 'x_length': structure length along the x axis
            - 'y_length': structure length along the y axis
            - 'z_length': structure length along the z axis
        '''

        # Parse volume and area units
        vol_units_name = vol_units
        if vol_units in ['mm', 'mm3']:
            vol_units = 'mm'
            vol_units_name = 'mm3'
        area_units_name = vol_units
        if area_units in ['mm', 'mm2']:
            area_units = 'mm'
            area_units_name = 'mm2'

        # Make dict of property names
        names = {
            'volume': f'Volume ({vol_units_name})',
            'area': f'Area ({area_units_name})',
        }
        for ax in _axes:
            names[f'{ax}_length'] = f'{ax} length ({length_units})'
            names[f'centroid_{ax}'] = f'Centroid {ax} ({centroid_units})'
            names[f'centroid_global_{ax}'] = f'Global centroid {ax} ({centroid_units})'

        # Make dict of functions and args for each metric
        funcs = {
            'volume': (
                self.get_volume, {'units': vol_units}
            ),
            'area': (
                self.get_area, {'units': area_units, 'view': view, 'sl': sl, 
                                'pos': pos, 'idx': idx}
            ),
            'centroid': (
                self.get_centroid, {'units': centroid_units, 'view': view, 
                                    'sl': sl, 'pos': pos, 'idx': idx}
            ),
            'centroid_global': (
                self.get_centroid, {'units': centroid_units}
            )
        }
        for ax in _axes:
            funcs[f'{ax}_length'] = (
                self.get_length, {'ax': ax, 'units': length_units}
            )

        # Make dict of metrics
        geom = {
            m: funcs[m][0](**funcs[m][1]) for m in metrics
        }

        # Split centroid into multiple entries
        for cname in ['centroid', 'centroid_global']:
            if cname in geom:
                centroid_vals = geom.pop(cname)
                axes = [0, 1, 2] if len(centroid_vals) == 3 \
                    else _plot_axes[view]
                for i, i_ax in enumerate(axes):
                    ax = _axes[i_ax]
                    geom[f'{cname}_{ax}'] = centroid_vals[i]

        geom_named = {names[m]: g for m, g in geom.items()}
        return pd.DataFrame(geom_named, index=[self.name])

    def get_centroid_vector(self, roi, **kwargs):
        '''Get centroid displacement vector with respect to another ROI.'''

        this_centroid = np.array(self.get_centroid(**kwargs))
        other_centroid = np.array(roi.get_centroid(**kwargs))
        if None in this_centroid or None in other_centroid:
            return None, None
        return other_centroid - this_centroid

    def get_centroid_distance(self, roi, **kwargs):
        '''Get absolute centroid distance.'''

        centroid = self.get_centroid_vector(roi, **kwargs)
        if None in centroid:
            return None
        return np.linalg.norm(centroid)

    def get_dice(self, roi, view='x-y', sl=None, idx=None, pos=None, 
                 flatten=False):
        '''Get Dice score, either global or on a given slice.'''

        if view is None:
            view = 'x-y'
        if sl is None and idx is None and pos is None:
            data1 = self.get_mask(view, flatten)
            data2 = roi.get_mask(view, flatten)
        else:
            data1 = self.get_slice(view, sl, idx, pos)
            data2 = roi.get_slice(view, sl, idx, pos)

        return (data1 & data2).sum() / np.mean([data1.sum(), data2.sum()])

    def get_volume_ratio(self, roi):
        '''Get ratio of another ROI's volume with respect to own volume.'''

        own_volume = roi.get_volume()
        other_volume = self.get_volume()
        if not other_volume:
            return None
        return own_volume / other_volume

    def get_area_ratio(self, roi, **kwargs):
        '''Get ratio of another ROI's area with respect to own area.'''

        own_area = roi.get_area(**kwargs)
        other_area = self.get_area(**kwargs)
        if not other_area:
            return None
        return own_area / other_area

    def get_relative_volume_diff(self, roi, units='mm'):
        '''Get relative volume of another ROI with respect to own volume.'''

        own_volume = self.get_volume(units)
        other_volume = roi.get_volume(units)
        if not own_volume:
            return None
        return (own_volume - other_volume) / own_volume

    def get_area_diff(self, roi, **kwargs):
        '''Get absolute area difference between two ROIs.'''

        own_area = self.get_area(**kwargs)
        other_area = roi.get_area(**kwargs)
        if not own_area or not other_area:
            return None
        return own_area - other_area

    def get_relative_area_diff(self, roi, **kwargs):
        '''Get relative area of another ROI with respect to own area.'''

        own_area = self.get_area(**kwargs)
        other_area = roi.get_area(**kwargs)
        if not own_area or not other_area:
            return None
        return (own_area - other_area) / own_area

    def get_surface_distances(self, roi, signed=False, view=None, sl=None, 
                              idx=None, pos=None, connectivity=2, 
                              flatten=False):
        '''Get vector of surface distances between two ROIs.'''

        # Ensure both ROIs are loaded
        self.load()
        roi.load()

        # Check whether ROIs are empty
        if not np.any(self.get_mask()) or not np.any(roi.get_mask()):
            return

        # Get binary masks and voxel sizes
        if flatten and view is None:
            view = 'x-y'
        if view or sl or idx or pos:
            voxel_size = [self.voxel_size[i] for i in _plot_axes[view]]
            if not flatten:
                mask1 = self.get_slice(view, sl=sl, idx=idx, pos=pos)
                mask2 = roi.get_slice(view, sl=sl, idx=idx, pos=pos)
            else:
                mask1 = self.get_mask(view, True)
                mask2 = roi.get_mask(view, True)
        else:
            vx, vy, vz = self.voxel_size
            voxel_size = [vy, vx, vz]
            mask1 = self.get_mask()
            mask2 = roi.get_mask()

        # Make structuring element
        conn2 = morphology.generate_binary_structure(2, connectivity)
        if mask1.ndim == 2:
            conn = conn2
        else:
            conn = np.zeros((3, 3, 3), dtype=bool)
            conn[:, :, 1] = conn2

        # Get outer pixel of binary maps
        surf1 = mask1 ^ morphology.binary_erosion(mask1, conn)
        surf2 = mask2 ^ morphology.binary_erosion(mask2, conn)

        # Make arrays of distances to surface of each pixel
        dist1 = morphology.distance_transform_edt(~surf1, voxel_size)
        dist2 = morphology.distance_transform_edt(~surf2, voxel_size)

        # Get signed arrays
        if signed:
            dist1 = dist1 * ~mask1 - dist1 * mask1
            dist2 = dist2 * ~mask2 - dist2 * mask2

        # Make vector containing all distances
        sds = np.concatenate([np.ravel(dist1[surf2 != 0]), 
                                       np.ravel(dist2[surf1 != 0])])
        return sds

    def get_mean_surface_distance(self, roi, **kwargs):

        sds = self.get_surface_distances(roi, **kwargs)
        if sds is None:
            return
        return sds.mean()

    def get_rms_surface_distance(self, roi, **kwargs):

        sds = self.get_surface_distances(roi, **kwargs)
        if sds is None:
            return
        return np.sqrt((sds ** 2).mean())

    def get_hausdorff_distance(self, roi, **kwargs):

        sds = self.get_surface_distances(roi, **kwargs)
        if sds is None:
            return
        return sds.max()

    def get_surface_distance_metrics(self, roi, **kwargs):
        '''Get the mean surface distance, RMS surface distance, and Hausdorff
        distance.'''

        sds = self.get_surface_distances(roi, **kwargs)
        if sds is None:
            return
        return sds.mean(), np.sqrt((sds ** 2).mean()), sds.max()

    def plot_surface_distances(self, roi, save_as=None, signed=False, **kwargs):
        '''Plot histogram of surface distances.'''

        sds = self.get_surface_distances(roi, signed=signed, **kwargs)
        if sds is None:
            return
        fig, ax = plt.subplots()
        ax.hist(sds)
        xlabel = 'Surface distance (mm)' if not signed \
                else 'Signed surface distance (mm)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of voxels')
        if save_as:
            plt.tight_layout()
            fig.savefig(save_as)

    def get_comparison(
        self, 
        roi, 
        metrics=['dice', 'abs_centroid', 'rel_volume_diff', 'rel_area_diff'],
        fancy_names=True,
        vol_units='mm',
        area_units='mm',
        centroid_units='mm',
        view=None,
        sl=None,
        idx=None,
        pos=None
    ):
        '''Get a pandas DataFrame of comparison metrics with another ROI.

        Possible metrics:
            - 'dice': Dice score, either on a slice (if view and sl/idx/pos
            are given), or globally.
            - 'dice_global': Global dice score (note that this is the same as
            'dice' if no view and sl/idx/pos are given).
            - 'dice_flat': Global dice score of flat ROIs.
            - 'centroid_global': Centroid distance vector.
            - 'abs_centroid_global': Absolute centroid distance.
            - 'centroid': Centroid distance vector on slice.
            - 'abs_centroid': Absolute centroid distance on slice.
            - 'abs_centroid_flat': Absolute centroid distance on flat ROIs.
            - 'rel_volume_diff': Relative volume difference.
            - 'area_diff': Absolute area difference.
            - 'area_diff_flat': Absolute area difference of projections of 
            each ROI.
            - 'rel_area_diff': Relative area difference, either on a specific
            slice or on the central 'x-y' slice of each structure, if no 
            view and idx/pos/sl are given.
            - 'rel_area_diff_flat': Relative area difference of projections of 
            each ROI.
            - 'volume_ratio': Volume ratio.
            - 'area_ratio': Area ratio, either on a specific slice or of the
            central slices of the two ROIs.
            - 'area_ratio_flat': Area ratio of projections of each ROI.
            - 'mean_surface_distance': Mean surface distance.
            - 'mean_surface_distance_flat': Mean surface distance of 
            flat ROIs.
            - 'mean_surface_distance_signed_flat': Mean signed surface 
            distance of flat ROIs.
            - 'rms_surface_distance'
            - 'rms_surface_distance_flat'
            - 'hausdorff_distance'
            - 'hausdorff_distance_flat'
            
        '''

        # Parse volume and area units
        vol_units_name = vol_units
        if vol_units in ['mm', 'mm3']:
            vol_units = 'mm'
            vol_units_name = 'mm3'
        area_units_name = vol_units
        if area_units in ['mm', 'mm2']:
            area_units = 'mm'
            area_units_name = 'mm2'

        # Make dict of property names
        names = {
            'dice': f'Dice score',
            'dice_global': 'Global Dice score',
            'dice_flat': 'Flattened Dice score',
            'abs_centroid': f'Centroid distance ({centroid_units})',
            'abs_centroid_flat': f'Flattened centroid distance ({centroid_units})',
            'abs_centroid_global': f'Global centroid distance ({centroid_units})',
            'rel_volume_diff': f'Relative volume difference ({vol_units_name})',
            'area_diff': f'Area difference ({area_units_name})',
            'area_diff_flat': f'Flattened area difference ({area_units_name})',
            'rel_area_diff': f'Relative area difference ({area_units_name})',
            'rel_area_diff_flat': 
            f'Flattened relative area difference ({area_units_name})',
            'volume_ratio': f'Volume ratio',
            'area_ratio': f'Area ratio',
            'area_ratio_flat': f'Flattened area ratio',
            'mean_surface_distance': f'Mean surface distance (mm)',
            'mean_surface_distance_flat': f'Flattened mean surface distance (mm)',
            'mean_surface_distance_signed_flat': f'Flattened mean signed surface distance (mm)',
            'rms_surface_distance': f'RMS surface distance (mm)',
            'rms_surface_distance_flat': f'Flattened RMS surface distance (mm)',
            'hausdorff_distance': f'Hausdorff distance (mm)',
            'hausdorff_distance_flat': f'Flattened Hausdorff distance (mm)',
        }
        for ax in _axes:
            names[f'centroid_{ax}'] = f'Centroid {ax} distance ({centroid_units})'
            names[f'centroid_global_{ax}'] \
                    = f'Global centroid {ax} distance ({centroid_units})'

        # Make dict of functions and args for each metric
        funcs = {
            'dice': (
                self.get_dice, {'roi': roi, 'view': view, 'sl': sl, 'idx': idx,
                                'pos': pos}
            ),
            'dice_global': (
                self.get_dice, {'roi': roi}
            ),
            'dice_flat': (
                self.get_dice, {'roi': roi, 'flatten': True}
            ),
            'abs_centroid': (
                self.get_centroid_distance, {'roi': roi, 
                                             'units': centroid_units, 
                                             'view': view, 'sl': sl, 
                                             'pos': pos, 'idx': idx}
            ),
            'abs_centroid_flat': (
                self.get_centroid_distance, {'roi': roi, 'flatten': True,
                                             'units': centroid_units, 
                                             'view': view, 'sl': sl, 
                                             'pos': pos, 'idx': idx}
            ),
            'centroid': (
                self.get_centroid_vector, {'roi': roi, 
                                           'units': centroid_units, 
                                           'view': view, 'sl': sl, 
                                           'pos': pos, 'idx': idx}
            ),
            'abs_centroid_global': (
                self.get_centroid_distance, {'roi': roi, 
                                             'units': centroid_units}
            ),
            'centroid_global': (
                self.get_centroid_vector, {'roi': roi, 
                                           'units': centroid_units}
            ),
            'rel_volume_diff': (
                self.get_relative_volume_diff, {'roi': roi, 'units': vol_units}
            ),
            'rel_area_diff': (
                self.get_relative_area_diff, {'roi': roi, 'units': area_units,
                                              'view': view, 'sl': sl,
                                              'pos': pos, 'idx': idx}
            ),
            'rel_area_diff_flat': (
                self.get_relative_area_diff, {'roi': roi, 'units': area_units,
                                              'flatten': True}
            ),
            'volume_ratio': (
                self.get_volume_ratio, {'roi': roi}
            ),
            'area_ratio': (
                self.get_area_ratio, {'roi': roi, 'view': view, 'sl': sl,
                                      'pos': pos, 'idx': idx}
            ),
            'area_ratio_flat': (
                self.get_area_ratio, {'roi': roi, 'flatten': True}
            ),
            'area_diff': (
                self.get_area_diff, {'roi': roi, 'view': view, 'sl': sl,
                                     'pos': pos, 'idx': idx}
            ),
            'area_diff_flat': (
                self.get_area_diff, {'roi': roi, 'flatten': True}
            ),
            'mean_surface_distance': (
                self.get_mean_surface_distance, {'roi': roi},
            ),
            'mean_surface_distance_flat': (
                self.get_mean_surface_distance, {'roi': roi, 'flatten': True},
            ),
            'mean_surface_distance_signed_flat': (
                self.get_mean_surface_distance, {'roi': roi, 'flatten': True,
                                                 'signed': True},
            ),
            'rms_surface_distance': (
                self.get_rms_surface_distance, {'roi': roi},
            ),
            'rms_surface_distance_flat': (
                self.get_rms_surface_distance, {'roi': roi, 'flatten': True},
            ),
            'hausdorff_distance': (
                self.get_hausdorff_distance, {'roi': roi},
            ),
            'hausdorff_distance_flat': (
                self.get_hausdorff_distance, {'roi': roi, 'flatten': True},
            ),
        }

        # Make dict of metrics
        comp = {
            m: funcs[m][0](**funcs[m][1]) for m in metrics
        }

        # Split centroid into multiple entries
        for cname in ['centroid', 'centroid_global']:
            if cname in comp:
                centroid_vals = comp.pop(cname)
                if view is None:
                    view = 'x-y'
                axes = [0, 1, 2] if len(centroid_vals) == 3 \
                    else _plot_axes[view]
                for i, i_ax in enumerate(axes):
                    ax = _axes[i_ax]
                    comp[f'{cname}_{ax}'] = centroid_vals[i]

        if fancy_names:
            comp = {names[m]: c for m, c in comp.items()}
        name = self.get_comparison_name(roi)
        return pd.DataFrame(comp, index=[name])

    def get_comparison_name(self, roi, camelcase=False):
        '''Get name of comparison between this ROI and another.'''

        if self.name == roi.name:
            name = self.name
            if camelcase:
                return name.replace(' ', '_')
            return name
        else:
            if camelcase:
                return f'{self.name}_vs_{roi.name}'.replace(' ', '_')
            return f'{self.name} vs. {roi.name}'

    def set_color(self, color):
        '''Set plotting color.'''
        
        if color is not None and not matplotlib.colors.is_color_like(color):
            print(f'Warning: {color} not a valid color!')
            color = None
        if color is None:
            color = StructDefaults().get_default_struct_color()
        self.color = matplotlib.colors.to_rgba(color)

    def plot(
        self, 
        view='x-y',
        plot_type='contour',
        sl=None,
        idx=None,
        pos=None,
        opacity=None,
        linewidth=None,
        contour_kwargs=None,
        mask_kwargs=None,
        zoom=None,
        zoom_centre=None,
        color=None,
        **kwargs
    ):
        '''Plot this structure as either a mask or a contour.'''

        show_centroid = 'centroid' in plot_type
        if zoom and zoom_centre is None:
            zoom_centre = self.get_zoom_centre(view)
        if color is None:
            color = self.color

        # Plot a mask
        if plot_type == 'mask':
            self.plot_mask(view, sl, idx, pos, mask_kwargs, opacity, 
                           zoom_centre=zoom_centre, **kwargs)

        # Plot a contour
        elif plot_type in ['contour', 'centroid']:
            self.plot_contour(view, sl, idx, pos, contour_kwargs, linewidth,
                              centroid=show_centroid, zoom=zoom, 
                              zoom_centre=zoom_centre, color=color,
                              **kwargs)

        # Plot transparent mask + contour
        elif 'filled' in plot_type:
            if opacity is None:
                opacity = 0.3
            self.plot_mask(view, sl, idx, pos, mask_kwargs, opacity, **kwargs)
            kwargs['ax'] = self.ax
            kwargs['include_image'] = False
            self.plot_contour(view, sl, idx, pos, contour_kwargs, linewidth,
                              centroid=show_centroid, zoom=zoom, 
                              zoom_centre=zoom_centre, color=color,
                              **kwargs)

        else:
            print('Unrecognised structure plotting option:', plot_type)

    def plot_mask(
        self,
        view='x-y',
        sl=None,
        idx=None,
        pos=None,
        mask_kwargs=None,
        opacity=None,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        color=None,
        flatten=False,
        **kwargs
    ):
        '''Plot the structure as a mask.'''

        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)
        self.create_mask()
        self.set_ax(view, ax, gs, figsize)
        mask_slice = self.get_slice(view, idx=idx, flatten=flatten)

        # Make colormap
        norm = matplotlib.colors.Normalize()
        cmap = matplotlib.cm.hsv
        s_colors = cmap(norm(mask_slice))
        if color is None:
            color = self.color
        s_colors[mask_slice > 0, :] = color
        s_colors[mask_slice == 0, :] = (0, 0,  0, 0)

        # Get plotting arguments
        if mask_kwargs is None:
            mask_kwargs = {}
        mask_kwargs.setdefault('alpha', opacity)
        mask_kwargs.setdefault('interpolation', 'none')

        # Make plot
        if include_image:
            self.image.plot(view, idx=idx, ax=self.ax, show=False)
        self.ax.imshow(s_colors, extent=self.plot_extent[view], **mask_kwargs)

        # Adjust axes
        self.label_ax(view, idx, **kwargs)
        self.zoom_ax(view, zoom, zoom_centre)

    def plot_contour(
        self,
        view='x-y',
        sl=None,
        idx=None,
        pos=None,
        contour_kwargs=None,
        linewidth=None,
        centroid=False,
        show=True,
        ax=None,
        gs=None,
        figsize=None,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        color=None,
        flatten=False,
        **kwargs
    ):
        '''Plot the structure as a contour.'''

        self.load()
        if not hasattr(self, 'contours') or view not in self.contours:
            self.create_contours()

        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)
        if not self.on_slice(view, idx=idx):
            return
        if figsize is None:
            x_ax, y_ax = _plot_axes[view]
            aspect = self.get_length(ax=_axes[x_ax]) / self.get_length(ax=_axes[y_ax])
            figsize = (aspect * _default_figsize, _default_figsize)
        self.set_ax(view, ax, gs, figsize)

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs.setdefault('color', color)
        contour_kwargs.setdefault('linewidth', linewidth)

        # Plot underlying image
        if include_image:
            self.image.plot(view, idx=idx, ax=self.ax, show=False)

        # Get contour points
        if flatten:
            mask = self.get_slice(view, idx=idx, flatten=True)
            contours = self.mask_to_contours(mask, view, invert=True)
        else:
            contours = self.contours[view][idx]

        # Plot contour
        for points in contours:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            self.ax.plot(points_x, points_y, **contour_kwargs)
        
        # Check whether y axis needs to be inverted
        if not (self.plot_extent[view][3] > self.plot_extent[view][2]) \
           == (self.ax.get_ylim()[1] > self.ax.get_ylim()[0]):
            self.ax.invert_yaxis()

        # Plot centroid point
        if centroid:
            self.ax.plot(
                *self.get_centroid(view, sl, idx, pos, flatten=flatten), '+',
                **contour_kwargs)

        # Adjust axes
        self.ax.set_aspect('equal')
        self.label_ax(view, idx, **kwargs)
        self.zoom_ax(view, zoom, zoom_centre)

    def plot_comparison(self, other, legend=True, save_as=None, names=None, 
                        **kwargs):
        '''Plot comparison with another ROI.'''

        if self.color == other.color:
            roi2_color = StructDefaults().get_default_struct_color()
        else:
            roi2_color = other.color

        self.plot(**kwargs)
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()
        kwargs['ax'] = self.ax
        kwargs['color'] = roi2_color
        kwargs['include_image'] = False
        other.plot(**kwargs)
        self.ax.set_title(self.get_comparison_name(other))

        if legend:
            if names:
                roi1_name = names[0]
                roi2_name = names[1]
            else:
                roi1_name = self.name
                roi2_name = other.name
            handles = [
                mpatches.Patch(color=self.color, label=roi1_name),
                mpatches.Patch(color=roi2_color, label=roi2_name)
            ]
            self.ax.legend(handles=handles, framealpha=1, 
                           facecolor='white', loc='lower left')

        if save_as:
            plt.tight_layout()
            self.fig.savefig(save_as)

    def get_zoom_centre(self, view):
        '''Get coordinates to zoom in on this structure.'''

        zoom_centre = [None, None, None]
        x_ax, y_ax = _plot_axes[view]
        x, y = self.get_centre(view)
        zoom_centre[x_ax] = x
        zoom_centre[y_ax] = y
        return zoom_centre

    def write(self, outname=None, outdir='.', ext=None, **kwargs):

        self.load()

        # Generate output name if not given
        possible_ext = ['.dcm', '.nii.gz', '.nii', '.npy', '.txt']
        if outname is None:
            if ext is None:
                ext = '.nii'
            else:
                if ext not in possible_ext:
                    raise RuntimeError(f'Unrecognised file extension: {ext}')
            if not ext.startswith('.'):
                ext = f'.{ext}'
            outname = f'{outdir}/{self.name}{ext}'

        # Otherwise, infer extension from filename
        else:

            # Find any of the valid file extensions
            for pos in possible_ext:
                if outname.endswith(pos):
                    ext = pos
            if ext not in possible_ext:
                raise RuntimeError(f'Unrecognised output file type: {outname}')

            outname = os.path.join(outdir, outname)

        # Write points to text file
        if ext == '.txt':

            self.load()
            if not 'x-y' in self.contours:
                self.create_contours()

            with open(outname, 'w') as file:
                file.write('point\n')
                points = []
                for z, contours in self.contours['x-y'].items():
                    for contour in contours:
                        for point in contour:
                            points.append(f'{point[0]} {point[1]} {z}')
                file.write(str(len(points)) + '\n')
                file.write('\n'.join(points))
                file.close()
            return

        # Write array to nifti or npy
        elif ext != '.dcm':
            self.create_mask()
            Image.write(self, outname, **kwargs)
        else:
            print('Warning: dicom structure writing not currently available!')


class RtStruct(ArchiveObject):
    '''Structure set.'''

    def __init__(
        self,
        sources=None,
        name=None,
        image=None,
        load=True,
        names=None,
        to_keep=None,
        to_remove=None,
    ):
        '''Load structures from sources.'''

        self.name = name
        if name is None:
            self.name = StructDefaults().get_default_structure_set_name()
        self.sources = sources
        if self.sources is None:
            self.sources = []
        elif not is_list(sources):
            self.sources = [sources]
        self.structs = []
        self.set_image(image)
        self.to_keep = to_keep
        self.to_remove = to_remove
        self.names = names

        path = sources if isinstance(sources, str) else ''
        ArchiveObject.__init__(self, path)

        self.loaded = False
        if load:
            self.load()

    def load(self, sources=None, force=False):
        '''Load structures from sources. If None, will load from own 
        self.sources.'''

        if self.loaded and not force and sources is None:
            return

        if sources is None:
            sources = self.sources
        elif not is_list(sources):
            sources = [sources]

        # Expand any directories
        sources_expanded = []
        for source in sources:
            if os.path.isdir(source):
                sources_expanded.extend([os.path.join(source, file) for file in
                                        os.listdir(source)])
            else:
                sources_expanded.append(source)

        for source in sources_expanded:

            if isinstance(source, ROI):
                self.structs.append(source)
                continue

            if os.path.basename(source).startswith('.') or source.endswith('.txt'):
                continue
            if os.path.isdir(source):
                continue

            # Attempt to load from dicom
            structs = load_structs_dicom(source)
            if len(structs):
                for struct in structs.values():
                    self.structs.append(ROI(
                        struct['contours'],
                        name=struct['name'],
                        color=struct['color'],
                        image=self.image
                    ))

            # Load from struct mask
            else:
                try:
                    self.structs.append(ROI(
                        source, image=self.image
                    ))
                except RuntimeError:
                    continue

        self.rename_structs()
        self.filter_structs()
        self.loaded = True

    def reset(self):
        '''Reload structures from sources.'''

        self.structs = []
        self.loaded = False
        self.load(force=True)

    def set_image(self, image):
        '''Set image for self and all structures.'''

        if image and not isinstance(image, Image):
            image = Image(image)

        self.image = image
        for s in self.structs:
            s.image = image

    def rename_structs(self, names=None, first_match_only=True,
                       keep_renamed_only=False):
        '''Rename structures if a naming dictionary is given. If 
        <first_match_only> is True, only the first structure matching the 
        possible matches will be renamed.'''

        if names is None:
            names = self.names
        if not names:
            return

        # Loop through each new name
        already_renamed = []
        for name, matches in names.items():

            if not is_list(matches):
                matches = [matches]

            # Loop through all possible original names
            name_matched = False
            for m in matches:

                # Loop through structures and see if there's a match
                for i, s in enumerate(self.structs):

                    # Don't rename a structure more than once
                    if i in already_renamed:
                        continue

                    if fnmatch.fnmatch(s.name.lower(), m.lower()):
                        s.name = name
                        name_matched = True
                        already_renamed.append(i)
                        if first_match_only:
                            break

                # If first_match_only, don't rename more than one structure
                # with this new name
                if name_matched and first_match_only:
                    break

        # Keep only the renamed structs
        if keep_renamed_only:
            renamed_structs = [self.structs[i] for i in already_renamed]
            self.structs = renamed_structs

    def filter_structs(self, to_keep=None, to_remove=None):
        '''Keep only structs in the to_keep list, and remove any in the 
        to_remove list.'''

        if to_keep is None:
            to_keep = self.to_keep
        elif not is_list(to_keep):
            to_keep = [to_keep]
        if to_remove is None:
            to_remove = self.to_remove
        elif not is_list(to_remove):
            to_remove = [to_remove]

        if to_keep is not None:
            keep = []
            for s in self.structs:
                if any([fnmatch.fnmatch(s.name.lower(), k.lower())
                       for k in to_keep]):
                    keep.append(s)
            self.structs = keep

        if to_remove is not None:
            keep = []
            for s in self.structs:
                if not any([fnmatch.fnmatch(s.name.lower(), r.lower())
                           for r in to_remove]):
                    keep.append(s)
            self.structs = keep

    def add_structs(self, sources):
        '''Add additional structures from sources.'''

        if not is_list(sources):
            sources = [sources]
        self.sources.extend(sources)
        self.load_structs(sources)

    def add_struct(self, source, **kwargs):
        '''Add a single structure with  optional kwargs.'''

        self.sources.append(source)
        if isinstance(source, ROI):
            roi = source
        else:
            roi = ROI(source, **kwargs)
        self.structs.append(roi)

    def copy(self, name=None, names=None, to_keep=None, to_remove=None,
             keep_renamed_only=False):
        '''Create a copy of this structure set, with structures optionally
        renamed or filtered.'''

        if not hasattr(self, 'n_copies'):
            self.n_copies = 1
        else:
            self.n_copies += 1
        if name is None:
            name = f'{self.name} (copy {self.n_copies})'

        ss = RtStruct(self.sources, name=name, image=self.image, 
                          load=False, names=names, to_keep=to_keep, 
                          to_remove=to_remove)
        if self.loaded:
            ss.structs = self.structs
            ss.loaded = True
            ss.rename_structs(names, keep_renamed_only=keep_renamed_only)
            ss.filter_structs(to_keep, to_remove)

        return ss

    def get_structs(self):
        '''Get list of ROI objects.'''

        self.load()
        return self.structs

    def get_struct_names(self):
        '''Get list of names of structures.'''

        self.load()
        return [s.name for s in self.structs]

    def get_struct_dict(self):
        '''Get dict of structure names and ROI objects.'''

        self.load()
        return {s.name: s for s in self.structs}

    def get_struct(self, name):
        '''Get a structure with a specific name.'''

        structs = self.get_struct_dict()
        if name not in structs:
            print(f'Structure {name} not found!')
            return
        return structs[name]

    def print_structs(self):

        self.load()
        print('\n'.join(self.get_struct_names()))

    def __repr__(self):

        self.load()
        out_str = 'RtStruct\n{'
        out_str += '\n  name : ' + str(self.name)
        out_str += '\n  structs :\n    '
        out_str += '\n    '.join(self.get_struct_names())
        out_str += '\n}'
        return out_str

    def get_geometry(self, **kwargs):
        '''Get pandas DataFrame of geometric properties for all structures.'''

        return pd.concat([s.get_geometry(**kwargs) for s in self.get_structs()])

    def get_comparison(self, other=None, method=None, **kwargs):
        '''Get pandas DataFrame of comparison metrics vs a single ROI or 
        another RtStruct.'''

        dfs = []
        if isinstance(other, ROI):
            dfs = [s.get_comparison(other, **kwargs) for s in self.get_structs()]

        elif isinstance(other, RtStruct) or other is None:
            pairs = self.get_comparison_pairs(other, method)
            dfs = []
            for roi1, roi2 in pairs:
                dfs.append(roi1.get_comparison(roi2, **kwargs))

        else:
            raise TypeError('<other> must be ROI or RtStruct!')

        return pd.concat(dfs)

    def get_comparison_pairs(self, other=None, method=None):
        '''Get list of ROIs to compare with one another.'''

        if other is None:
            other = self
            if method is None:
                method = 'diff'
        elif method is None:
            method = 'auto'

        # Check for name matches
        matches = []
        if method in ['auto', 'named']:
            matches = [s for s in self.get_struct_names() if s in
                       other.get_struct_names()]
            if len(matches) or method == 'named':
                return [(self.get_struct(name), other.get_struct(name))
                        for name in matches]

        # Otherwise, pair each structure with every other
        pairs = []
        for roi1 in self.get_structs():
            for roi2 in other.get_structs():
                pairs.append((roi1, roi2))

        # Remove matching names if needed
        if method == 'diff':
            pairs = [p for p in pairs if p[0].name != p[1].name]

        return pairs

    def plot_comparisons(self, other=None, method=None, outdir=None, 
                         legend=True, **kwargs):
        '''Plot comparison pairs.'''

        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        for roi1, roi2 in self.get_comparison_pairs(other, method):

            outname=None
            if outdir:
                comp_name = roi1.get_comparison_name(roi2, True)
                outname = os.path.join(outdir, f'{comp_name}.png')

            names = None
            if roi1.name == roi2.name:
                names = [self.name, other.name]

            roi1.plot_comparison(roi2, legend=legend, save_as=outname,
                                 names=names, **kwargs)

    def plot_surface_distances(self, other, outdir=None, signed=False, 
                               method='auto', **kwargs):
        '''Plot surface distances for all ROI pairs.'''

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for roi1, roi2 in self.get_comparison_pairs(other, method):
            comp_name = roi1.get_comparison_name(roi2, True)
            if outdir:
                outname = os.path.join(outdir, f'{comp_name}.png')
            else:
                outname = None
            roi1.plot_surface_distances(roi2, signed=signed, save_as=outname,
                                        **kwargs)

    def write(self, outname=None, outdir='.', ext=None, overwrite=False, 
              **kwargs):
        '''Write to a dicom RtStruct file or directory of nifti files.'''

        if ext is not None and not ext.startswith('.'):
            ext = f'.{ext}'

        # Check whether to write to dicom file
        if isinstance(outname, str) and outname.endswith('.dcm'):
            ext = '.dcm'
            outname = os.path.join(outdir, outname)
        
        if ext == '.dcm':
            if outname is None:
                outname = f'{outdir}/{self.name}.dcm'
            print('Warning: dicom writing not yet available!')
            return

        # Otherwise, write to individual structure files
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif overwrite:
            shutil.rmtree(outdir)
            os.mkdir(outdir)
        for s in self.get_structs():
            s.write(outdir=outdir, ext=ext, **kwargs)


def load_structs_dicom(path, names=None):
    '''Load structure(s) from a dicom structure file. <name> can be a single
    name or list of names of structures to load.'''

    # Load dicom object
    try:
        ds = pydicom.read_file(path)
    except pydicom.errors.InvalidDicomError:
        return []
    if not (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'):
        print(f'Warning: {path} is not a DICOM structure set file!')
        return

    # Get struture names
    seq = get_dicom_sequence(ds, 'StructureSetROI')
    structs = {}
    for struct in seq:
        structs[int(struct.ROINumber)] = {'name': struct.ROIName}

    # Find structures matching requested names
    names_to_load = None
    if isinstance(names, str):
        names_to_load = [names]
    elif is_list(names):
        names_to_load = names
    if names_to_load:
        structs = {i: s for i, s in structs.items() if 
                   any([fnmatch.fnmatch(s['name'].lower(), n.lower()) 
                        for n in names_to_load]
                      )
                  }
        if not len(structs):
            print(f'Warning: no structures found matching name(s): {names}')
            return

    # Get structure details
    roi_seq = get_dicom_sequence(ds, 'ROIContour')
    for roi in roi_seq:

        number = roi.ReferencedROINumber
        if number not in structs:
            continue
        data = {'contours': {}}

        # Get structure colour
        if 'ROIDisplayColor' in roi:
            data['color'] = [int(c) / 255 for c in list(roi.ROIDisplayColor)]
        else:
            data['color'] = None

        # Get contours
        contour_seq = get_dicom_sequence(roi, 'Contour')
        if contour_seq:
            contour_data = {}
            for c in contour_seq:
                plane_data = [
                    [float(p) for p in c.ContourData[i * 3: i * 3 + 3]]
                    for i in range(c.NumberOfContourPoints)
                ]
                z = float(c.ContourData[2])
                if z not in data['contours']:
                    data['contours'][z] = []
                data['contours'][z].append(np.array(plane_data))

        structs[number].update(data)

    return structs


def get_dicom_sequence(ds=None, basename=''):

    sequence = []
    for suffix in ['Sequence', 's']:
        attribute = f'{basename}{suffix}'
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break
    return sequence


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
            if (ds.StudyInstanceUID != study_uid 
                or ds.SeriesNumber != series_num
                or ds.Modality != modality
                or ds.ImageOrientationPatient != orientation
               ):
                continue
            if not hasattr(ds, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

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
        slice_thickness = (sorted_slices[-1] - sorted_slices[0]) \
                / (len(sorted_slices) - 1)

    # Apply rescaling
    if rescale_slope:
        data = data * rescale_slope
    if rescale_intercept:
        data = data + rescale_intercept

    # Make affine matrix
    zmin = sorted_slices[0]
    zmax = sorted_slices[-1]
    n = len(sorted_slices)
    affine =  np.array([
        [
            orient[0, 0] * pixel_size[0], 
            orient[1, 0] * pixel_size[1],
            (image_position[zmax][0] - image_position[zmin][0]) / (n - 1),
            image_position[zmin][0]
        ],
        [
            orient[0, 1] * pixel_size[0], 
            orient[1, 1] * pixel_size[1],
            (image_position[zmax][1] - image_position[zmin][1]) / (n - 1),
            image_position[zmin][1]
        ],
        [
            orient[0, 2] * pixel_size[0], 
            orient[1, 2] * pixel_size[1],
            (image_position[zmax][2] - image_position[zmin][2]) / (n - 1),
            image_position[zmin][2]
        ],
        [0, 0, 0, 1]
    ])

    return data, affine, window_centre, window_width


def load_npy(path):
    '''Load a numpy array from a .npy file.'''

    try:
        data = np.load(path)
        return data

    except (IOError, ValueError):
        return


def is_list(var):
    '''Check whether a variable is a list, tuple, or array.'''

    is_a_list = False
    for t in [list, tuple, np.ndarray]:
        if isinstance(var, t):
            is_a_list = True
    return is_a_list


def downsample(data, dx=None, dy=None, dz=None):
    '''Downsample an array by the factors specified in <dx>, <dy>, and <dz>.
    '''

    if dx is None:
        dx = 1
    if dy is None:
        dy = 1
    if dx is None:
        dz = 1

    return data[::round(dy), ::round(dx), ::round(dz)]


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
    root_uid=None
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
        xy_slice_init = xy_slice.copy()
        xy_slice = ((xy_slice - intercept) / slope)
        xy_slice = xy_slice.astype(np.uint16)
        ds.PixelData = xy_slice.tobytes()
        ds.SliceLocation = pos
        ds.ImagePositionPatient[2] = pos
        outname = f'{outdir}/{sl}.dcm'
        ds.save_as(outname)


def create_dicom(orientation=None, patient_id=None, modality=None, 
                 root_uid=None):
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
        print('Warning: using generic root UID 1.2.3.4. You should use a root '
              'UID unique to your institution. A unique root ID can be '
              'obtained free of charge from: '
              'https://www.medicalconnections.co.uk/FreeUID/')
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


def to_three(val):
    '''Ensure that a value is a list of three items.'''

    if val is None:
        return None
    if is_list(val):
        if not len(val) == 3:
            print(f'Warning: {val} should be a list containing 3 items!')
        return val
    elif not is_list(val):
        return [val, val, val]


class RtDose(MachineObject):

    def __init__(self, path=''):

        MachineObject.__init__(self, path)

        if not os.path.exists(path):
            return

        ds = pydicom.read_file(path, force=True)

        # Get dose summation type
        try:
            self.summation_type = ds.DoseSummationType
        except AttributeError:
            self.summation_type = None

        # Get slice thickness
        if ds.SliceThickness:
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = None

        # Get scan position and voxel sizes
        if ds.GridFrameOffsetVector[-1] > ds.GridFrameOffsetVector[0]:
            self.reverse = False
            self.scan_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[0]),
            )
        else:
            self.reverse = True
            self.scan_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[-1]),
            )
        self.voxel_size = (
            float(ds.PixelSpacing[0]),
            float(ds.PixelSpacing[1]),
            slice_thickness,
        )
        self.transform_ijk_to_xyz = get_transform_ijk_to_xyz(self)
        self.image_stack = None

    def get_image_stack(self, rescale=True, renew=False):

        if self.image_stack is not None and not renew:
            return self.image_stack

        # Load dose array from dicom
        ds = pydicom.read_file(self.path, force=True)
        self.image_stack = np.transpose(ds.pixel_array, (1, 2, 0))

        # Rescale voxel values
        if rescale:
            try:
                rescale_intercept = ds.RescaleIntercept
            except AttributeError:
                rescale_intercept = 0
            self.image_stack = self.image_stack * float(ds.DoseGridScaling) \
                    + float(rescale_intercept)

        if self.reverse:
            self.image_stack[:, :, :] = self.image_stack[:, :, ::-1]

        return self.image_stack


class RtPlan(MachineObject):

    def __init__(self, path=''):

        MachineObject.__init__(self, path)

        ds = pydicom.read_file(path, force=True)

        try:
            self.approval_status = ds.ApprovalStatus
        except AttributeError:
            self.approval_status = None

        try:
            self.n_fraction_group = len(ds.FractionGroupSequence)
        except AttributeError:
            self.n_fraction_group = None

        try:
            self.n_beam_seq = len(ds.BeamSequence)
        except AttributeError:
            self.n_beam_seq = None

        self.n_fraction = None
        self.target_dose = None
        if self.n_fraction_group is not None:
            self.n_fraction = 0
            for fraction in ds.FractionGroupSequence:
                self.n_fraction += fraction.NumberOfFractionsPlanned
                if hasattr(fraction, 'ReferencedDoseReferenceSequence'):
                    if self.target_dose is None:
                        self.target_dose = 0.0
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        self.target_dose += dose.TargetPrescriptionDose


class Patient(PathObject):
    '''Object associated with a top-level directory whose name corresponds to
    a patient ID, and whose subdirectories contain studies.'''

    def __init__(self, path=None, exclude=['logfiles']):

        start = time.time()

        # Set path and patient ID
        if path is None:
            path = os.getcwd()
        self.path = fullpath(path)
        self.id = os.path.basename(self.path)

        # Find studies
        self.studies = self.get_dated_objects(dtype='Study')
        if not self.studies:
            if os.path.isdir(self.path):
                if os.access(self.path, os.R_OK):
                    subdirs = sorted(os.listdir(self.path))
                    for subdir in subdirs:
                        if subdir not in exclude:
                            self.studies.extend(
                                self.get_dated_objects(
                                    dtype='Study', subdir=subdir
                                )
                            )

    def combined_files(self, dtype, min_date=None, max_date=None):
        '''Get list of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified.'''

        files = []
        for study in self.studies:
            objs = getattr(study, dtype)
            for obj in objs:
                for file in obj.files:
                    if file.in_date_interval(min_date, max_date):
                        files.append(file)
        files.sort()
        return files

    def combined_files_by_dir(self, dtype, min_date=None, max_date=None):
        '''Get dict of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified. The dict keys 
        will be the directories that the files are in.'''

        files = {}
        for study in self.studies:
            objs = getattr(study, dtype)
            for object in objs:
                for file in object.files:
                    if file.in_date_interval(min_date, max_date):
                        folder = os.path.dirname(fullpath(file.path))
                        if folder not in files:
                            files[folder] = []
                        files[folder].append(file)

        for folder in files:
            files[folder].sort()

        return files

    def combined_objs(self, dtype):
        '''Get list of all objects of a given data type <dtype> associated
        with this patient.'''

        all_objs = []
        for study in self.studies:
            objs = getattr(study, dtype)
            if objs:
                all_objs.extend(objs)
        all_objs.sort()
        return all_objs

    def load_demographics(self):
        '''Load a patient's birth date, age, and sex.'''

        info = {'BirthDate': None, 'Age': None, 'Sex': None}

        # Find an object from which to extract the info
        obj = None
        if self.studies:
            obj = getattr(self.studies[0], 
                          f'{self.studies[0].im_types[0].lower()}_scans')[-1]

        # Read demographic info from the object
        if obj and obj.files:
            ds = pydicom.read_file(fp=obj.files[-1].path, force=True)
            for key in info:
                for prefix in ['Patient', 'Patients']:
                    attr = f'{prefix}{key[0].upper()}{key[1:]}'
                    if hasattr(ds, attr):
                        info[key] = getattr(ds, attr)
                        break

        # Ensure sex is uppercase and single character
        if info['Sex']:
            info['Sex'] = info['Sex'][0].upper()

        # Store data
        self.age = info['Age']
        self.sex = info['Sex']
        self.birth_date = info['BirthDate']

    def get_age(self):

        self.load_demographics()
        return self.age

    def get_sex(self):

        self.load_demographics()
        return self.sex

    def get_birth_date(self):

        self.load_demographics()
        return self.birth_date

    def get_subdir_studies(self, subdir=''):
        '''Get list of studies within a given subdirectory.'''

        subdir_studies = []
        for study in self.studies:
            if subdir == study.subdir:
                subdir_studies.append(study)

        subdir_studies.sort()

        return subdir_studies

    def last_in_interval(self, dtype=None, min_date=None, max_date=None):
        '''Get the last object of a given data type <dtype> in a given
        date interval.'''

        files = self.combined_files(dtype)
        last = None
        files.reverse()
        for file in files:
            if file.in_date_interval(min_date, max_date):
                last = file
                break
        return last

    def write(
        self, 
        outdir='.', 
        ext='.nii.gz', 
        to_ignore=None, 
        overwrite=True,
        structure_set=None,
    ):
        '''Write files tree.'''

        if not ext.startswith('.'):
            ext = f'.{ext}'

        patient_dir = os.path.join(os.path.expanduser(outdir), self.id)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        elif overwrite:
            shutil.rmtree(patient_dir)
            os.mkdir(patient_dir)

        if to_ignore is None:
            to_ignore = []

        for study in self.studies:

            # Make study directory
            study_dir = os.path.join(
                patient_dir, os.path.relpath(study.path, self.path))
            if not os.path.exists(study_dir):
                os.makedirs(study_dir)

            # Loop through image types
            for im_type in study.im_types:

                if im_type in to_ignore:
                    continue

                im_type_dir = os.path.join(study_dir, im_type)
                if not os.path.exists(im_type_dir):
                    os.mkdir(im_type_dir)

                # Write all scans of this image type
                for im in getattr(study, f'{im_type.lower()}_scans'):

                    # Make directory for this scan
                    im_dir = os.path.join(
                        study_dir,
                        os.path.relpath(im.path, study.path)
                    )

                    # Write image data to nifti
                    if ext == '.dcm':
                        outname = im_dir
                    else:
                        outname = f'{im_dir}{ext}'
                    if os.path.exists(outname) and not overwrite:
                        continue
                    im.write(outname)

                    # Find structure sets to write
                    if structure_set == 'all':
                        ss_to_write = im.structs
                    elif structure_set is None:
                        ss_to_write = []
                    elif isinstance(structure_set, int):
                        ss_to_write = [im.structs[structure_set]]
                    elif is_list(structure_set):
                        ss_to_write = [im.structs[i] for i in structure_set]
                    else:
                        raise TypeError('Unrecognised structure_set option '
                                        f'{structure_set}')

                    # Write structure sets for this image
                    for ss in ss_to_write:

                        # Find path to output structure directory
                        ss_path = os.path.join(
                            study_dir,
                            os.path.relpath(ss.path, study.path)
                        ) 
                        if ext == '.dcm':
                            ss_dir = os.path.dirname(ss_path)
                        else:
                            ss_dir = ss_path.replace('.dcm', '')

                        # Ensure it exists
                        if not os.path.exists(ss_path):
                            os.makedirs(ss_path)

                        # Write dicom structure set
                        if ext == '.dcm':
                            if os.path.exists(ss_path) and not overwrite:
                                continue
                            ss.write(ss_path)

                        # Write structs to individual files
                        else:
                            ss.write(outdir=ss_dir, ext=ext)


class Study(ArchiveObject):

    def __init__(self, path=''):

        ArchiveObject.__init__(self, path, allow_dirs=True)

        special_dirs = ['RTPLAN', 'RTSTRUCT', 'RTDOSE']
        self.im_types = []
        for file in self.files:

            subdir = os.path.basename(file.path)
            if subdir in special_dirs:
                continue
            self.im_types.append(subdir)

            # Get images
            im_name = f'{subdir.lower()}_scans'
            setattr(
                self, 
                im_name,
                self.get_dated_objects(dtype='Image', subdir=subdir, 
                                       load=False)
            )

            # Get associated structs
            struct_subdir = f'RTSTRUCT/{subdir}'
            if os.path.exists(os.path.join(self.path, struct_subdir)):
                setattr(
                    self,
                    f'{subdir.lower()}_structs',
                    self.get_structs(subdir=struct_subdir, 
                                     images=getattr(self, im_name))
                )

        # Plans, dose etc: leave commented for now
        #  self.plans = self.get_plan_data(dtype='RtPlan', subdir='RTPLAN')
        #  self.doses = self.get_plan_data(
            #  dtype='RtDose',
            #  subdir='RTDOSE',
            #  exclude=['MVCT', 'CT'],
            #  images=self.ct_scans
        #  )

        # Load CT-specific RT doses
        #  self.ct_doses = self.get_plan_data(
            #  dtype='RtDose', subdir='RTDOSE/CT', images=self.ct_scans
        #  )
        #  self.ct_doses = self.correct_dose_scan_position(self.ct_doses)

    def correct_dose_scan_position(self, doses=[]):
        '''Correct for scan positions from CheckTomo being offset by one slice
        relative to scan positions.'''

        for dose in doses:
            dx, dy, dz = dose.voxel_size
            x0, y0, z0 = dose.scan_position
            dose.scan_position = (x0, y0, z0 + dz)
        return doses

    def get_machine_sublist(self, dtype='', machine='', ignore_case=True):
        '''Get list of doses or treatment plans corresponding to a specific
        machine.'''

        sublist = []
        if dtype.lower() in ['plan', 'rtplan']:
            objs = self.plans
        elif dtype.lower() in ['dose', 'rtdose']:
            objs = self.doses
        else:
            objs = []

        if ignore_case:
            for obj in objs:
                if objs.machine.lower() == machine.lower():
                    sublist.append(obj)
        else:
            for obj in objs:
                if objs.machine == machine:
                    sublist.append(object)
        return sublist

    def get_mvct_selection(self, mvct_dict={}, min_delta_hours=0.0):
        '''Get a selection of MVCT scans which were taken at least 
        <min_delta_hours> apart. <mvct_dict> is a dict where the keys are 
        patient IDs, and the paths are directory paths from which to load scans
        for that patient.'''

        # Find scans meeting the time separation requirement
        if min_delta_hours > 0:
            mvct_scans = get_time_separated_objects(
                self.mvct_scans, min_delta_hours)
        else:
            mvct_scans = self.mvct_scans

        # Find scans matching the directory requirement
        selected = []
        patient_id = self.get_patient_id()
        if patient_id in mvct_dict:

            # Get all valid directories for this patient
            valid_dirs = [fullpath(path) for path in mvct_dict[patient_id]]

            # Check for scans matching that directory requirement
            for mvct in mvct_scans:
                mvct_dir = os.path.dirname(mvct.files[-1].path)
                if fullpath(mvct_dir) in valid_dirs:
                    selected.append(mvct)

        # Otherwise, just return all scans for this patient
        else:
            selection = mvct_scans

        return selection

    def get_patient_id(self):
        patient_id = os.path.basename(os.path.dirname(self.path))
        return patient_id

    def get_plan_data(
        self, dtype='RtPlan', subdir='RTPLAN', exclude=[], images=[]
    ):
        '''Get list of RT dose or plan objects specified by dtype='RtDose' or 
        'RtPlan' <dtype>, respectively) by searching within a given directory, 
        <subdir> (or within the top level directory of this Study, if 
        <subdir> is not provided).

        Subdirectories with names in <exclude> will be ignored.

        Each dose-like object will be matched by timestamp to one of the scans 
        in <scans> (which should be a list of DatedObjects), if provided.'''

        doses = []

        # Get initial path to search
        if subdir:
            path1 = os.path.join(self.path, subdir)
        else:
            path1 = self.path

        # Look for subdirs up to two levels deep from initial dir
        subdirs = []
        if os.path.isdir(path1):

            # Search top level of dir
            path1_subdirs = os.listdir(path1)
            for item1 in path1_subdirs:

                if item1 in exclude:
                    continue
                path2 = os.path.join(path1, item1)
                n_sub_subdirs = 0

                # Search any directories in the top level dir
                if os.path.isdir(path2):
                    path2_subdirs = os.listdir(path2)
                    for item2 in path2_subdirs:
                        path3 = os.path.join(path2, item2)

                        # Search another level (subdir/item1/item2/*)
                        if os.path.isdir(path3):
                            n_sub_subdirs += 1
                            if subdir:
                                subdirs.append(os.path.join(
                                    subdir, item1, item2))
                            else:
                                subdirs.append(item1, item2)

                if not n_sub_subdirs:
                    if subdir:
                        subdirs = [os.path.join(subdir, item1)]
                    else:
                        subdirs = [item1]

                for subdir_item in subdirs:
                    doses.extend(
                        self.get_dated_objects(
                            dtype=dtype, subdir=subdir_item
                        )
                    )

        # Assign dose-specific properties
        if dtype == 'RtDose':
            new_doses = []
            for dose in doses:

                # Search for scans with matching timestamp
                timestamp = os.path.basename(os.path.dirname(dose.path))
                if scans:
                    try:
                        dose.date, dose.time = timestamp.split('_')
                        scan = get_dated_obj(scans, dose)
                        dose.machine = scan.machine
                    except BaseException:
                        scan = scans[-1]
                        dose.date = scan.date
                        dose.time = scan.time

                    dose.timestamp = f'{dose.date}_{dose.time}'
                    dose.scan = scan

                dose.couch_translation, dose.couch_rotation \
                        = get_couch_shift(dose.path)
                # WARNING!
                #     Couch translation third component (y) inverted with
                #     respect to CT scan
                # WARNING!
                new_doses.append(dose)
            doses = new_doses

        doses.sort()
        return doses

    def get_plan_dose(self):

        plan_dose = None
        dose_dict = {}

        # Group doses by summation type
        for dose in self.doses:
            if dose.summationType not in dose_dict:
                dose_dict[dose.summationType] = []
            dose_dict[dose.summationType].append(dose)
        for st in dose_dict:
            dose_dict[st].sort()

        # 'PLAN' summation type: just take the newest entry
        if 'PLAN' in dose_dict:
            plan_dose = dose_dict['PLAN'][-1]
            plan_dose.imageStack = plan_dose.getImageStack()

        else:
            
            # Get fraction froup and beam sequence
            if self.plans:
                n_frac_group = self.plans[-1].nFractionGroup
                n_beam_seq = self.plans[-1].nBeamSequence
            else:
                n_frac_group = None
                n_beam_seq = None

            # Sum over fractions
            if 'FRACTION' in dose_dict:
                if len(dose_dict['FRACTION']) == n_frac_group:
                    
                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = doseDict['FRACTION'][0]

                    # Sum fractions
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, 'FRACTION')

            # Sum over beams
            elif 'BEAM' in sum_type:
                if len(dose_dict['BEAM']) == n_beam_seq:

                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = dose_dict['BEAM'][0]

                    # Sum beams
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, 'BEAM')

        return plan_dose

    def get_structs(self, subdir='', images=[]):
        '''Make list of RtStruct objects found within a given subdir, and
        set their associated scan objects.'''

        # Find RtStruct directories associated with each scan
        groups = self.get_dated_objects(dtype='ArchiveObject', subdir=subdir)

        # Load RtStruct files for each
        structs = []
        for group in groups:

            # Find the matching Image for this group
            image = Image(None, load=False)
            image_dir = os.path.basename(group.path)
            image_found = False

            # Try matching on path
            for im in images:
                if image_dir == os.path.basename(im.path):
                    image = im
                    image_found = True
                    break

            # If no path match, try matching on timestamp
            if not image_found:
                for im in images:
                    if (group.date == im.date) and (group.time == im.time):
                        image = im
                        break

            # Find all RtStruct files inside the dir
            for file in group.files:

                # Create RtStruct
                rt_struct = RtStruct(file.path, image=image)

                # Add to Image
                image.add_structs(rt_struct)

                # Add to list of all structure sets
                structs.append(rt_struct)

        return structs

    def get_description(self):
        '''Load a study description.'''

        # Find an object from which to extract description
        obj = None
        if self.studies:
            obj = getattr(self, f'{self.im_types[0].lower()}_scans')[-1]
        description = ''
        if obj:
            if obj.files:
                scan_path = obj.files[-1].path
                ds = pydicom.read_file(fp=scan_path, force=True)
                if hasattr(ds, 'StudyDescription'):
                    description = ds.StudyDescription

        return description

    def sum_dose_plans(self, dose_dict={}, sum_type=''):
        '''Sum over doses using a given summation type.'''

        plan_dose = None
        if sum_type in dose_dict:
            dose = dose_dict[sum_type].pop()
            plan_dose = RtDose()
            plan_dose.machine = dose.machine
            plan_dose.path = dose.path
            plan_dose.subdir = dose.subdir
            plan_dose.date = dose.date
            plan_dose.time = dose.time
            plan_dose.timestamp = dose.timestamp
            plan_dose.summationType = 'PLAN'
            plan_dose.scanPosition = dose.scanPosition
            plan_dose.reverse = dose.reverse
            plan_dose.voxelSize = dose.voxelSize
            plan_dose.transform_ijk_to_xyz = dose.transform_ijk_to_xyz
            plan_dose.imageStack = dose.getImageStack()
            for dose in dose_dict[sum_type]:
                plan_dose.imageStack += dose.getImageStack()

        return plan_dose


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

    def plot_chequerboard(self, view='x-y', invert=False, n_splits=2, 
                          mpl_kwargs=None, save_as=None, **kwargs):

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
            [[1, 0] * n_splits, [0, 1] * n_splits] * n_splits, np.ones((nx, ny))
        )
        mask = mask[:im2_slice.shape[0], :im2_slice.shape[1]]

        # Plot second image
        self.ims[i1].ax.imshow(
            np.ma.masked_where(mask < 0.5, im2_slice),
            **self.ims[i2].get_mpl_kwargs(view, mpl_kwargs)
        );

        if save_as:
            self.ims[0].fig.savefig(save_as)
            plt.close()

    def plot_overlay(self, view='x-y', invert=False, opacity=0.5, 
                     mpl_kwargs=None, save_as=None, show=True, **kwargs):

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
