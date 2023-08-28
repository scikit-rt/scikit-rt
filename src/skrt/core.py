"""Scikit-rt core data classes and functions."""

from collections.abc import Iterable
import copy
import itertools
from logging import getLogger, Formatter, StreamHandler
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import sys
import time
import timeit
from typing import Any, List, Optional, Tuple
from types import FunctionType
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pydicom


class Defaults:
    """
    Singleton class for storing default values of parameters
    that may be used in object initialisation.

    Implementation of the singleton design pattern is based on:
    https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
    """

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __Defaults:
        def __init__(self, opts: Optional[dict] = None):
            """Define instance attributes based on opts dictionary."""

            if opts:
                for key, value in opts.items():
                    setattr(self, key, value)

        def __repr__(self):
            """Print instance attributes."""

            out = []
            for key, value in sorted(self.__dict__.items()):
                out.append(f"{key}: {value}")
            return "\n".join(out)

    def __init__(self, opts: Optional[dict] = None, reset: bool = False):
        """
        Constructor of Defaults singleton class.

        **Parameters:**

        opts: dict, default={}
            Dictionary of attribute-value pairs.

        reset: bool, default=False
            If True, delete all pre-existing instance attributes before
            adding attributes and values from opts dictionary.
            If False, don't delete pre-existing instance attributes,
            but add to them, or modify values, from opts dictionary.
        """

        if not Defaults.instance:
            Defaults.instance = Defaults.__Defaults(opts)
        else:
            if reset:
                Defaults.instance.__dict__ = {}
            if opts:
                for key, value in opts.items():
                    setattr(Defaults.instance, key, value)

    def __getattr__(self, name: str):
        """Get instance attributes."""

        return getattr(self.instance, name)

    def __setattr__(self, name: str, value: Any):
        """Set instance attributes."""

        return setattr(self.instance, name, value)

    def __repr__(self):
        """Print instance attributes."""

        return self.instance.__repr__()


# Initialise default parameter values:
Defaults()

# Depth to which recursion is performed when printing instances
# of classes that inherit from the Data class.
Defaults({"print_depth": 0})

# Severity level for event logging.
# Defined values are: 'NOTSET' (0), 'DEBUG' (10), 'INFO' (20),
# 'WARNING' (30), 'ERROR' (40), 'CRITICAL' (50)
Defaults({"log_level": "WARNING"})

# Lengths of date and time stamps.
Defaults({"len_date": 8})
Defaults({"len_time": 6})

# Define whether to call compress_user(), when printing the attributes
# of a Data object, for each string corresponding to a file path.
Defaults({"compress_user": False})

# Initialise default mappings between identifiers and names
# of imaging stations.
Defaults({"stations": {}})

# Initialise location of MATLAB application.
# If True, location is taken to be as specified by environment setup.
Defaults({"matlab_app": True})

# Initialise location of MATLAB runtime installation.
# If None, MATLAB runtime is assumed not to be installed.
Defaults({"matlab_runtime": None})


class Data:
    """
    Base class for objects serving as data containers.
    An object has user-defined data attributes, which may include
    other Data objects and lists of Data objects.

    The class provides for printing attribute values recursively, to
    a chosen depth, and for obtaining nested dictionaries of
    attributes and values.
    """

    def __init__(self, opts: Optional[dict] = None, **kwargs):
        """
        Constructor of Data class, allowing initialisation of an
        arbitrary set of attributes.

        **Parameters:**

        opts: dict, default={}
            Dictionary to be used in setting instance attributes
            (dictionary keys) and their initial values.

        `**`kwargs
            Keyword-value pairs to be used in setting instance attributes
            and their initial values.
        """

        self.print_depth = None

        if opts:
            for key, value in opts.items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """
        Create string recursively listing attributes and values.
        """

        depth = self.get_print_depth()
        out = [f"\n{self.__class__.__name__}", "{"]

        # Loop over attributes, with different treatment
        # depending on whether attribute value is a list.
        # Where an attribute value of list item is
        # an instance of Data or a subclass
        # it's string representation is obtained by calling
        # the instance's __repr__() method with depth decreased
        # by 1, or (depth less than 1) is the class representation.
        for key in sorted(self.__dict__):
            # Ignore private attributes
            if key.startswith("_"):
                continue

            item = self.__dict__[key]

            # Handle printing of numpy arrays
            if isinstance(item, np.ndarray):
                value_string = f"{item.shape} array"

            # Handle printing of lists
            elif isinstance(item, list):
                items = item
                n_item = len(items)
                if n_item:
                    if depth > 0:
                        value_string = "["
                        for i, item in enumerate(items):
                            try:
                                item_string = item.__repr__(depth=depth - 1)
                            except TypeError:
                                item_string = self.get_item_string(item)
                            comma = "," if (i + 1 < n_item) else ""
                            value_string = (
                                f"{value_string} {item_string}{comma}"
                            )
                        value_string = f"{value_string}]"
                    else:
                        value_string = f"[{n_item} * {item[0].__class__}]"
                else:
                    value_string = "[]"

            # Handle printing of dicts
            elif isinstance(item, dict):
                items = item
                n_item = len(items)
                if n_item:
                    if depth > 0:
                        value_string = "{"
                        for i, key in enumerate(items):
                            item_string = "{key}: "
                            try:
                                item_string += item.__repr__(depth=depth - 1)
                            except TypeError:
                                item_string += self.get_item_string(item)
                            comma = "," if (i + 1 < n_item) else ""
                            value_string = (
                                f"{value_string} {item_string}{comma}"
                            )
                        value_string = f"{{{value_string}}}"
                    else:
                        value_string = (
                            f"{{{n_item} * keys of type "
                            f"{list(item.keys())[0].__class__}}}"
                        )
                else:
                    value_string = "{}"

            # Handle printing of pydicom datasets
            elif isinstance(item, pydicom.dataset.FileDataset):
                value_string = str(item.__class__)

            # Handle printing of nested Data objects
            else:
                if issubclass(item.__class__, Data):
                    if depth > 0:
                        value_string = item.__repr__(depth=depth - 1)
                    else:
                        value_string = f"{item.__class__}"
                else:
                    value_string = self.get_item_string(item)
            out.append(f"  {key}: {value_string} ")

        out.append("}")
        return "\n".join(out)

    def get_item_string(self, item):
        """
        Return string representation of an item.

        If Defaults().compress_user is set to True, the function
        compress_user() is called for each item corresponding to a file
        path, where the file or its parent directory exists.
        """
        if (
            Defaults().compress_user
            and isinstance(item, (str, Path))
            and (
                os.path.exists(str(item))
                or os.path.exists(os.path.dirname(str(item)))
            )
        ):
            return compress_user(item)
        return repr(item)

    def clone(self, **kwargs):
        """
        Return a clone of the Data object. All attributes of the original
        object will be copied by reference to the new object, with some
        exceptions (see parameters below).


        **Parameters:**

        data_types_to_copy : list, default=None
            List of types inherting from the Data class.
            Any objects of the types in this list that are either directly
            stored as an attribute or stored in a list or dict attribute will
            be cloned, rather than assigning the same object as to an
            attribute of the cloned parent object.
            (Note that these child Data object will be copied with
            data_types_to_copy=None to prevent recursion.)

        copy_data : bool, default=True
            If True, any lists, dicts, and numpy arrays will be shallow
            copied rather than copied by reference.
        """

        clone = copy.copy(self)
        self.clone_attrs(clone, **kwargs)
        return clone

    def clone_attrs(self, obj, data_types_to_copy=None, copy_data=True):
        """
        Assign all attributes of <self> to another object, <obj>.


        **Parameters:**

        obj : object
            Object to which attributes of <self> will be copied.

        data_types_to_copy : list, default=None
            List of types inherting from the Data class.
            Any objects of the types in this list that are either directly
            stored as an attribute or stored in a list or dict attribute will
            be cloned, rather than assigning the same object as to an
            attribute of the cloned parent object.
            (Note that these child Data object will be copied with
            data_types_to_copy=None to prevent recursion.)

        copy_data : bool, default=True
            If True, any lists, dicts, and numpy arrays will be shallow
            copied rather than copied by reference.
        """

        # Check the data types to copy are valid
        dtypes_valid = []
        if data_types_to_copy is not None:
            for dtype in data_types_to_copy:
                if issubclass(dtype, Data):
                    dtypes_valid.append(dtype)
                else:
                    print(
                        "Warning: data_types_to_copy must inherit from "
                        "skrt.Data! Type",
                        dtype,
                        "will be ignored.",
                    )

        for attr_name in dir(self):
            # Don't copy private variables
            if attr_name.startswith("__"):
                continue

            # Don't copy methodms
            attr = getattr(self, attr_name)
            if callable(attr):
                continue

            # Make new copy of lists/dicts/arrays
            if copy_data and type(attr) in [dict, list, np.ndarray]:
                setattr(obj, attr_name, copy.copy(attr))

                # Also clone given Data types
                if isinstance(attr, list):
                    for i, item in enumerate(attr):
                        for dtype in dtypes_valid:
                            if isinstance(item, dtype):
                                getattr(obj, attr_name)[i] = item.clone()
                                break
                elif isinstance(attr, dict):
                    for key, item in attr.items():
                        for dtype in dtypes_valid:
                            if isinstance(item, dtype):
                                getattr(obj, attr_name)[key] = item.clone()
                                break

            # Clone any owned Data objects if cloning children
            elif issubclass(type(attr), Data):
                for dtype in dtypes_valid:
                    if isinstance(attr, dtype):
                        setattr(obj, attr_name, attr.clone())
                        break
                if not hasattr(obj, attr_name):
                    setattr(obj, attr_name, attr)

            # Otherwise, copy reference to attribute
            else:
                setattr(obj, attr_name, attr)

    def get_dict(self) -> dict:
        """
        Return a nested dictionary of object attributes (dictionary keys)
        and their values.
        """

        objects = {}
        for key, value in self.__dict__.items():
            try:
                objects[key] = value.get_dict()
            except AttributeError:
                objects[key] = value

        return objects

    def get_print_depth(self) -> int:
        """
        Retrieve the value of the object's print depth,
        setting an initial value if not previously defined.
        """

        if self.print_depth is None:
            self.set_print_depth()
        return self.print_depth

    def print(self, depth: Optional[int] = None):
        """
        Convenience method for recursively printing
        object attributes and values, with recursion
        to a specified depth.

        **Parameters:**

        depth: integer/None, default=None
            Depth to which recursion is performed.
            If None, the value returned by self's get_print_depth() method
            is used.
        """

        if depth:
            default_depth = self.get_print_depth()
            self.set_print_depth(depth)

        print(repr(self))

        if depth:
            self.set_print_depth(default_depth)

    def set_print_depth(self, depth: Optional[int] = None):
        """
        Set the object's print depth.

        **Parameters:**

        depth: integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, the object's print depth is
            set to the value of Defaults().print_depth.
        """

        if depth is None:
            depth = Defaults().print_depth
        self.print_depth = depth


class DicomFile(Data):
    """
    Class representing files in DICOM format, with conversion to skrt objects.

    **Methods:**
    - **_init__()** : Create instance of DicomFile class.
    - **get_object()** : Instantiate object of specified Scikit-rt class.
    - **get_matched_attributes()** : Identify objects with matched attributes.
    - **set_dates_and_times()** : Store timing for specified elements.
    - **set_referenced_sop_instance_uids** : Store UIDs of referenced objects.
    - **set_slice_thickness** : Store slice thickness for 3D imaging data.
    """

    def __init__(self, path):
        """
        Create instance of DicomFile class.

        **Parameter:**

        path : str/pathlib.Path
            Relative or absolute path to DICOM file.
        """
        # Perform base-class initialisation.
        super().__init__()

        # Store absolute file path as string.
        self.path = fullpath(path)

        # Initialise time-related properties.
        self.study_date = None
        self.study_time = None
        self.study_timestamp = None
        self.item_date = None
        self.item_time = None
        self.item_timestamp = None

        # Attempt to read DICOM dataset.
        self.ds = None
        if not os.path.isdir(self.path):
            try:
                self.ds = pydicom.dcmread(self.path, force=True)
            except IsADirectoryError:
                pass

        # Try to protect against cases where file read isn't a DICOM file.
        if (
            not isinstance(self.ds, pydicom.dataset.FileDataset)
            or len(self.ds) < 2
        ):
            self.ds = None

        # Define prefixes of dataset attributes to be used
        # to set study and item timestamps.
        elements = {
            "study": ["Study", "Content", "InstanceCreation"],
            "item": ["Instance", "Content", "Series", "InstanceCreation"],
        }

        # Set object attributes.
        self.set_dates_and_times(elements)
        self.set_slice_thickness()
        self.acquisition_number = getattr(self.ds, "AcquisitionNumber", None)
        self.frame_of_reference_uid = getattr(
            self.ds, "FrameOfReferenceUID", None
        )
        self.modality = getattr(self.ds, "Modality", "unknown")
        if self.modality:
            self.modality = self.modality.lower()
        self.series_number = getattr(self.ds, "SeriesNumber", None)
        self.series_instance_uid = getattr(self.ds, "SeriesInstanceUID", None)
        self.sop_instance_uid = getattr(self.ds, "SOPInstanceUID", None)
        self.study_instance_uid = getattr(self.ds, "StudyInstanceUID", None)
        self.set_referenced_sop_instance_uids()

    def get_object(self, cls, **kwargs):
        """
        Instantiate object of specified Scikit-rt class.

        **Parameters:**

        cls : Class
            Class from which object is to be instantiated.  In principle
            this could be an arbitrary class, but this method is intended
            for use with:
            - skrt.dose.Dose;
            - skrt.dose.Plan;
            - skrt.image.Image;
            - skrt.patient.Study;
            - skrt.structures.StructureSet.

        **kwrgs
            Keyword arguments to be passed to contstructor of the class
            from which object is to be instantiated.
        """
        # Set class-specific keyword arguments.
        if cls.__name__ in ["Dose", "Plan", "StructureSet"]:
            kwargs["path"] = self.path
            kwargs["load"] = False

        # Instantiate class.
        obj = cls(**kwargs)

        # Set attributes specific to Study and non-Study objects.
        if cls.__name__ == "Study":
            obj.date = self.study_date
            obj.time = self.study_time
            obj.timestamp = self.study_timestamp
            for item in ["dose", "image", "plan", "structure_set"]:
                setattr(obj, f"{item}_types", {})
        else:
            obj.date = self.item_date
            obj.time = self.item_time
            obj.timestamp = self.item_timestamp

        # Set attributes for all.
        # Some values may be null, depending on object instantiated.
        obj.modality = self.modality
        obj.frame_of_reference_uid = self.frame_of_reference_uid
        obj.referenced_image_sop_instance_uid = (
            self.referenced_image_sop_instance_uid
        )
        obj.referenced_plan_sop_instance_uid = (
            self.referenced_plan_sop_instance_uid
        )
        obj.referenced_structure_set_sop_instance_uid = (
            self.referenced_structure_set_sop_instance_uid
        )
        obj.series_instance_uid = self.series_instance_uid
        obj.series_number = self.series_number
        obj.sop_instance_uid = self.sop_instance_uid
        obj.study_instance_uid = self.study_instance_uid

        return obj

    def get_matched_attributes(self, others, attributes=None):
        """
        Identify objects with attribute values matching own attribute values.

        **Parameters:**

        others : list
            List of objects to be checked for matches.

        attributes : list/str, default=None
            List of strings corresponding to attributes whose values
            are to be checked.  If passed as a string, this is converted
            to a single-element list.
        """
        # Initialise list of attributes to check.
        attributes = attributes or []
        if isinstance(attributes, str):
            attributes = [attributes]

        # Initialise list of matches to all input objects.
        matches = others or []

        # Loop over attributes, filtering list of matches at each iteration.
        for attribute in attributes:
            matches = [
                match
                for match in matches
                if (
                    getattr(match, attribute, None)
                    == getattr(self, attribute, None)
                )
            ]

        return matches

    def set_dates_and_times(self, elements):
        """
        Store timing information for specified elements.

        **Parameter:**

        elements: dict
            Dictionary where keys are element names - for example,
            "study", "item" - and values are DICOM dataset attributes
            to be checked, in the order listed, for date and time information.
        """
        # Define time measurements of interest, and values to be used
        # in case of missing information.
        measurements = {"Date": "00000000", "Time": "000000"}

        # Set time-measurement values for each element.
        for element, timed_items in elements.items():
            for measurement, value in measurements.items():
                # Initialise time-measurement attribute
                # to None (no dataset available) or fallback value.
                attribute = f"{element}_{measurement.lower()}"
                if self.ds is None:
                    setattr(self, attribute, None)
                    continue
                setattr(self, attribute, value)

                # Reset time-measurement attribute to value of
                # first target attribute found in DICOM dataset.
                for timed_item in timed_items:
                    target = f"{timed_item}{measurement}"
                    if hasattr(self.ds, target):
                        target_value = getattr(self.ds, target)
                        if target_value:
                            setattr(self, attribute, target_value.split(".")[0])
                            break

            # Set timestamp.
            if self.ds is not None:
                date_part = getattr(self, f"{element}_date")
                time_part = getattr(self, f"{element}_time")
                setattr(
                    self, f"{element}_timestamp", f"{date_part}_{time_part}"
                )

    def set_referenced_sop_instance_uids(self):
        """
        Store SOP instance UIDs for referenced image, structure set, and plan.
        """

        # Store SOP instance UID for referenced image.
        try:
            uid = self.ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
        except (IndexError, AttributeError):
            uid = None

        if uid is None:
            try:
                uid = (
                    self.ds.ROIContourSequence[0]
                    .ContourSequence[0]
                    .ContourImageSequence[0]
                    .ReferencedSOPInstanceUID
                )
            except (IndexError, AttributeError):
                pass

        self.referenced_image_sop_instance_uid = uid

        # Store SOP instance UID for referenced structure set.
        try:
            uid = self.ds.ReferencedStructureSetSequence[
                0
            ].ReferencedSOPInstanceUID
        except (IndexError, AttributeError):
            pass

        self.referenced_structure_set_sop_instance_uid = uid

        # Store SOP instance UID for referenced plan.
        try:
            uid = self.ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
        except (IndexError, AttributeError):
            uid = None

        self.referenced_plan_sop_instance_uid = uid

    def set_slice_thickness(self):
        """
        Store slice thickness for 3D imaging data.
        """
        # Initialise slice thickness.
        self.slice_thickness = ""

        # Set slice thickness from DICOM dataset.
        if hasattr(self.ds, "SliceThickness"):
            slice_thickness = self.ds.SliceThickness
            if slice_thickness:
                self.slice_thickness = f"{self.ds.SliceThickness : .3f}"


class PathData(Data):
    """Data with an associated path or directory; has the ability to
    extract a list of dated objects from within this directory."""

    def __init__(self, path=""):
        # Perform base-class initialisation.
        super().__init__()

        self.path = fullpath(str(path))
        self.subdir = ""

    def create_objects(
        self, dtype: type, subdir: str = "", timestamp_only=True, **kwargs
    ) -> List[Any]:
        """
        For all the files inside own directory, or own directory + <subdir>
        if <subdir> is given, create an object of given data type <dtype> if
        the filename corresponds to a timestamp. Return the created objects in
        a list.

        **Parameters**:

        dtype : type
            Type of object to create from files in the specified directory.

        subdir : str, default=""
            Subdirectory from which to take files. If empty, own top-level
            directory will be used.

        timestamp_only : bool, default=True
            If True, only files whose names correspond to a timestamp will
            be used to initialise objects.

        `**`kwargs :
            Keyword arguments to pass to object creation.

        **Returns**:

        objs : list
            List of created objects of type <dtype>.
        """

        # Attempt to create object for each file in the subdir
        objs = []
        path = os.path.join(self.path, subdir)
        if os.path.isdir(path):
            for filename in os.listdir(path):
                # Ignore files with no timestamp in name
                if timestamp_only and not is_timestamp(filename):
                    continue

                # Attempt to initialise object of type <dtype>
                filepath = os.path.join(path, filename)
                try:
                    objs.append(dtype(path=filepath, **kwargs))
                except RuntimeError:
                    pass

        # Sort and assign subdir to the created objects
        objs.sort()
        if subdir:
            for obj in objs:
                obj.subdir = subdir

        return objs

    def get_file_size(self):
        """
        Return size in bytes of associated file.
        """
        return Path(self.path).stat().st_size if Path(self.path).exists() else 0

    def get_n_file(self):
        """
        Return number of data files associated with this object.
        """
        # Only 1 data file associated with a non-Archive object.
        return 1

    def get_n_file_below(self):
        """
        Where self.path is a directory, return number of files below this.

        Returns None if self.path isn't the path to a directory.  In
        counting files, hidden files are ignored.
        """
        return get_n_file_below(self.path)

    def print_paths(self, max_path=None):
        """
        Print paths of data files associated with this object.

        File paths are listed in natural order, with one path per line.

        **Parameters:**
        max_path: int/None, default=None
            Indication of maximum number of paths to print.  If a positive
            integer, the first <max_path> paths are printed.  If a negative
            integer, the last <max_path> paths are printed.  If None,
            all paths are printed.
        """
        if os.path.isdir(self.path):
            print_paths(self.path, max_path)
        else:
            print(self.path)


class Dated(PathData):
    """PathData with an associated date and time, which can be used for
    sorting multiple Dateds."""

    def __init__(self, path: str = "", auto_timestamp=False):
        """
        Initialise dated object from a path and assign its timestamp. If
        no valid timestamp is found within the path string, it will be set
        automatically from the current date and time if auto_timestamp is True.
        """

        super().__init__(path)

        # Assign date and time
        timestamp = os.path.basename(self.path)
        self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(os.path.dirname(self.path))
            self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(self.path)
            try:
                self.date, self.time = timestamp.split("_")
            except ValueError:
                self.date, self.time = ("", "")

        # Set date and time from current time
        if not self.date and auto_timestamp:
            self.date = time.strftime("%Y%m%d")
            self.time = time.strftime("%H%M%S")

        # Make full timestamp string
        if not self.date and not self.time:
            self.timestamp = ""
        else:
            self.timestamp = f"{self.date}_{self.time}"

    def get_pandas_timestamp(self):
        """Obtain own timestamp as a pandas.Timestamp object."""
        try:
            timestamp = pd.Timestamp("".join([self.date, self.time]))
        except ValueError:
            timestamp = None
        return timestamp

    def in_date_interval(
        self, min_date: Optional[str] = None, max_date: Optional[str] = None
    ) -> bool:
        """Check whether own date falls within an interval."""

        if min_date:
            if self.date < min_date:
                return False
        if max_date:
            if self.date > max_date:
                return False
        return True

    def get_matched_timestamps(self, others=None, delta_seconds=120):
        """
        Identify Dated objects that have matching timestamps, within tolerance.

        **Parameters:**

        others : list, default=None
            List of objects to be checked for matching timestamps.

        delta_seconds : int/float, default=120
            Maximum time difference (seconds) between timestamps for
            the timestamps to be considered matched.
        """
        others = others or []
        matches = []

        # Loop over input objects, checking for match with self.
        for other in others:
            try:
                delta_time = abs(float(other.time) - float(self.time))
            except TypeError:
                delta_time = 0
            if other.date == self.date and delta_time <= delta_seconds:
                matches.append(other)

        return matches

    def __gt__(self, other):
        """
        Define whether <self> is greater than <other>.

        The comparison is based first on object date, then on time,
        then on path.
        """
        for attribute in ["date", "time", "path"]:
            if getattr(self, attribute) != getattr(other, attribute):
                return getattr(self, attribute) > getattr(other, attribute)
        return False

    def __ge__(self, other):
        """
        Define whether <self> is greater than, or equal to, <other>.

        The comparison is based first on object date, then on time,
        then on path.
        """
        return (self > other) or (self == other)

    def copy_dicom(self, outdir=None, overwrite=True, sort=True, index=None):
        """
        Copy (single) source dicom file.

        Derived classes may override this method, to provide for
        copying multiple files.  See, for example:
            skrt.image.Image.copy_dicom().

        **Parameters:**

        outdir : pathlib.Path/str, default=None
            Path to directory to which source file is to be copied.
            if None, this is set to be a directory with name equal
            to the modality (all upper case) of the dicom data.

        overwrite : bool, default=True
            If True, delete and recreate <outdir> before copying
            file.  If False and <outdir> exists already, copy
            file only if this doesn't mean overwriting an existing
            file.

        sort : bool, default=True
            If True, the copied dicom file will be given name of form
            'MODALITY_YYMMDD_hhmmss'.  If False, the file is copied
            with name unaltered.

        index : int, default=None
            Integer representing associated object position in an ordered
            list of objects of source data type.  Used in construction
            of output filename if not None, otherwise Ignored.
        """
        if not callable(getattr(self, "load", None)):
            raise NotImplementedError(
                f"{type(self)}.copy_dicom() failed - "
                "class has no load() method"
            )

        # Create clone for data loading.
        # Clone is deleted once data are no longer needed.
        obj = self.clone()
        obj.load()

        # Check that object has dicom file to be copied.
        path = Path(obj.path)
        dset = getattr(obj, "dicom_dataset", None)
        if not dset or not path.exists():
            raise NotImplementedError(
                f"{type(obj)}.copy_dicom() failed - "
                "object has no associated DICOM file"
            )

        # Obtain the data modality.
        modality = getattr(dset, "Modality", "unknown")

        # Define the output directory.
        outdir = make_dir(outdir or modality, overwrite)

        # Define path to the output file.
        if sort:
            idx = "" if index is None else f"_{index+1:03}"
            name = f"{modality}_{obj.timestamp}{idx}.dcm"
        else:
            name = path.name
        outpath = outdir / name

        # Copy file.
        if overwrite or not outpath.exists():
            shutil.copy2(path, outpath)

    def copy_dicom_files(
        self,
        data_type=None,
        index=None,
        indices=None,
        outdir="dicom",
        overwrite=True,
        sort=True,
    ):
        """
        Copy DICOM file(s) associated with this object.

        **Parameters:**

        data_type : str, default=None
            String indicting type of data associated with
            this object.  This may be any key of the <indices>
            dictionary.  Ignored if not a key of this dictionary,
            or if None.

        index : int, default=None
            Integer representing object position in an ordered
            list of objects of specified <data_type>.  Ignored if None.

        indices : dict, default=None
            Dictionary where the keys are data types and the values are
            lists of indices for the objects whose data is to be written.
            Ignored if None.

        outdir : pathlib.Path/str, default="dicom"
            Path to directory to which dicom files are to be copied.

        overwrite : bool, default=True
            If True, allow copied file(s) to overwrite existing file(s),
            as defined in derived class's copy_dicom() method.

        sort : bool, default=True
            If True, files are sorted for copying, as defined in derived
            class's copy_dicom() method.
        """
        # Check whether object index satisfies requirements.
        if (
            isinstance(indices, dict)
            and isinstance(data_type, str)
            and isinstance(indices.get(data_type, None), list)
            and isinstance(index, int)
            and index not in indices[data_type]
        ):
            return

        # Create output directory, and copy object files.
        outdir = make_dir(outdir, overwrite)
        self.copy_dicom(outdir, overwrite, sort, index)


class Archive(Dated):
    """Dated object associated with multiple files."""

    def __init__(
        self, path: str = "", auto_timestamp=False, allow_dirs: bool = False
    ):
        super().__init__(path, auto_timestamp)

        # Find names of files within the directory
        self.files = []
        try:
            filenames = os.listdir(self.path)
        except OSError:
            filenames = []

        for filename in filenames:
            # Disregard hidden files
            if not filename.startswith("."):
                filepath = os.path.join(self.path, filename)

                # Disregard directories unless allow_dirs is True
                if not os.path.isdir(filepath) or allow_dirs:
                    self.files.append(File(path=filepath))

        self.files.sort()

    def get_file_size(self):
        """
        Return size in bytes of associated file.
        """
        if Path(self.path).is_file():
            size = Path(self.path).stat().st_size
        else:
            size = 0
            for file in self.files:
                if Path(file.path).exists():
                    size += Path(file.path).stat().st_size

        return size

    def get_n_file(self):
        """
        Return number of data files associated with this object.
        """
        if Path(self.path).is_file():
            n_file = 1
        else:
            n_file = len(self.files)
        return n_file


class File(Dated):
    """File with an associated date. Files can be sorted based on their
    filenames."""

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return self.path != other.path

    def __lt__(self, other):
        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = int(self_name) < int(other_name)
        except (NameError, TypeError, ValueError):
            result = self.path < other.path
        return result

    def __gt__(self, other):
        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = int(self_name) > int(other_name)
        except (NameError, TypeError, ValueError):
            result = self.path > other.path
        return result


def alphanumeric(in_str: str = "") -> List[str]:
    """Function that can be passed as value for list sort() method
    to have alphanumeric (natural) sorting"""

    elements = []
    for substr in re.split("(-*[0-9]+)", str(in_str)):
        try:
            element = int(substr)
        except (TypeError, ValueError):
            element = substr
        elements.append(element)
    return elements


def fullpath(path="", pathlib=False):
    """
    Evaluate full path, expanding '~', environment variables, and
    symbolic links.

    path: str/pathlib.Path, default=""
        Path to be expanded.

    pathlib: bool, default=False
        If False, return full path as string.  If True, return full path
        as pathlib.Path object.
    """

    expanded = ""
    if path:
        path = str(path)
        tmp = os.path.expandvars(path.strip())
        tmp = os.path.abspath(os.path.expanduser(tmp))
        expanded = os.path.realpath(tmp)
        if pathlib:
            expanded = Path(expanded)
    return expanded


def compress_user(path=""):
    """If path starts with home directory, replace by '~'"""
    new_path = Path(fullpath(path))
    try:
        new_path = f"~/{new_path.relative_to(new_path.home())}"
    except ValueError:
        pass
    return str(new_path)


def qualified_name(cls=None):
    """
    Return qualified name of a class.

    Return None if non-class given as input.
    """
    if isinstance(cls, type):
        return f"{cls.__module__}.{cls.__name__}"
    return None


def get_logger(name="", log_level=None, identifier="name"):
    """
    Retrieve named event logger.

    **Parameters:**

    name: string, default=""
        Name of logger (see documentation of logging module)

    log_level: string/integer/None, default=None
        Severity level for event logging.  If the value is None,
        log_level is set to the value of Defaults().log_level.

    identifier: str, default="name"
        Attribute to use to identify logger messages.  Possibilities
        include "name", "filename", "funcName".  For a list of available
        attributes, not all of which make sense as identifiers, see:
        https://docs.python.org/3/library/logging.html#logrecord-attributes
    """
    formatter = Formatter(f"%({identifier})s - %(levelname)s - %(message)s")
    handler = StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
    log_level = log_level or Defaults().log_level
    logger.setLevel(log_level)
    return logger


def get_time_and_date(timestamp: str = "") -> Tuple[str, str]:
    """Extract time and date separately from timestamp."""

    time_date = (None, None)
    if is_timestamp(timestamp):
        items = os.path.splitext(timestamp)[0].split("_")
        items = [item.strip() for item in items]
        if items[0].isalpha():
            time_date = tuple(items[1:3])
        else:
            time_date = tuple(items[0:2])
    else:
        idx1 = timestamp.find("_")
        idx2 = timestamp.rfind(".")
        if (-1 != idx1) and (-1 != idx2):
            bitstamp = timestamp[idx1 + 1 : idx2]
            if is_timestamp(bitstamp):
                items = bitstamp.split("_")
                if len(items) > 2:
                    if (
                        items[0].isalpha()
                        and items[1].isdigit()
                        and items[2].isdigit()
                    ):
                        items = items[1:3]
                    elif items[0].isdigit() and items[1].isdigit():
                        items = items[:2]
                    elif items[0].isdigit() and items[1].isdigit():
                        items = items[:2]
                if len(items) != 2:
                    time_date = tuple(items)

    return time_date


def is_timestamp(string: str = "") -> bool:
    """Check whether a string is a valid timestamp."""

    valid = True
    items = os.path.splitext(string)[0].split("_")
    items = [item.strip() for item in items]
    if len(items) > 2:
        if items[0].isalpha() and items[1].isdigit() and items[2].isdigit():
            items = items[1:3]
        elif items[0].isdigit() and items[1].isdigit():
            items = items[:2]
    if len(items) != 2:
        valid = False
    else:
        for item in items:
            if not item.isdigit():
                valid = False
                break
        if valid:
            if isinstance(Defaults().len_date, int):
                if len(items[0]) != Defaults().len_date:
                    valid = False
            if isinstance(Defaults().len_time, int):
                if len(items[1]) != Defaults().len_time:
                    valid = False

    return valid


def is_list(var: Any) -> bool:
    """Check whether a variable is a list, tuple, or array."""

    is_a_list = False
    for t in [list, tuple, np.ndarray]:
        if isinstance(var, t):
            is_a_list = True
    return is_a_list


def to_list(val, n_item=3, keep_none_single=True):
    """Ensure that a value is a list of n_item items."""

    if val is None and keep_none_single:
        return None
    if is_list(val):
        if not len(val) == n_item:
            print(f"Warning: {val} should be a list containing {n_item} items!")
        return list(val)
    return [val] * n_item


def generate_timestamp() -> str:
    """Make timestamp from the current time."""

    return time.strftime("%Y%m%d_%H%M%S")


def get_data_by_filename(
    data_objects=None, remove_timestamp=True, remove_suffix=True
):
    """
    Create dictionary of data_objects from list.

    Dictionary keys are derived from the paths to the data sources.

    **Parameters:**

    data_objects : list, default=None
        List of data objects.

    remove_timestamp : bool, default=True
        If True, remove and timestamps from filenames to be used as
        dictionary keys.

    remove_suffix : bool, default=True
        If True, remove any suffixes from filenames to be used as
        dictionary keys.
    """

    data_by_filename = {}
    if data_objects:
        idx = 0
        for data_object in data_objects:
            path = getattr(data_object, "path", "")
            if path:
                # Determine filename, removing timestamp and suffix if needed.
                filename = Path(path).name
                if remove_timestamp:
                    time_and_date = get_time_and_date(filename)
                    if None not in time_and_date:
                        timestamp = "_".join(time_and_date)
                        filename = "".join(filename.split(f"{timestamp}_"))
                if remove_suffix:
                    filename = filename.split(".")[0]
            else:
                # Create dummy filename for object with no associated path.
                idx += 1
                filename = f"unknown_{idx:03}"

            data_by_filename[filename] = data_object

    return data_by_filename


def get_n_file(objs=None):
    """
    Return number of data files associated with listed objects.

    **Parameter:**

    objs : list, default=None
        List of objects for which numbers of files are to be summed.
    """
    if not isinstance(objs, Iterable):
        objs = [objs]

    n_file = 0
    for obj in objs:
        n_file += obj.get_n_file()
    return n_file


def get_file_size(objs=None):
    """
    Return size in bytes of data files associated with listed objects.

    **Parameter:**

    objs : list, default=None
        List of objects for which file sizes are to be summed.
    """
    if not isinstance(objs, Iterable):
        objs = [objs]

    size = 0
    for obj in objs:
        size += obj.get_file_size()
    return size


def get_time_separated_objects(
    objs, min_delta=4, unit="hour", most_recent=True
):
    """
    Return ordered list of dated objects, filtering for minimum time separation.

    objs : list
        List of dated objects.

    min_delta : int/pandas.Timedelta, default=4
        Minimum time interval required between objects.  If an integer,
        the unit must be specified.

    unit : str, default='hour'
        Unit of min_delta if the latter is specified as an integer;
        ignored otherwise.  Valid units are any accepted by pandas.Timedelta:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html

    most_recent : bool, default=True
        When objects aren't separated by the minimum time interval, keep
        the most recent if True, or the least recent otherwise.
    """

    # Deal with cases where the number of input objects is 0 or 1.
    if len(objs) <= 1:
        return objs

    # Ensure that min_delta is a pandas Timedelta object.
    if not isinstance(min_delta, pd.Timedelta):
        min_delta = pd.Timedelta(min_delta, unit)

    # Only consider objects with non-null date and time.
    timed_objs = {}
    if objs:
        for obj in objs:
            if obj.date and obj.time:
                try:
                    timestamp = pd.Timestamp("".join([obj.date, obj.time]))
                except ValueError:
                    timestamp = None

                if timestamp is not None:
                    timed_objs[timestamp] = obj

    # Obtain the list of objects with time separation greater than minimum.
    time_separated_objs = []
    if timed_objs:
        # Sort time stamps,
        # reversing order if most-recent objects are to be kept.
        timestamps = sorted(timed_objs.keys(), reverse=most_recent)
        time_separated_objs = [timed_objs[timestamps[0]]]

        # Check that absolute time separation is greater than minimum.
        # (Time separations will be all positive or all negative,
        # depending on sort order.)
        for idx in range(1, len(timestamps)):
            if abs(timestamps[idx] - timestamps[idx - 1]) > min_delta:
                time_separated_objs.append(timed_objs[timestamps[idx]])

    # Ensure ascending order.
    if most_recent:
        time_separated_objs.reverse()

    return time_separated_objs


def get_hour_in_day(timestamp):
    """
    Return a timestamp's hour in day, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which hour in day is to be returned.
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    return None


def get_hour_in_week(timestamp):
    """
    Return a timestamp's hour in week, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which hour in week is to be returned.
    """
    if isinstance(timestamp, pd.Timestamp):
        return 24 * (timestamp.isoweekday() - 1) + get_hour_in_day(timestamp)
    return None


def get_day_in_week(timestamp):
    """
    Return a timestamp's day in week, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which day in week is to be returned.
    """
    if isinstance(timestamp, pd.Timestamp):
        return get_hour_in_week(timestamp) / 24
    return None


def get_interval_in_days(timestamp1, timestamp2):
    """
    Return interval in days between two timestamps.

    **Parameters:**

    timestamp1 : pandas.Timestamp
        Timestamp corresponding to start of interval.

    timestamp2 : pandas.Timestamp
        Timestamp corresponding to end of interval.
    """
    if isinstance(timestamp1, pd.Timestamp) and isinstance(
        timestamp2, pd.Timestamp
    ):
        interval = (timestamp2 - timestamp1).total_seconds()
        interval /= pd.Timedelta("1d").total_seconds()
    else:
        interval = None
    return interval


def get_interval_in_whole_days(timestamp1, timestamp2):
    """
    Return interval in whole days between two timestamps.

    **Parameters:**

    timestamp1 : pandas.Timestamp
        Timestamp corresponding to start of interval.

    timestamp2 : pandas.Timestamp
        Timestamp corresponding to end of interval.
    """
    if isinstance(timestamp1, pd.Timestamp) and isinstance(
        timestamp2, pd.Timestamp
    ):
        interval = (
            timestamp2.floor("1d") - timestamp1.floor("1d")
        ).total_seconds()
        interval /= pd.Timedelta("1d").total_seconds()
        interval = round(interval)
    else:
        interval = None
    return interval


def year_fraction(timestamp):
    """
    Convert from timestamp to year, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp to be converted.
    """
    if isinstance(timestamp, pd.Timestamp):
        # Determine year.
        year = timestamp.year

        # Determine seconds in year (different for leap year and non-leap year).
        year_start = pd.Timestamp(f"{year}0101")
        next_year_start = pd.Timestamp(f"{year + 1}0101")
        seconds_in_year = (next_year_start - year_start).total_seconds()

        # Determine seconds elapsed so far in year.
        seconds_to_date = (timestamp - year_start).total_seconds()

        # Return year with fractional part.
        return year + seconds_to_date / seconds_in_year

    return None


def get_uid_without_slice(uid):
    """Obtain copy of <uid>, truncated to before final dot."""
    return ".".join(uid.split(".")[:-1])


def get_referenced_image(referrer=None, image_types=None):
    """
    Retrieve from <image_types> image object referred to by <referrer>.

    **Parameters:**

    referrer : object, default=None
        Object that references an image object via its SOP instance UID.

    image_types : dict, default=None
        Dictionary where keys are imaging modalities and values are lists
        of image objects for this modality.
    """
    image_types = image_types or {}
    image = None

    # Search for referenced image based on matching
    # referenced image SOP instance UID.
    for images in image_types.values():
        image = get_referenced_object(
            referrer, images, "referenced_image_sop_instance_uid", True
        )
        if image:
            break

    # Search for referenced image based on matching
    # frame-of-reference UID.
    if image is None:
        for images in image_types.values():
            matched_attributes = DicomFile.get_matched_attributes(
                referrer, images, "frame_of_reference_uid"
            )
            if matched_attributes:
                image = matched_attributes[0]

    return image


def get_referenced_object(referrer, others, tag, omit_slice=False):
    """
    Retrieve from <others> object referred to via <tag> by <referrer>.

    **Parameters:**

    referrer : object
        Object that references another object via a tag.  This tag
        should be a reference to a SOP instance UID.

    others : list
        List of objects to be considered for identifying referenced object.

    tag : str
        String identifying object attribute that references a SOP instance UID.
        Valid values are those defined in
        skrt.DicomFile.set_referenced_sop_instance_uids(), namely:
        'referenced_image_sop_instance_uid';
        'referenced_structure_set_sop_instance_uid';
        'referenced_plan_sop_instance_uid'.

    omit_slice : bool, default=False
        If True, disregard the last part of all UIDs, meaning the part from
        the last dot onwards.  For imaging data, this part distinguishes
        between different slices.
    """
    referenced_object = None

    # Obtain SOP instance UID of referenced object.
    uid1 = getattr(referrer, tag, None)
    if uid1:
        # Optionally omit part of UID from last dot onwards.
        if omit_slice:
            uid1 = get_uid_without_slice(uid1)

        # Loop over list of objects.
        for other in others:
            # Obtain object's SOP instance UID.
            if omit_slice:
                uid2 = get_uid_without_slice(other.sop_instance_uid)
            else:
                uid2 = other.sop_instance_uid

            # Check for match between referenced UID and object's UID.
            if uid1 == uid2:
                referenced_object = other
                break

    return referenced_object


def get_associated_image(objs, voxel_selection="most"):
    """
    Identify an image associated with at least one of a set of objects.

    If none of the input objects has an image attribute, None is returned.

    **Parameters:**

    objs : list
        List of objects, which potentially have an image attribute,
        identifying an associated image.

    voxel_selection : str, default="most"
        Criterion used for selecting associated image is cases where there
        is more than one candidate.  Possible values are:
        - None: return first image encountered;
        - "most": return image with the highest number of voxels;
        - "least": return image with the lowest number of voxels.
    """
    # Initialise variables.
    if voxel_selection not in ["least", "most"]:
        voxel_selection = None
    associated_image = None
    associated_images = {}

    for obj in objs:
        # If no selection, return first image encountered.
        associated_image = getattr(obj, "image", None)
        if associated_image and not voxel_selection:
            break
        # Store image in dictionary, with number of voxels as key.
        if associated_image:
            n_voxel = np.prod(associated_image.get_n_voxels())
            associated_images[n_voxel] = associated_image

    # Return image based on selection criterion.
    if associated_images:
        if voxel_selection == "most":
            key = max(list(associated_images.keys()))
        else:
            key = min(list(associated_images.keys()))
        associated_image = associated_images[key]

    return associated_image


def relative_path(path, nlevel=None):
    """
    Return relative path.

    The returned path will be relative to the user's top-level directory,
    or will include a specified number of directory levels.

    **Parameter:**

    nlevel : int, default=None
        If None, indicates that the returned path should be relative to
        the user's top-level directory.  Otherwise, this specifies the
        number of directory levels to include in the returned path.
        If nlevel is positive, the top-most nlevels, up to the number
        of directory levels in the path, are omitted.  If nlevel is
        negative, the bottom-most abs(nlevels), up to the number of
        directory levels in the path, are retained.
    """
    # Return path relative to the user's top-level directory.
    if nlevel is None:
        try:
            return str(Path(path).relative_to(Path("~").expanduser()))
        except ValueError:
            pass

    # Return input path.
    if not nlevel:
        return str(path)

    # Return relative path with the number of directory levels specified.
    elements = fullpath(path).split(os.sep)
    idx = min(nlevel + 1, len(elements) - 1) if nlevel > 0 else nlevel
    return os.sep.join(elements[idx:])


def get_indexed_objs(objs, indices=True):
    """
    Select subset of objects from a list.

    **Parameters:**
    objs : list
        List of objects.

    indices : bool/list/int
        Specification of objects to select from list:

        - True: select all objects;
        - False: select no objects;
        - integer: select object with index equal to given integer;
        - list of integers: select objects with indices equal to given integers.
    """

    # Select all objects.
    if indices is True:
        indexed_objs = objs
    # Select no objects.
    elif indices is False:
        indexed_objs = []
    # Select object with specified index.
    elif isinstance(indices, int):
        indexed_objs = [objs[indices]]
    # Select objects with specified indices.
    else:
        indexed_objs = [objs[idx] for idx in indices]

    return indexed_objs


def make_dir(path=".", overwrite=True, require_empty=False):
    """
    Create a directory if it doesn't exist already, or if overwriting allowed.

    **Parameters:**

    path : pathlib.Path/str, default="."
        Path to directory.

    overwrite : bool, default=True
        If True, delete any pre-existing directory and its contents.
        If False, leave unaltered any pre-existing directory and its contents.

    require_empty : bool, default=False
        If True, return None if the directory already exists, the directory
        isn't empty, and <overwrite> is False.
    """
    # Obtain pathlib.Path object for directory.
    dirpath = Path(fullpath(path))

    if dirpath.exists():
        # Directory exists, and should be overwritten.
        if overwrite:
            shutil.rmtree(dirpath)
        # Directory exists, and isn't empty.
        elif require_empty and any(dirpath.iterdir()):
            dirpath = None

    if dirpath is not None:
        # Create directory if needed.
        dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def get_float(obj, attribute, default=None):
    """
    Return object attribute as a float.

    **Parameters:**

    obj : Object to be considered.

    attribute : str
        Name of object attribute to be returned as float.

    default : Value to be returned in case object attribute
        can't be converted to float.
    """
    value = getattr(obj, attribute, default)
    if value != default:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
    return value


def get_qualified_class_name(cls):
    """
    Determine qualified name of class <cls>.
    """
    if isinstance(cls, type):
        return f"{cls.__module__}.{cls.__name__}"
    return None


def get_subdir_paths(parent):
    """
    Return paths to a directory's sub-directories.

    **Parameter:**

    parent : pathlib.Path/str
        Path to directory for which sub-directory paths are to be found.
    """
    return sorted(
        [subdir for subdir in Path(parent).iterdir() if subdir.is_dir()]
    )


def get_data_indices(in_value, valid_data_types):
    """
    Obtain dictionary associating indices with data types.

    This is used in patient.skrt.Study.copy_dicom() to obtain
    dictionaries from input parameters that may be dictionaries,
    lists, or None.

    **Parameters:**

    in_value : dict/str/int/list/None
        Specification of indices associated with data types.
        list of data types, or None.  If a dictionary, return
        after removing any keys not in <valid_data_types>.
        If a string, return a dictionary that associates with
        it the value True (all file indices) if the string is
        in <valid_data_types>, or otherwise return an empty
        dictionary.  If an integer, return a dictionary that associates
        this value with all data types of <valid_data_types>.  If a
        list, return a dictionary that associates the value True with
        each listed data type that is in <valid_data_types>.  If None,
        return a dictionary that associates the value True with all
        data types of <valid_data_types>.

    valid_data_types : list
        List of valid data_types.
    """
    # Filter out keys that aren't valid data types.
    if isinstance(in_value, dict):
        data_indices = {
            data_type: indices
            for data_type, indices in in_value.items()
            if data_type in valid_data_types
        }

    # Associate specific index with all valid data types.
    elif isinstance(in_value, int):
        data_indices = {data_type: in_value for data_type in valid_data_types}

    # Associate value True with specific data type.
    elif isinstance(in_value, str):
        data_indices = {in_value: True} if in_value in valid_data_types else {}

    else:
        # Accept all valid data types.
        if in_value is None:
            data_types = valid_data_types

        # Accept subset of valid data types.
        else:
            data_types = [
                data_type
                for data_type in in_value
                if data_type in valid_data_types
            ]

        data_indices = {data_type: True for data_type in data_types}

    return data_indices


class TicToc:
    """
    Singleton class for storing timing information,
    to allow emulation of MATLAB tic/toc functionality.

    Implementation of the singleton design pattern is based on:
    https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
    """

    # Define the single instance as a class attribute
    instance = None

    # Single instance inner class
    class __TicToc:
        def __init__(self, message=None, time_format=None, log_level=None):
            """
            Constructor of TicToc inner class.

            This constructor is called the first time that the TicToc
            (outer) class is called.

            Parameters are as documemnted for
            skrt.core.TicToc.__init__() method.
            """
            # Initialise parameter values.

            # Set attribute values.
            self.start = None
            self.tics = []
            self.message = False
            self.default_message = "Time elapsed is "
            self.time_format = ".6f"
            self.log_level = Defaults().log_level
            self.set_options(message, time_format, log_level)
            self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level
            )

        def set_options(self, message=None, time_format=None, log_level=None):
            """
            Set instance attributes.

            Parameters are as documented for
            skrt.core.TicToc.__init__() method.

            A parameter value of None leaves the corresponding
            attribute value unaltered.
            """
            if message is not None:
                self.message = message
            if time_format is not None:
                self.time_format = time_format
            if log_level is not None:
                self.log_level = log_level

        def __repr__(self):
            """Print instance attributes."""
            out = []
            for key, value in sorted(self.__dict__.items()):
                out.append(f"{key}: {value}")
            return "\n".join(out)

    def __init__(self, message=None, time_format=".6f", log_level=None):
        """
        Constructor of TicToc singleton class.

        **Parameters:**

        message: str/bool, default=None
            Value to be assigned to skrt.core.TicToc().message,
            which defines the default behaviour for printing
            elapsed time when calling skrt.core.toc().  If a value
            of True is set, the string skrt.core.TicToc().default_message
            (initialised to "Time elapsed is ") is printed.  In the
            first call to this method, a value of None initialises
            skrt.core.TicToc().message to False.  In subsequent calls,
            a value of None is disregarded.

        time_format : str, default=None
            Value to be assigned to skrt.core.TicToc().time_format,
            which defines the Format for printing elapsed time (seconds).
            In the first call to this method, a value of None initialises
            skrt.core.TicToc().time_format to ".6f".  In subsequent calls,
            a value of None is disregarded.

        log_level : str, default=None
            Value to be assigned to skrt.core.TicToc().log_level, which
            defines the Severity level for event logging.  In the first
            call to this method, a value of None initialises
            skrt.core.TicToc().time_format to Defaults().log_level.  In
            subsequent calls, a value of None is disregarded.
        """
        if not TicToc.instance:
            TicToc.instance = TicToc.__TicToc(message, time_format, log_level)
        else:
            TicToc.instance.set_options(message, time_format, log_level)

    def __getattr__(self, name):
        """Get instance attribute."""
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        """Set instance attribute."""
        return setattr(self.instance, name, value)

    def __repr__(self):
        """Print instance attributes."""
        return self.instance.__repr__()


def tic():
    """
    Set timer start time.
    """
    timer = TicToc()
    setattr(timer, "start", timeit.default_timer())
    timer.tics.append(timer.start)
    return timer.start


def toc(message=None, time_format=None):
    """
    Record, and optionally report, time since corresponding call to tic().

    Each call to tic() adds a start time to the list skrt.core.TicToc().tics.
    Each call to toc() removes the last start time from the list, and records
    the time elapsed since that start time.  If the list is empty, the call
    to toc() records the time elapsed since the last call to tic().  This
    allows for nested timings, and for cumulative timings.

    For example:

        tic() # tic_1
        for idx in range(5)
            tic() # tic_idx
            # do something
            toc() # time since tic_idx
        toc() # time since tic_1
        # do something
        toc() # time since tic_1

    **Parameters:**

    message : str/bool, default=None
        If at least one of message and skrt.core.TicToc().message evaluates
        to True, elapsed time will be printed, preceded by message if
        this is a string, by skrt.core.TicToc().message (initialised
        to False) if this is a string, otherwise by
        skrt.core.TicToc().default_message (initialised to "Elapsed time is ").

    time_format : str, default=None
        Format for printing elapsed time (seconds).  If None,
        use skrt.core.TicToc().time_format, which is initialised to ".6f"
    """
    # Obtain stop time.
    stop = timeit.default_timer()

    timer = TicToc()
    # Log error and return None if timer hasn't been started.
    if timer.start is None:
        timer.logger.error("Timer not started - to start timer, call: tic()")
        return None

    # Obtain time elapsed since relevant call to tic().
    if timer.tics:
        time_taken = stop - timer.tics.pop()
    else:
        time_taken = stop - timer.start

    # Print time elapsed.
    if message or timer.message:
        if not message or not isinstance(message, str):
            if isinstance(timer.message, str):
                message = timer.message
            else:
                message = timer.default_message
    if message:
        if not isinstance(message, str):
            message = ""
        if not time_format:
            time_format = timer.time_format
        print(f"{message}{time_taken:{time_format}} seconds")

    return time_taken


def download(url, outdir=".", outfile=None, binary=True, unzip=False):
    """
    Download data from specified URL.

    **Parameters:**

    url : str
        URL from which to download data.

    outdir : str/pathlib.Path, default="."
        Path to (local) directory to which data are to be downloaded.

    outfile : str, default=None
        Name of the file to be downloaded.  If None, the name is taken
        to be the part of the URL following the last (non-trailing) slash.

    binary : bool, default=True
        If True, treat downloaded file as being in binary format.

    unzip : bool, default=True
        If True, treat downloaded file as being a zip-format archive,
        and unzip.
    """
    # Ensure that output directory exists.
    outdir = Path(fullpath(outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Retreive data from URL.
    with urlopen(url) as resource:
        data = resource.read()

    # Write data.
    outfile = outfile or Path(url).name
    outpath = outdir / outfile
    write_opts = "wb" if binary else "w"
    with open(outpath, write_opts) as outdata:
        outdata.write(data)

    # Unzip archive.
    if unzip:
        with ZipFile(outpath) as zipfile:
            zipfile.extractall(outdir)


def get_stat(values=None, value_for_none=None, stat="mean", **kwargs):
    """
    Calculate statistic(s) for values in a list or dictionary.

    If input values are lists or tuples, a list is returned with
    element-by-element statistics.

    **Parameters:**

    values: list/tuple/dict
        Values for which to calculate statistic(s).  If a dictionary,
        the dictionary values are used.

    value_for_none: int/float/None, default=None
        Value to be substituted for any None values, before calculation
        of statistic(s).  If None, None values are omitted, rather than
        being substituted.

    stat: str, default="mean"
        Statistic(s) to be calculated.  This should be the name of the
        function for calculation of the statistic(s) in the Python
        statistics module:

            https://docs.python.org/3/library/statistics.html

        Available options include: "mean", "median", "mode", "stdev",
        "quantiles".

    kwargs : dict, default=None
        Keyword arguments to be passed to the relevant function of
        the Python statistics module:

            https://docs.python.org/3/library/statistics.html

        For example, if quantiles are required for 10 intervals,
        rather than for the default of 4, this can be specified using:

        kwargs={"n" : 10}
    """
    logger = get_logger(identifier="funcName")

    if isinstance(values, dict):
        values = list(values.values())

    if not values:
        logger.warning("No input values: returning None")
        return None

    if not is_list(values[0]):
        if value_for_none is None:
            values = [value for value in values if value is not None]
        else:
            values = [
                (value if value is not None else value_for_none)
                for value in values
            ]
        try:
            return getattr(statistics, stat)(values, **kwargs)
        except statistics.StatisticsError as error:
            logger.warning("%s: returning None", error)
            return None

    if value_for_none is None:
        components = [
            [value[idx] for value in values if value[idx] is not None]
            for idx in range(len(values[0]))
        ]
    else:
        components = [
            [
                (value[idx] if value[idx] is not None else value_for_none)
                for value in values
            ]
            for idx in range(len(values[0]))
        ]
    try:
        return [
            (
                get_stat(
                    component_values, value_for_none, stat, **(kwargs or {})
                )
                if component_values
                else value_for_none
            )
            for component_values in components
        ]
    except statistics.StatisticsError as error:
        logger.warning("%s: returning None", error)
        return None


def get_stat_functions():
    """
    Get names of statistical functions implemented in Python statistics module.

    For details of statistics module, see:

    https://docs.python.org/3/library/statistics.html
    """
    return [
        function
        for function in dir(statistics)
        if isinstance(getattr(statistics, function), FunctionType)
        and not function.startswith("_")
    ]


def get_dict_permutations(in_dict=None):
    """
    Get list of permuations of key-value pairs from dictionary of lists.

    **Parameter:**

    in_dict: dict, default=None
        Dictionary of lists for which a list of permutations of
        key-value pairs is to be obtained.  In case the object passed
        as in_dict isn't a dictionary, a list containing an empty
        dictionary is returned.

    As an example, the list of permuations for the dictionary:

        {"A": [1, 2], "B": [3, 4]}

    is:

        [{"A": 1, "B": 3}, {"A": 1, "B": 4}, {"A": 2, "B": 3}, {"A": 2, "B": 4}]
    """
    if isinstance(in_dict, dict) and in_dict:
        keys, values = zip(*in_dict.items())
        permutations = [
            dict(zip(keys, value)) for value in itertools.product(*values)
        ]
    else:
        permutations = [{}]
    return permutations


def qualified__name(cls):
    """
    Return qualified name of a class.

    **Parameter:**

    cls : class
        Class for which qualified name is to be determined.  If non-class
        is passed as argument, None is returned.
    """
    if isinstance(cls, type):
        return f"{cls.__module__}.{cls.__name__}"
    return None


def get_n_file_below(indir):
    """
    Return number of files below a directory, ignoring hidden files.

    Returns None if indir isn't the path to a directory.

    **Parameter:**
    indir: str, pathlib.Path
        Path to directory below which files are to be counted.
    """
    if isinstance(indir, (str, Path)) and str(indir):
        indir = Path(indir)
        if indir.is_dir():
            return len(
                [path for path in indir.glob("**/[!.]*") if path.is_file()]
            )
    return None


def print_paths(data_dir, max_path=None):
    """
    Print paths to files below a directory, ignoring hidden files.

    File paths are listed in natural order, with one path per line.

    **Parameters:**
    data_dir: str, pathlib.Path
        Path to directory below which file paths are to be printed.

    max_path: int/None, default=None
        Indication of maximum number of paths to print.  If a positive
        integer, the first <max_path> paths are printed.  If a negative
        integer, the last <max_path> paths are printed.  If None,
        all paths are printed.
    """
    # Obtain sorted list of paths.
    local_paths = sorted(
        list(Path(data_dir).glob("**/[!.]*")), key=alphanumeric
    )

    # Reduce number of paths as needed.
    if max_path is None:
        selected_paths = local_paths
    else:
        if max_path >= 0:
            selected_paths = local_paths[:max_path]
        else:
            selected_paths = local_paths[max_path:]

    # Print paths.
    for path in selected_paths:
        out_path = compress_user(path) if Defaults().compress_user else path
        print(out_path)


def prepend_path(variable, path, path_must_exist=True):
    """
    Prepend path to environment variable.

    **Parameters:**

    variable : str
        Environment variable to which path is to be prepended.

    path : str/pathlib.Path
        Path to be prepended.

    path_must_exist : bool, default=True
        If True, only append path if it exists.
    """
    path = str(path)
    path_ok = True
    if path_must_exist:
        if not os.path.exists(path):
            path_ok = False

    if path_ok:
        if variable in os.environ:
            if os.environ[variable]:
                items = os.environ[variable].split(os.pathsep)
                if path in items:
                    items.remove(path)
                if items:
                    os.environ[variable] = os.pathsep.join([path] + items)
                else:
                    os.environ[variable] = path
        else:
            os.environ[variable] = path

    return path_ok


def set_matlab_runtime(matlab_runtime=None, log_level=None):
    """
    Set environment to allow use of MATLAB runtime installation in subprocess.

    **Parameters:**

    matlab_runtime: str/pathlib.Path, default=None
        Path to root directory of MATLAB runtime installation.  It the
        value is None, matlab_runtime is set to the value of
        Defaults().log_level.

    log_level: string/integer/None, default=None
        Severity level for event logging.  If the value is None,
        log_level is set to the value of Defaults().log_level.
    """
    logger = get_logger(identifier="funcName", log_level=log_level)

    # Check that matlab_runtime is a non-null string or pathlib.Path
    matlab_runtime = matlab_runtime or Defaults().matlab_runtime
    if isinstance(matlab_runtime, (Path, str)) and matlab_runtime:
        matlab_runtime = Path(fullpath(matlab_runtime))
    else:
        logger.warning("Root directory of MATLAB runtime not defined.")
        return False

    # Check that mathlib_runtime specifies a directory that exists.
    if not matlab_runtime.is_dir():
        logger.warning(
            "Root directory of MATLAB runtime not found: %s", matlab_runtime
        )
        return False

    # Define platform-specific subdirectories and path variable.
    subdirs = [
        "runtime",
        "bin",
        Path("sys") / "os",
    ]
    if "Linux" == platform.system():
        subdirs.append(Path("sys") / "opengl" / "lib")
        env_var = "LD_LIBRARY_PATH"
        arch = "glnxa64"
    elif "Darwin" == platform.system():
        env_var = "DYLD_LIBRARY_PATH"
        arch = "maci64"
    else:
        env_var = "PATH"
        subdirs = ["runtime"]
        arch = "win64"

    # Set path variable, and check that runtime paths exists.
    all_ok = True
    for subdir in subdirs:
        env_val = matlab_runtime / subdir / arch
        if env_var not in os.environ or str(env_val) not in os.environ[
            env_var
        ].split(os.pathsep):
            runtime_ok = prepend_path(env_var, env_val)
            if not runtime_ok:
                logger.warning(
                    "MATLAB runtime directory not found: '%s'", env_val
                )
            all_ok *= runtime_ok

    return all_ok
