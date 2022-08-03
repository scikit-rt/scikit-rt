"""Scikit-rt core data classes and functions."""

from collections.abc import Iterable
from pathlib import Path

import copy
import os
import re
import time
from logging import getLogger, Formatter, StreamHandler
from typing import Any, List, Optional, Tuple

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

        if opts:
            for key, value in opts.items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self, depth: Optional[int] = None) -> str:
        """
        Create string recursively listing attributes and values.

        **Parameters:**

        depth: integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, depth is set to the value
            of the object's print_depth property, if defined,
            or otherwise to the value of Defaults().print_depth.
        """

        if depth is None:
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
                n = len(items)
                if n:
                    if depth > 0:
                        value_string = "["
                        for i, item in enumerate(items):
                            try:
                                item_string = item.__repr__(depth=(depth - 1))
                            except TypeError:
                                item_string = item.__repr__()
                            comma = "," if (i + 1 < n) else ""
                            value_string = f"{value_string} {item_string}{comma}"
                        value_string = f"{value_string}]"
                    else:
                        value_string = f"[{n} * {item[0].__class__}]"
                else:
                    value_string = "[]"

            # Handle printing of dicts
            elif isinstance(item, dict):
                items = item
                n = len(items)
                if n:
                    if depth > 0:
                        value_string = "{"
                        for i, (key, value) in enumerate(items.items()):
                            item_string = "{key}: "
                            try:
                                item_string += item.__repr__(depth=(depth - 1))
                            except TypeError:
                                item_string += item.__repr__()
                            comma = "," if (i + 1 < n) else ""
                            value_string = f"{value_string} {item_string}{comma}"
                        value_string = f"{{{value_string}}}"
                    else:
                        value_string = f"{{{n} * keys of type {list(item.keys())[0].__class__}}}"
                else:
                    value_string = "{}"

            # Handle printing of pydicom datasets
            elif isinstance(item, pydicom.dataset.FileDataset):
                value_string = str(item.__class__)

            # Handle printing of nested Data objects
            else:
                if issubclass(item.__class__, Data):
                    if depth > 0:
                        value_string = item.__repr__(depth=(depth - 1))
                    else:
                        value_string = f"{item.__class__}"
                else:
                    value_string = item.__repr__()
            out.append(f"  {key}: {value_string} ")

        out.append("}")
        return "\n".join(out)

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
                    print("Warning: data_types_to_copy must inherit from "
                          "skrt.Data! Type", dtype, "will be ignored.")

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

        if not hasattr(self, "print_depth"):
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
            If the value is None, depth is set in the
            __repr__() method.
        """

        print(self.__repr__(depth))

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
    '''
    Class representing files in DICOM format, with conversion to skrt objects.

    **Methods:**
    - **_init__()** : Create instance of DicomFile class.
    - **get_object()** : Instantiate object of specified Scikit-rt class.
    - **get_matched_attributes()** : Identify objects with matched attributes.
    - **set_dates_and_times()** : Store timing for specified elements.
    - **set_referenced_sop_instance_uids** : Store UIDs of referenced objects.
    - **set_slice_thickness** : Store slice thickness for 3D imaging data.
    '''

    def __init__(self, path):
        '''
        Create instance of DicomFile class.

        **Parameter:**

        path : str/pathlib.Path
            Relative or absolute path to DICOM file.
        '''
        # Store absolute file path as string.
        self.path = fullpath(path)

        # Attempt to read DICOM dataset.
        try:
            self.ds = pydicom.dcmread(self.path, force=True)
        except IsADirectoryError:
            self.ds = None

        # Define prefixes of dataset attributes to be used
        # to set study and item timestamps.
        elements = {
                "study": ["Study", "Content", "InstanceCreation"],
                "item": ["Instance", "Content", "Series", "InstanceCreation"]
                }

        # Set object attributes.
        self.set_dates_and_times(elements)
        self.set_slice_thickness()
        self.frame_of_reference_uid = getattr(
                self.ds, "FrameOfReferenceUID", None)
        self.modality = getattr(self.ds, "Modality", "unknown")
        if self.modality:
            self.modality = self.modality.lower()
        self.series_number = getattr(self.ds, "SeriesNumber", None)
        self.series_instance_uid = getattr(self.ds, "SeriesInstanceUID", None)
        self.sop_instance_uid = getattr(self.ds, "SOPInstanceUID", None)
        self.study_instance_uid = getattr(self.ds, "StudyInstanceUID", None)
        self.set_referenced_sop_instance_uids()

    def get_object(self, cls, **kwargs):
        '''
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
        '''
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
                self.referenced_image_sop_instance_uid)
        obj.referenced_plan_sop_instance_uid = (
                self.referenced_plan_sop_instance_uid)
        obj.referenced_structure_set_sop_instance_uid = (
                self.referenced_structure_set_sop_instance_uid)
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
            matches = [match for match in matches if
                    (getattr(match, attribute, None)
                    == getattr(self, attribute, None))]

        return matches 

    def set_dates_and_times(self, elements):
        '''
        Store timing information for specified elements.

        **Parameter:**

        elements: dict
            Dictionary where keys are element names - for example,
            "study", "item" - and values are DICOM dataset attributes
            to be checked, in the order listed, for date and time information.
        '''
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
            if self.ds is None:
                timestamp = None
            else:
                date = getattr(self, f"{element}_date")
                time = getattr(self, f"{element}_time")
                setattr(self, f"{element}_timestamp", f"{date}_{time}")

    def set_referenced_sop_instance_uids(self):
        '''
        Store SOP instance UIDs for referenced image, structure set, and plan.
        '''

        # Store SOP instance UID for referenced image.
        try:
            uid = self.ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
        except (IndexError, AttributeError):
            uid = None

        if uid is None:
            try:
                uid = (self.ds.ROIContourSequence[0].ContourSequence[0]
                        .ContourImageSequence[0].ReferencedSOPInstanceUID)
            except (IndexError, AttributeError):
                pass

        self.referenced_image_sop_instance_uid = uid

        # Store SOP instance UID for referenced structure set.
        try:
            uid = (self.ds.ReferencedStructureSetSequence[0]
                    .ReferencedSOPInstanceUID)
        except (IndexError, AttributeError):
            None

        self.referenced_structure_set_sop_instance_uid = uid

        # Store SOP instance UID for referenced plan.
        try:
            uid = (self.ds.ReferencedRTPlanSequence[0]
                    .ReferencedSOPInstanceUID)
        except (IndexError, AttributeError):
            uid = None

        self.referenced_plan_sop_instance_uid = uid

    def set_slice_thickness(self):
        '''
        Store slice thickness for 3D imaging data.
        '''
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
        self.path = fullpath(str(path))
        self.subdir = ""

    def create_objects(
        self, 
        dtype: type, 
        subdir: str = "", 
        timestamp_only=True,
        **kwargs
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
        '''
        Return size in bytes of associated file.
        '''
        return Path(self.path).stat().st_size if Path(self.path).exists() else 0

    def get_n_file(self):
        '''
        Return number of data files associated with this object.
        '''
        # Only 1 data file associated with a non-Archive object.
        return 1


class Dated(PathData):
    """PathData with an associated date and time, which can be used for
    sorting multiple Dateds."""

    def __init__(self, path: str = "", auto_timestamp=False):
        """
        Initialise dated object from a path and assign its timestamp. If
        no valid timestamp is found within the path string, it will be set
        automatically from the current date and time if auto_timestamp is True.
        """

        PathData.__init__(self, path)

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
            timestamp = pd.Timestamp(''.join([self.date, self.time]))
        except ValueError:
            timestamp = None
        return timestamp

    def in_date_interval(self,
                         min_date: Optional[str] = None,
                         max_date: Optional[str] = None) -> bool:
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
            except NoneType:
                delta_time = 0
            if other.date == self.date and delta_time <= delta_seconds:
                matches.append(other)

        return matches

    def __gt__(self, other):
        '''
        Define whether <self> is greater than <other>.

        The comparison is based first on object date, then on time,
        then on path.
        '''
        for attribute in ["date", "time", "path"]:
            if getattr(self, attribute) != getattr(other, attribute):
                return getattr(self, attribute) > getattr(other, attribute)
            return False

    def __ge__(self, other):
        '''
        Define whether <self> is greater than, or equal to, <other>.

        The comparison is based first on object date, then on time,
        then on path.
        '''
        return (self > other) or (self == other)


class MachineData(Dated):
    """Dated object with an associated machine name."""

    def __init__(self, path: str = ""):
        Dated.__init__(self, path)
        self.machine = os.path.basename(os.path.dirname(path))


class Archive(Dated):
    """Dated object associated with multiple files."""

    def __init__(self, path: str = "", auto_timestamp=False,
            allow_dirs: bool = False):

        Dated.__init__(self, path, auto_timestamp)

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
        '''
        Return size in bytes of associated file.
        '''
        if Path(self.path).is_file():
            size = Path(self.path).stat().st_size
        else:
            size = 0
            for file in self.files:
                if Path(file.path).exists():
                    size += Path(file.path).stat().st_size

        return size

    def get_n_file(self):
        '''
        Return number of data files associated with this object.
        '''
        if Path(self.path).is_file():
            n_file = 1
        else:
            n_file = len(self.files)
        return n_file

class File(Dated):
    """File with an associated date. Files can be sorted based on their
    filenames."""

    def __init__(self, path: str = "", auto_timestamp=False):
        Dated.__init__(self, path, auto_timestamp)

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


def alphanumeric(in_str: str = "") -> List[str]:
    """Function that can be passed as value for list sort() method
    to have alphanumeric (natural) sorting"""

    elements = []
    for substr in re.split("(-*[0-9]+)", in_str):
        try:
            element = int(substr)
        except BaseException:
            element = substr
        elements.append(element)
    return elements


def fullpath(path=""):
    """Evaluate full path, expanding '~', environment variables, and
    symbolic links."""

    expanded = ""
    if path:
        path = str(path)
        tmp = os.path.expandvars(path.strip())
        tmp = os.path.abspath(os.path.expanduser(tmp))
        expanded = os.path.realpath(tmp)
    return expanded

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
    handler = StreamHandler()
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
        i1 = timestamp.find("_")
        i2 = timestamp.rfind(".")
        if (-1 != i1) and (-1 != i2):
            bitstamp = timestamp[i1 + 1: i2]
            if is_timestamp(bitstamp):
                items = bitstamp.split("_")
                if len(items) > 2:
                    if (items[0].isalpha() and items[1].isdigit()
                            and items[2].isdigit()):
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


def to_list(val: Any, n : int = 3, keep_none_single : bool = True) -> Optional[List]:
    """Ensure that a value is a list of n items."""

    if val is None and keep_none_single:
        return None
    if is_list(val):
        if not len(val) == n:
            print(f"Warning: {val} should be a list containing {n} items!")
        return list(val)
    return [val] * n


def generate_timestamp() -> str:
    '''Make timestamp from the current time.'''

    return time.strftime('%Y%m%d_%H%M%S')

def get_data_by_filename(data_objects=None, remove_timestamp=True,
        remove_suffix=True):

    '''
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
    '''

    data_by_filename = {}
    if data_objects:

        idx = 0
        for data_object in data_objects:
            path = getattr(data_object, 'path', '')
            if path:
                # Determine filename, removing timestamp and suffix if needed.
                filename = Path(path).name
                if remove_timestamp:
                    time_and_date = get_time_and_date(filename)
                    if None not in time_and_date:
                        timestamp = '_'.join(time_and_date)
                        filename = ''.join(filename.split(f'{timestamp}_'))
                if remove_suffix:
                    filename = filename.split('.')[0]
            else:
                # Create dummy filename for object with no associated path.
                idx += 1
                filename = f'unknown_{idx:03}'

            data_by_filename[filename] = data_object

    return data_by_filename

def get_n_file(objs=None):
    '''
    Return number of data files associated with listed objects.

    **Parameter:**

    objs : list, default=None
        List of objects for which numbers of files are to be summed.
    '''
    if not isinstance(objs, Iterable):
        objs = [objs]

    n_file = 0
    for obj in objs:
        n_file += obj.get_n_file()
    return n_file

def get_file_size(objs=None):
    '''
    Return size in bytes of data files associated with listed objects.

    **Parameter:**

    objs : list, default=None
        List of objects for which file sizes are to be summed.
    '''
    if not isinstance(objs, Iterable):
        objs = [objs]

    size = 0
    for obj in objs:
        size += obj.get_file_size()
    return size

def get_time_separated_objects(objs, min_delta=4, unit='hour',
        most_recent=True):
    '''
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
    '''

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
                    timestamp = pd.Timestamp(''.join([obj.date, obj.time]))
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
    '''
    Return a timestamp's hour in day, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which hour in day is to be returned.
    '''
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600

def get_hour_in_week(timestamp):
    '''
    Return a timestamp's hour in week, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which hour in week is to be returned.
    '''
    if isinstance(timestamp, pd.Timestamp):
        return (24 * (timestamp.isoweekday() - 1) + get_hour_in_day(timestamp))

def get_day_in_week(timestamp):
    '''
    Return a timestamp's day in week, including fractional part.

    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp for which day in week is to be returned.
    '''
    if isinstance(timestamp, pd.Timestamp):
        return get_hour_in_week(timestamp) / 24

def get_interval_in_days(timestamp1, timestamp2):
    '''
    Return interval in days between two timestamps.

    **Parameters:**

    timestamp1 : pandas.Timestamp
        Timestamp corresponding to start of interval.

    timestamp2 : pandas.Timestamp
        Timestamp corresponding to end of interval.
    '''
    if (isinstance(timestamp1, pd.Timestamp) and
            isinstance(timestamp2, pd.Timestamp)):
        interval = (timestamp2 - timestamp1).total_seconds()
        interval /= pd.Timedelta('1d').total_seconds()
    else:
        interval = None
    return interval

def get_interval_in_whole_days(timestamp1, timestamp2):
    '''
    Return interval in whole days between two timestamps.
    
    **Parameters:**

    timestamp1 : pandas.Timestamp
        Timestamp corresponding to start of interval.

    timestamp2 : pandas.Timestamp
        Timestamp corresponding to end of interval.
    '''
    if (isinstance(timestamp1, pd.Timestamp) and
            isinstance(timestamp2, pd.Timestamp)):
        interval = ((timestamp2.floor('1d') - timestamp1.floor('1d'))
                .total_seconds())
        interval /= pd.Timedelta('1d').total_seconds()
        interval = round(interval)
    else:
        interval = None
    return interval

def year_fraction(timestamp):
    '''
    Convert from timestamp to year, including fractional part.
    
    **Parameter:**

    timestamp : pandas.Timestamp
        Timestamp to be converted.
    '''
    if isinstance(timestamp, pd.Timestamp):
        # Determine year.
        year = timestamp.year

        # Determine seconds in year (different for leap year and non-leap year).
        year_start = pd.Timestamp(f'{year}0101')
        next_year_start = pd.Timestamp(f'{year + 1}0101')
        seconds_in_year = (next_year_start - year_start).total_seconds()

        # Determine seconds elapsed so far in year.
        seconds_to_date = (timestamp - year_start).total_seconds()

        # Add year and fractional part.
        year_fraction = year + seconds_to_date / seconds_in_year
    else:
        year_fraction = None
    return year_fraction

def get_uid_without_slice(uid):
    """Obtain copy of <uid>, truncated to before final dot."""
    return ".".join(uid.split(".")[:-1])

'''
def get_sequence_value(ds=None, sequence=None, tag=None):
    value = None
    sequence_data = getattr(ds, sequence, None)
    if None not in [ds, sequence, tag]:
        sequence_data = getattr(ds, sequence, None)
        if sequence_data:
            value = getattr(sequence_data[-1], tag, None)
    return value
'''

def get_referenced_image(referrer=None, image_types=None):
    '''
    Retrieve from <image_types> image object referred to by <referrer>.

    **Parameters:**

    referrer : object, default=None
        Object that references an image object via its SOP instance UID.

    image_types : dict, default=None
        Dictionary where keys are imaging modalities and values are lists
        of image objects for this modality.
    '''
    image_types = image_types or {}
    image = None

    # Search for referenced image based on matching
    # referenced image SOP instance UID.
    for modality, images in image_types.items():
        image = get_referenced_object(referrer, images,
                "referenced_image_sop_instance_uid", True)
        if image:
            break

    # Search for referenced image based on matching
    # frame-of-reference UID.
    if image is None:
        for modality, images in image_types.items():
            matched_attributes = DicomFile.get_matched_attributes(
                    referrer, images, "frame_of_reference_uid")
            if matched_attributes:
                image = matched_attributes[0]

    return image

def get_referenced_object(referrer, others, tag, omit_slice=False):
    '''
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
    '''
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
