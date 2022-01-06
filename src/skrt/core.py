"""Core data classes and functions."""

import copy
import functools
import os
import re
import time
from logging import getLogger, Formatter, StreamHandler
from typing import Any, List, Optional, Tuple

import numpy as np
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


class PathData(Data):
    """Data with an associated path or directory; has the ability to
    extract a list of dated objects from within this directory."""

    def __init__(self, path=""):
        self.path = fullpath(path)
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


@functools.total_ordering
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
        if self.date == other.date:
            return self.time < other.time
        return self.date < other.date


class MachineData(Dated):
    """Dated object with an associated machine name."""

    def __init__(self, path: str = ""):
        Dated.__init__(self, path)
        self.machine = os.path.basename(os.path.dirname(path))


class Archive(Dated):
    """Dated object associated with multiple files."""

    def __init__(self, path: str = "", allow_dirs: bool = False):

        Dated.__init__(self, path)

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


class File(Dated):
    """File with an associated date. Files can be sorted based on their
    filenames."""

    def __init__(self, path: str = ""):
        Dated.__init__(self, path)

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


def fullpath(path: str = "") -> str:
    """Evaluate full path, expanding '~', environment variables, and
    symbolic links."""

    expanded = ""
    if path:
        tmp = os.path.expandvars(path.strip())
        tmp = os.path.abspath(os.path.expanduser(tmp))
        expanded = os.path.realpath(tmp)
    return expanded

def get_logger(name="", log_level=None):
    """
    Retrieve named event logger.

    **Parameters:**

    name: string, default=""
        Name of logger (see documentation of logging module)

    log_level: string/integer/None, default=None
        Severity level for event logging.  If the value is None,
        log_level is set to the value of Defaults().log_level.
    """
    formatter = Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger = getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
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
                time_date = tuple(bitstamp.split("_"))

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
