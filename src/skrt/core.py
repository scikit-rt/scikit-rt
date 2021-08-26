'''Core data classes.'''

import numpy as np
import os
import functools


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


Defaults({'print_depth': 0})


class Data:
    '''
    Base class for objects serving as data containers.
    An object has user-defined data attributes, which may include
    other Data objects and lists of Data objects.

    The class provides for printing attribute values recursively, to
    a chosen depth, and for obtaining nested dictionaries of
    attributes and values.
    '''

    def __init__(self, opts={}, **kwargs):
        '''
        Constructor of Data class, allowing initialisation of an
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
        # an instance of Data or a subclass
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
                if issubclass(item.__class__, Data):
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


class PathData(Data):
    '''Data with and associated directory; has the ability to
    extract a list of dated objects from within this directory.'''

    def __init__(self, path=''):
        self.path = fullpath(path)
        self.subdir = ''

    def get_dated_objects(self, dtype, subdir='', **kwargs):
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
    '''PathData with an associated date and time, which can be used for
    sorting multiple Dateds.'''

    def __init__(self, path=''):

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


class MachineData(Dated):
    '''Dated object with an associated machine name.'''

    def __init__(self, path=''):
        Dated.__init__(self, path)
        self.machine = os.path.basename(os.path.dirname(path))


class Archive(Dated):
    '''Dated object associated with multiple files.'''

    def __init__(self, path='', allow_dirs=False):

        Dated.__init__(self, path)
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


class File(Dated):
    '''File with an associated date. Files can be sorted based on their
    filenames.'''

    def __init__(self, path=''):
        Dated.__init__(self, path)

    def __cmp__(self, other):

        result = Dated.__cmp__(self, other)
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
    '''Extract time and date separately from timestamp.'''

    time_date = (None, None)
    if is_timestamp(timestamp):
        items = os.path.splitext(timestamp)[0].split('_')
        items = [item.strip() for item in items]
        if items[0].isalpha():
            time_date = tuple(items[1:3])
        else:
            time_date = tuple(items[0:2])
    else:
        i1 = timestamp.find('_')
        i2 = timestamp.rfind('.')
        if (-1 != i1) and (-1 != i2):
            bitstamp = timestamp[i1 + 1: i2]
            if is_timestamp(bitstamp):
                time_date = tuple(bitstamp.split('_'))

    return time_date


def is_timestamp(string=''):
    '''Check whether a string is a valid timestamp.'''

    valid = True
    items = os.path.splitext(string)[0].split('_')
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
    return valid


def is_list(var):
    '''Check whether a variable is a list, tuple, or array.'''

    is_a_list = False
    for t in [list, tuple, np.ndarray]:
        if isinstance(var, t):
            is_a_list = True
    return is_a_list


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
