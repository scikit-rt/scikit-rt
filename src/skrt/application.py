'''
Framework for running analysis over patient data.

This module provides for construction of an analysis application
as a sequence of algorithms, each of which inherits from the
Algorithm class.  This defines three methods relating to execution:

* **initialise()** - run before any patient data are read;
* **execute()** - run once for each patient;
* **finalise()** - run after all patient data have been read.

Each method returns an instance of the Status class,
providing information on whether execution problems were encountered.
Running of algorithm methods is handled by the Application class.

This module also provides the following function that may be useful within an
application:
* **get_paths()** - get list of paths to patient datasets.
'''
from itertools import chain
from pathlib import Path

from skrt.core import Defaults, fullpath, get_logger, is_list
from skrt.patient import Patient


class Algorithm():
    '''
    Base class for analysis algorithms.

    Each analysis algorithm should inherit from Algorithm,
    overwriting as needed the methods initialise, execute, finalise,
    to implement analysis-specific functionality.

    **Methods:**

    * **__init__()**: Create instance of Algorithm class.
    * **initialise()**: Perform tasks required before considering any patients.
    * **execute()**: Perform tasks required for each patient.
    * **finalise()**: Perform tasks required after considering all patients.
    * **set_attributes()**: Set values for algorithm attributes.
    '''

    def __init__(self, opts=None, name=None, log_level=None):
        '''
        Create instance of Algorithm class.

        **Parameters:**

        opts: dict, default=None
            Dictionary for setting algorithm attributes.  If null,
            set to empty dictionary.

        name: str, default=None
            Name for identifying algorithm instance.  If null, set to
            class name.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        self.opts = opts or {}
        class_name = type(self).__name__
        self.name = class_name if name is None else name
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(name=class_name, log_level=self.log_level)
        self.status = Status(name=self.name)

        # Initialise algorithm attributes.
        self.set_attributes(self.opts)

    def execute(self, patient=None):
        '''
        Perform tasks required for each patient.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        '''

        # Print patient identifier and path
        print(f'{patient.id}: {patient.path}')

        return self.status

    def finalise(self):
        '''
        Perform tasks required after considering all patients.
        '''
        return self.status

    def initialise(self):
        '''
        Perform tasks required before considering any patients.
        '''
        return self.status

    def set_attributes(self, opts):
        '''
        Set values for algorithm attributes.

        **Parameter:**

        opts: dict, default={}
            Dictionary for setting algorithm attributes.
        '''
        for key in opts:
            setattr(self, key, opts[key])

class Application():
    '''
    Represent application as sequence of algorithm, and manage execution.

    **Methods:**

    * **__init__()**: Create instance of Application class.
    * **initialise()**: Call each algorithm's initialise method.
    * **execute()**: For each patient, call each algorithm's execute method.
    * **finalise()**: Call each algorithm's finalise method.
    * **run()**: Initialise analysis, process patient data, finalise analysis.
    '''

    def __init__(self, algs=None, log_level=None):
        '''
        Create instance of Application class.

        **Parameters:**

        algs: list, default=None
            List of algorithms to be managed.  Algorithms are processed
            in the order in which they are specified in the list.  If null,
            set to empty list.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        self.algs = algs or []
        class_name = type(self).__name__
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(name=class_name, log_level=self.log_level)
        self.status = Status(name=class_name)

        if not self.algs:
            self.status.code = 1
            self.status.reason = 'No algorithms to run'
        else:
            for alg in self.algs:
                if not alg.status.ok():
                    self.status.code = alg.status.code
                    self.status.reason = f'{alg.name}: {alg.status.reason}'

    def execute(self, patient=None):
        '''
        For each patient, call each algorithm's execute method.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None

            Object providing access to patient information.
        '''
        for alg in self.algs:
            self.status = alg.execute(patient=patient)
            if not self.status.ok():
                if not self.status.reason:
                    self.status.reason = (
                        f'Problem executing algorithm "{alg.name}" for data '
                        f'path "{patient.path}"')
                break

        return self.status

    def finalise(self):
        '''
        Call each algorithm's finalise method.
        '''
        for alg in self.algs:
            self.status = alg.finalise()
            if not self.status.ok():
                if not self.status.reason:
                    self.status.reason = \
                        f'Problem finalising algorithm "{alg.name}"'
                break

        return self.status

    def initialise(self):
        '''
        Call each algorithm's initialise method.
        '''

        for alg in self.algs:
            self.status = alg.initialise()
            if not self.status.ok():
                if not self.status.reason:
                    self.status.reason = \
                        f'Problem initialising algorithm "{alg.name}"'
                break

        return self.status

    def run(self, paths=None, PatientClass=None, **kwargs):
        '''
        Initialise analysis, process patient data, finalise analysis.

        **Parameters:**

        paths: list, default=None
            List of paths to folders containing patient data.  If null,
            set to empty list.

        PatientClass: class, default=None
            Class to be used to create objects representing patient
            datasets.  The class constructor must have a parameter
            <path>, which will be passed, one by one, the elements of
            <paths>.  The class constructor may have any number of
            additional parameters, values for which can be passed
            via <**kwargs>.  If a null value is given, the class
            used is skrt.patient.Patient.

        **kwargs:
            Keyword arguments that will be passed to the <PatientClass>
            constructor.
        '''
        if self.status.ok():
            self.status = self.initialise()

        if self.status.ok():
            PatientClass = PatientClass or Patient
            paths = paths or []
            if not paths:
                self.status.code = 1
                self.status.reason = 'List of paths to patient data is empty'
            else:
                for data_path in paths:
                    patient = PatientClass(path=data_path, **kwargs)
                    self.status = self.execute(patient=patient)
                    if not self.status.ok():
                        self.logger.error(self.status)
                        break

        if self.status.ok():
            self.status = self.finalise()

        return self.status


class Status():
    '''
    Represent an exit status.

    **Methods:**

    * **__init__()**: Create instance of Status class.
    * **copy_attributes()**: Copy attributes from another Status object.
    * **ok()**: Return boolean, indicating whether status is okay.
    '''

    def __init__(self, code=0, name=None, reason=None):
        '''
        Create instance of Status class.

        Instances of the Status class are intended to be returned by
        algorithm methods, with attributes set to indicate exit status.

        **Parameters:**

        code: int, default=0
            Value for exit status.

        name: str, default=None
            Name for exit status.  If null, set to empty string.

        reason: str, default=None
            Reason for exit status.  If null, set to empty string.
        '''
        self.code = code
        self.name = name or ""
        self.reason = reason or ""

    def __repr__(self):
        """Print status information."""
        return f"Status {self.code} ({self.name}): {self.reason}"

    def ok(self):
        '''
        Return boolean indicating whether status is okay (exit code non-zero).
        '''
        if self.code:
            is_ok = False
        else:
            is_ok = True

        return is_ok

    def copy_attributes(self, status):
        '''
        Copy attributes from another Status object.

        **Parameter:**

        status : skrt.application.Status
            Status object from which to copy attributes.
        '''
        self.code = status.code
        self.name = status.name
        self.reason = status.reason


def get_paths(data_locations=None, max_path=None,
        to_keep=None, to_exclude=None):
    """
    Get list of paths to patient datasets.

    **Parameter:**

    data_locations : pathlib.Path/str/list/dict, default=None
        Specification of how to identify paths to patient datasets:

        - string of pathlib.Path giving path to a single patient dataset;
        - list of strings or pathlib.Path giving full paths to patient datasets;
        - dictionary where keys are paths to directories containing
          patient datasets, and values are strings, or lists of strings,
          indicating patterns to be matched.

    max_path : int, default=None
        Maximum number of paths to return.  If None, return paths to all
        directories in data directory (currently hardcoded in this function).

    to_keep : str/list, default=None
        Patient identifier, or list of patient identifiers.  If non-null,
        only return dataset paths for the specified identifier(s).

    to_exclude : str/list, default=None
        Patient identifier, or list of patient identifiers.  If non-null,
        only return dataset paths excluding the specified identifier(s).
        If an identifier is specified as both <to_keep> and <to_exclude>,
        the latter takes precedence.
    """
    # Return empty list if data_locations is None.
    paths = []
    if data_locations is None:
        return paths

    # If data_locations passed as pathlib.Path or string, convert to list.
    if isinstance(data_locations, (Path, str)):
        data_locations = [data_locations]

    # Obtain dataset paths from list of data locations.
    if isinstance(data_locations, (list, set, tuple)):
        paths = [str(fullpath(data_location))
                for data_location in  data_locations]

    # Obtain dataset paths from dictionary associating directories and patterns.
    elif isinstance(data_locations, dict):
        for data_dir, patterns in data_locations.items():
            paths.extend([str(fullpath(path)) for path in chain.from_iterable(
                [Path(data_dir).glob(pattern) for pattern in patterns])
                if path.is_dir()])

    # Filter list of dataset paths.
    if to_keep:
        if not is_list(to_keep):
            to_keep = [to_keep]
        paths = [path for path in paths
                if any([Path(path).match(id) for id in to_keep])]

    if to_exclude:
        if not is_list(to_exclude):
            to_exclude = [to_exclude]
        paths = [path for path in paths
                if not any([Path(path).match(id) for id in to_exclude])]

    # Sort paths, ensuring that each path is included only once, 
    paths = sorted(list(set(paths)))

    # Determine maximum number of paths to return.
    max_path = min(max_path if max_path is not None else len(paths), len(paths))

    return paths[0: max_path]
