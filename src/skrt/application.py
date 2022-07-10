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
'''

from skrt.core import Defaults, get_logger
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
            self.reason = 'No algorithms to run'
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
            if not self.status.ok():
                if not self.status.reason:
                    self.status.reason = \
                        f'Problem initialising algorithm "{alg.name}"'
                break

        return self.status

    def run(self, paths=None, unsorted_dicom=False, id_mappings=None):
        '''
        Initialise analysis, process patient data, finalise analysis.

        **Parameters:**

        paths: list, default=None
            List of paths to folders containing patient data.  If null,
            set to empty list.

        unsorted_dicom: bool, default=False
            If True, examine all files below the top directory, and
            sort into images, structure sets, doses and plans.  If
            False, files are assumed to be organised according to the
            VoxTox model.

        id_mappings: dict, default=None
            By default, patient identifiers are taken to be the
            names of the top-level directories containing patient data.
            The dictionary id_mappings can optionally be provided to
            map between the default identifiers (keys) and alternative
            identifiers (values).
        '''

        if self.status.ok():
            paths = paths or []
            if not paths:
                self.logger.warning('List of paths to patient data is empty')
            for data_path in paths:
                patient = Patient(path=data_path, unsorted_dicom=unsorted_dicom,
                        id_mappings=id_mappings)
                self.status = self.execute(patient=patient)
                if not self.status.ok():
                    break

        if self.status.ok():
            self.status = self.finalise()

        return self.status


class Status():
    '''
    Represent an exit status.

    **Methods:**

    * **__init__()**: Create instance of Status class.
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

    def ok(self):
        '''
        Return boolean indicating whether status is okay (exit code non-zero).
        '''
        if self.code:
            is_ok = False
        else:
            is_ok = True

        return is_ok
