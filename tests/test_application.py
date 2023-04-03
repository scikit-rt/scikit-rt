'''Test application classes.'''
from pathlib import Path
from random import randint, sample, seed
from shutil import rmtree

from skrt.patient import Patient
from skrt.application import Algorithm, Application, Status, get_paths
from skrt.core import Defaults, fullpath

seed(1)

class PyTestAlgorithm(Algorithm):
    '''Minimal Algorithm subclass for testing.'''

    def __init__(self, opts=None, name=None, log_level=None):
        # Override default properties, based on contents of opts dictionary.
        self.test_opt1 = None
        self.test_values = {}
        super().__init__(opts, name, log_level)

    def execute(self, patient=None):
        assert patient.test_value == self.test_values.get(patient.id, None)
        return self.status

class PyTestPatient:
    '''Minimal Patient-like class for testing.'''

    def __init__(self, path=None, test_values=None):
        self.path = path
        self.id = Path(path).name
        if isinstance(test_values, dict):
            self.test_value = test_values.get(self.id, None)
        else:
            self.test_value = None

def test_algorithm_base_class():
    """Test methods of Algorithm base class."""
    opts = {"test_opt": "test_value"}
    name = "test_algorithm"
    alg = Algorithm(opts, name)
    for key, value in opts.items():
        assert getattr(alg, key) == value
    assert alg.name == name
    assert alg.initialise().ok()
    assert alg.execute(Patient("test")).ok()
    assert alg.finalise().ok()

def test_algorithm_creation():
    """Create test algorithms and check properties."""

    alg1 = PyTestAlgorithm()
    assert alg1.test_opt1 is None
    assert not hasattr(alg1, "test_opt2")
    assert alg1.name == "PyTestAlgorithm"
    assert alg1.log_level == Defaults().log_level

    opts = {"test_opt1": 1, "test_opt2": 2}
    name = "NamedPyTestAlgorithm"
    log_level = "CRITICAL"
    alg2 = PyTestAlgorithm(opts, name, log_level)
    for opt, opt_value in opts.items():
        assert getattr(alg2, opt) == opt_value
    assert alg2.name == name
    assert alg2.log_level == log_level

def test_application_creation():
    """Create test application and check properties."""

    alg1 = PyTestAlgorithm(name="name1")
    alg2 = PyTestAlgorithm(name="name2")
    app = Application([alg1, alg2])
    assert app.log_level == Defaults().log_level
    assert app.logger.name == "Application"
    assert len(app.algs) == 2
    assert app.algs[0].name == "name1"
    assert app.algs[1].name == "name2"

def test_application_creation_no_algorithms():
    """Create test application with no algorithms and check properties."""

    app = Application()
    assert not app.algs
    assert 1 == app.status.code
    assert "No algorithms to run" == app.status.reason

def test_application_creation_algorithm_status_not_ok():
    """Create test application with algorithm having status not okay."""

    alg1 = PyTestAlgorithm()
    alg1.status = Status(code=1, reason="No reason")
    app = Application([alg1])
    assert 1 == len(app.algs)
    assert not app.status.ok()
    assert alg1.status.code == app.status.code
    assert f"{alg1.name}: {alg1.status.reason}" == app.status.reason

def test_application_run_no_paths():
    """Test running of an application when paths are undefined."""

    status = Application(algs=[Algorithm()]).run()
    assert  1 == status.code
    assert "List of paths to patient data is empty" == status.reason

def test_application_run_non_default_patient():
    """Test running of an application, with non-default Patient-like class."""

    # Create <paths> list of pseudopaths,
    # and <test_values> dictionary associating random integers to ids.
    n_path = 4
    paths = []
    test_values = {}
    for idx in range(n_path):
        paths.append(Path(f"top_dir/{idx}"))
        test_values[paths[-1].name] = randint(1, 100)

    # Create application for running algorithm that knows about <test_values>.
    opts = {"test_values": test_values}
    alg1 = PyTestAlgorithm(name="alg1", opts=opts)
    app = Application(algs=[alg1])

    assert app.log_level == Defaults().log_level
    assert app.logger.name == "Application"
    assert len(app.algs) == 1
    assert app.algs[0].name == "alg1"
    assert app.algs[0].test_values == test_values

    # Run the application, creating PyTestPatient instance for each path.
    # Additional assertions included in the execute() method
    # of the application's algorithm.
    app.run(paths=paths, PatientClass=PyTestPatient, **opts)

def check_algorithm_method_status_not_okay(alg_cls):
    """
    Check application status after running algorithm <alg_cls>,
    with method returning non-zero status code.
    """
    # Define status code and data path.
    code = 101
    paths = ["test_path"]

    # Create application; check status code and reason after running.
    for reason in ["Unknown reason", None]:
        opts = {"new_status": Status(code=code, reason=reason)}
        alg = alg_cls(opts=opts)
        app = Application(algs=[alg])
        assert app.status.ok()
        app.run(paths=paths)
        assert not app.status.ok()
        assert app.status.code == code
        assert app.status.reason == reason or alg.default_reason

def test_algorithm_initialise_status_not_okay():
    """Test algorithm returning initialise status not okay."""

    # Define Algorithm subclass that sets initialise status from options.
    class TestInitialiseStatus(Algorithm):
    
        def initialise(self):
            self.status = self.new_status
            self.default_reason = (
                    f"Problem initialising algorithm \"{self.name}\"")
            return super().initialise()

    check_algorithm_method_status_not_okay(TestInitialiseStatus)

def test_algorithm_execute_status_not_okay():
    """Test algorithm returning execute status not okay."""

    # Define Algorithm subclass that sets execute status from options.
    class TestExecuteStatus(Algorithm):
    
        def execute(self, patient=None):
            self.status = self.new_status
            self.default_reason = (
                    f"Problem executing algorithm \"{self.name}\" "
                    f"for data path \"{patient.path}\"")
            return super().execute(patient)

    check_algorithm_method_status_not_okay(TestExecuteStatus)

def test_algorithm_finalise_status_not_okay():
    """Test algorithm returning finalise status not okay."""

    # Define Algorithm subclass that sets finalise status from options.
    class TestFinaliseStatus(Algorithm):
    
        def finalise(self):
            self.status = self.new_status
            self.default_reason = (
                    f"Problem finalising algorithm \"{self.name}\"")
            return super().initialise()

    check_algorithm_method_status_not_okay(TestFinaliseStatus)

def create_data_paths(overwrite=False, n_path=6):
    """
    Ensure existence of specified number of data directories, and return paths.

    **Parameters::**
  
    overwrite: bool, default=False
        Delete (overwrite=True) or leave (overwrite=False) any pre-existing
        data directories.
        

    n_path: int, default=6
        Number of data directories required.
    """
    data_dir = Path(fullpath("tmp/data"))
    if overwrite or not data_dir.exists():
        if data_dir.exists():
            rmtree(data_dir)
        data_dir.mkdir(parents=True)
        paths = []
        for idx in range(n_path):
            paths.append(Path(f"{data_dir}/{idx:02}"))
            paths[-1].mkdir()
    else:
        paths = data_dir.glob("0*")

    return sorted(list(paths))
     
def test_get_paths_null():
    """Test get_paths() for null data_locations."""
    assert get_paths() == []

def test_get_paths_str():
    """Test get_paths() for data_locations input as string."""
    paths = create_data_paths(True, 6)
    data_locations = paths[0]
    assert get_paths(data_locations) == [str(paths[0])]

def test_get_paths_all():
    """Test get_paths() for data_locations input as dictionary."""
    paths = create_data_paths()
    data_locations = {paths[0].parent : "0*"}
    assert get_paths(data_locations) == [str(path) for path in paths]

def test_get_paths_to_keep():
    """Test get_paths() with filenames to keep specified."""
    paths = create_data_paths()
    data_locations = {paths[0].parent : "0*"}
    names = [path.name for path in paths]
    to_keep = sample(names, 2)
    assert (get_paths(data_locations, to_keep=to_keep) ==
            [str(path) for path in paths if path.name in to_keep])
    for single_to_keep in to_keep:
        assert (get_paths(data_locations, to_keep=single_to_keep) ==
                [str(path) for path in paths if path.name == single_to_keep])

def test_get_paths_to_exclude():
    """Test get_paths() with filenames to exclude specified."""
    paths = create_data_paths()
    data_locations = {paths[0].parent : "0*"}
    names = [path.name for path in paths]
    to_exclude = sample(names, 2)
    assert (get_paths(data_locations, to_exclude=to_exclude) ==
            [str(path) for path in paths if path.name not in to_exclude])
    for single_to_exclude in to_exclude:
        assert (get_paths(data_locations, to_exclude=single_to_exclude) ==
                [str(path) for path in paths if path.name != single_to_exclude])

def test_status_copy_attributes():
    """Test attribute copying between Status objects."""
    # Create Status objects with different attributes.
    attributes = [{"code": idx, "name": f"name{idx}", "reason": f"reason{idx}"}
                  for idx in range(2)]
    statuses = [Status(**attributes[idx]) for idx in range(2)]

    # Check that initial attributes are as expected.
    for idx in range(2):
        for key, value in attributes[idx].items():
            assert getattr(statuses[idx], key) == value

    # Check that attributes after copying are as expected.
    statuses[0].copy_attributes(statuses[1])
    for idx in range(2):
        for key, value in attributes[1].items():
            assert getattr(statuses[idx], key) == value
