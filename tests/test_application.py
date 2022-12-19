'''Test application classes.'''
from pathlib import Path
from random import randint, sample, seed
from shutil import rmtree

from skrt.application import Algorithm, Application, get_paths
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

def test_application_run():
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


def create_data_paths(overwrite=False, n_path=6):

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
    assert get_paths() == []

def test_get_paths_all():
    paths = create_data_paths(True, 6)
    data_locations = {paths[0].parent : "0*"}
    assert get_paths(data_locations) == [str(path) for path in paths]

def test_get_paths_to_keep():
    paths = create_data_paths()
    data_locations = {paths[0].parent : "0*"}
    names = [path.name for path in paths]
    to_keep = sample(names, 2)
    assert (get_paths(data_locations, to_keep=to_keep) ==
            [str(path) for path in paths if path.name in to_keep])

def test_get_paths_to_exclude():
    paths = create_data_paths()
    data_locations = {paths[0].parent : "0*"}
    names = [path.name for path in paths]
    to_exclude = sample(names, 2)
    assert (get_paths(data_locations, to_exclude=to_exclude) ==
            [str(path) for path in paths if path.name not in to_exclude])
