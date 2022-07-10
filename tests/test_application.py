'''Test application classes.'''

from skrt.application import Algorithm, Application
from skrt.core import Defaults

class PyTestAlgorithm(Algorithm):

    def __init__(self, opts=None, name=None, log_level=None):
        # Override default properties, based on contents of opts dictionary.
        self.test_opt1 = None
        super().__init__(opts, name, log_level)

def test_algorithm_creation():

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
