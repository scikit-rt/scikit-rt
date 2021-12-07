"""Test the SyntheticImage class."""
import numpy as np

from skrt.simulation import SyntheticImage


def test_shape():
    shape = (10, 40, 20)
    sim = SyntheticImage(shape)
    sim_shape = sim.get_data().shape
    assert sim_shape[0] == shape[1]
    assert sim_shape[1] == shape[0]
    assert sim_shape[2] == shape[2]

def test_reset():
    sim = SyntheticImage((10, 10, 10), intensity=0)
    assert np.all(sim.get_data() == 0)
    sim.add_cube(2, intensity=10)
    assert not np.all(sim.get_data() == 0)
    sim.reset()
    assert np.all(sim.get_data() == 0)

def test_cube():
    sim = SyntheticImage((10, 10, 10), intensity=0)
    length = 2
    sim.add_cube(length, intensity=1)
    assert sim.get_data().sum() == length ** 3

def test_cuboid():
    sim = SyntheticImage((10, 10, 10), intensity=0)
    length = (2, 4, 6)
    sim.add_cube(length, intensity=1)
    assert sim.get_data().sum() == np.product(length)

def test_background_intensity():
    i = 5
    sim = SyntheticImage((10, 10, 10), intensity=i)
    assert sim.get_data().sum() == np.product(sim.shape) * i

def test_noise():
    std = 10
    sim = SyntheticImage((10, 10, 10), noise_std=std, intensity=0)
    assert sim.get_data()[0, 0, 0] != 0
    sim.set_noise_std(0)
    assert sim.get_data()[0, 0, 0] == 0

def test_view():
    sim = SyntheticImage((10, 10, 10))
    sim.view(show=False)

