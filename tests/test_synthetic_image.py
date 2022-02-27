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

def test_group():
    '''Test grouping.'''

    sim = SyntheticImage((100, 100, 100))
    sim.add_cube(side_length=40, name="cube", centre=(30, 30, 50),
            intensity=1, group='my_group')
    sim.add_sphere(radius=20, name="sphere", centre=(70, 70, 50),
            intensity=10, group='my_group')
    my_group = sim.get_roi('my_group')
    ss = sim.get_structure_set()
    # Check that cube and sphere are grouped as a single ROI
    assert len(ss.get_roi_names()) == 1
    assert ss.get_roi_names()[0] == 'my_group'

    sim2 = SyntheticImage((100, 100, 100))
    sim2.add_cube(
            side_length=40, name="cube", centre=(30, 30, 50), intensity=1)
    sim2.add_sphere(
            radius=20, name="sphere", centre=(70, 70, 50), intensity=10)
    cube = sim2.get_roi('cube')
    sphere = sim2.get_roi('sphere')
    ss2 = sim2.get_structure_set()

    # Check that grouped cube and sphere have same volume as components
    assert my_group.get_volume() == cube.get_volume() + sphere.get_volume()

    # Check that structure sets for grouped and ungrouped volumes
    # have same extent.
    assert ss.get_extent() == ss2.get_extent()
