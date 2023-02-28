"""
Test legacy QuickViewer code.

Tests migrated from tests/test_struct_metrics.py at:
    https://github.com/hlpullen/quickviewer

Functions to test the calculation of structure geometric properties and 
structure comparison metrics.
"""

import numpy as np
import pytest
from pytest import approx

from skrt.simulation import SyntheticImage
from skrt.viewer.core import Struct, StructComparison


# Make pair of cubes offset by 1mm in x and y
dx = 1
dy = 1
cx, cy, cz = 50.5, 50.5, 15.5
gn = SyntheticImage((100, 100, 30))
gn.add_cube(side_length=10, centre=(cx, cy, cz), name="cube1")
gn.add_cube(side_length=10, centre=(cx + dx, cy + dy, cz), name="cube2")
roi_cube1 = gn.get_roi("cube1")
cube1 = Struct(roi_cube1.get_mask()[::-1, ::-1, :], name=roi_cube1.name,
               origin=roi_cube1.get_origin(),
               voxel_sizes=roi_cube1.get_voxel_size())
roi_cube2 = gn.get_roi("cube2")
cube2 = Struct(roi_cube2.get_mask()[::-1, ::-1, :], name=roi_cube2.name,
               origin=roi_cube2.get_origin(),
               voxel_sizes=roi_cube2.get_voxel_size())

def test_centroid():
    centroid = cube1.centroid()
    assert centroid[0] == cx
    assert centroid[1] == cy
    assert centroid[2] == cz


def test_centroid_slice():
    assert np.array_equal(
        cube1.centroid("x-y", sl=cube1.mid_slice("x-y")),
        [cx, cy])
    assert np.array_equal(
        cube1.centroid("y-z", sl=cube1.mid_slice("y-z")),
        [cz, cy])
    assert np.array_equal(
        cube1.centroid("x-z", sl=cube1.mid_slice("x-z")),
        [cz, cx])


def test_volume():
    assert cube1.get_volume() == 1000
    assert cube1.get_volume("voxels") == 1000


def test_area():
    for view in ["x-y", "y-z", "x-z"]:
        assert cube1.get_area(view, sl=cube1.mid_slice(view)) == 100


def test_length():
    assert cube1.struct_extent() == [10, 10, 10]


def test_extent_slice():
    for view in ["x-y", "y-z", "x-z"]:
        assert cube1.struct_extent(view, cube1.mid_slice(view)) == [10, 10]


def test_centre():
    assert np.array_equal(
        cube1.get_centre("x-y", sl=cube1.mid_slice("x-y")),
        [cx, cy])
    assert np.array_equal(
        cube1.get_centre("y-z", sl=cube1.mid_slice("y-z")),
        [cz, cy])
    assert np.array_equal(
        cube1.get_centre("x-z", sl=cube1.mid_slice("x-z")),
        [cz, cx])


comp = StructComparison(cube1, cube2)


def test_centroid_distance():
    assert comp.abs_centroid_distance() == approx(np.sqrt(2))


def test_centroid_distance_slice():
    assert np.array_equal(
        comp.centroid_distance("x-y", sl=cube1.mid_slice("x-y")),
        [-dx, -dy]
    )
    assert np.array_equal(
        comp.centroid_distance("y-z", sl=cube1.mid_slice("y-z")),
        [0, -dy]
    )
    assert np.array_equal(
        comp.centroid_distance("x-z", sl=cube1.mid_slice("x-z")),
        [0, -dx]
    )


def test_dice():
    assert comp.dice() == 0.81
    assert comp.dice("x-y", cube1.mid_slice("x-y")) == 0.81
    for view in ["y-z", "x-z"]:
        assert comp.dice(view, cube1.mid_slice(view)) == 0.9


def test_vol_comparison():
    assert comp.vol_ratio() == 1
    assert comp.relative_vol() == 0
    

def test_area_comparison():
    for view in ["x-y", "y-z", "x-z"]:
        assert comp.area_ratio(view, cube1.mid_slice(view)) == 1
        assert comp.relative_area(view, cube1.mid_slice(view)) == 0


def test_extent_ratio():
    for view in ["x-y", "y-z", "x-z"]:
        assert comp.extent_ratio(view, cube1.mid_slice(view)) == [1, 1]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_hausdorff():
    assert comp.hausdorff_distance() == approx(np.sqrt(2))
    assert comp.hausdorff_distance("x-y", cube1.mid_slice("x-y")) \
            == approx(np.sqrt(2))
    for view in ["y-z", "x-z"]:
        assert comp.hausdorff_distance(view, cube1.mid_slice(view)) \
                == approx(1)


