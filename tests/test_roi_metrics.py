'''Test ROI metrics and comparison metrics.'''

import numpy as np
from pytest import approx

from skrt.simulation import SyntheticImage
from skrt.image import _plot_axes, _slice_axes, _axes

views = list(_slice_axes.keys())
methods = ["mask", "contour"]


# Create fake image
centre1 = np.array([5, 4, 5])
centre2 = np.array([5, 6, 5])
side_length = 4
name1 = 'cube1'
name2 = 'cube2'
sim = SyntheticImage((10, 10, 10), origin=(0.5, 0.5, 0.5))
sim.add_cuboid(side_length, centre=centre1, name=name1)
sim.add_cube(side_length, centre=centre2, name=name2)
cube1 = sim.get_roi(name1)
cube2 = sim.get_roi(name2)


def test_mid_idx():
    for view, z in _slice_axes.items():
        for method in "contour", "mask":
            assert abs(cube1.get_mid_idx(view=view, method=method) 
                       - centre1[z]) <= 1

def test_get_indices():
    for method in methods:
        assert cube1.get_indices(method=method) == [
            i for i in range(int(centre1[2] - side_length / 2),
                             int(centre1[2] + side_length / 2))
        ]

def test_on_slice():
    assert cube1.on_slice(view='x-y', idx=centre1[2] - 1)

def test_centroid_global():
    for method in methods:
        assert np.all(np.array(cube1.get_centroid(method=method)) == centre1)

def test_centroid_slice():
    for method in methods:
        for view in views:
            x, y = _plot_axes[view]
            assert np.all(np.array(cube1.get_centroid(
                method=method, view=view, single_slice=True)) == centre1[[x, y]])

def test_centre():
    for method in methods:
        assert np.all(np.array(cube1.get_centre(method=method)) == centre1)

def test_centre_slice():
    for method in methods:
        for view in views:
            x, y = _plot_axes[view]
            assert np.all(np.array(cube1.get_centre(
                single_slice=True, view=view, method=method)) == centre1[[x, y]])

def test_volume():
    assert cube1.get_volume() == side_length ** 3
    assert cube1.get_volume('voxels') == side_length ** 3
    assert cube1.get_volume('ml') == side_length ** 3 / 1000
    vol1 = cube1.get_volume(method="mask")
    assert abs(vol1 - cube1.get_volume(method="contour")) / vol1 < 0.1

def test_area():
    for view in views:
        assert cube1.get_area(view=view) == side_length ** 2
        assert cube1.get_area(sl=1, view=view) is None
        assert cube1.get_area(units="voxels", view=view) == side_length ** 2
        area1 = cube1.get_area(view=view, method="contour")
        assert abs(area1 - cube1.get_area(method="contour", view=view)) \
                / area1 < 0.1

def test_length():
    sides = [4, 2, 6]
    sim.add_cuboid(sides, name="cuboid")
    roi = sim.get_roi("cuboid")
    for i, ax in enumerate(['x', 'y', 'z']):
        for method in methods:
            assert roi.get_length(ax=ax, method=method) == sides[i]
            assert roi.get_length(units='voxels', ax=ax, method=method) \
                    == sides[i]

def test_centroid_distance():
    assert np.all(cube1.get_centroid_distance(cube2) == (centre2 - centre1))
    assert np.all(cube1.get_centroid_distance(cube2) 
                  == -cube2.get_centroid_distance(cube1))

def test_centroid_distance_slice():
    assert np.all(cube1.get_centroid_distance(cube2, single_slice=True, 
                                            idx=cube1.get_mid_idx(),
                                            view='x-y')
                  == (centre2 - centre1)[:2])

def test_abs_centroid_distance():
    assert cube1.get_abs_centroid_distance(cube2) == np.sqrt(
        ((centre2 - centre1) ** 2).sum())
    assert cube1.get_abs_centroid_distance(cube2) \
            == cube2.get_abs_centroid_distance(cube1)

def test_dice_slice():
    assert cube1.get_dice(cube2, single_slice=True, view='x-y', 
                          idx=cube1.get_mid_idx()) == 0.5

def test_dice_contour():
    d1 = cube1.get_dice(cube2, method="contour")
    d2 = cube1.get_dice(cube2, method="mask")
    assert abs(d1 - d2) / d1 < 0.1

def test_dice_contour_slice():
    d1 = cube1.get_dice(cube2, method="contour", single_slice=True)
    d2 = cube1.get_dice(cube2, method="mask", single_slice=True)
    assert abs(d1 - d2) / d1 < 0.1

def test_dice_flattened():
    assert cube1.get_dice(cube2, flatten=True) == 0.5

def test_volume_ratio():
    assert cube1.get_volume_ratio(cube2) == 1

def test_area_ratio():
    assert cube1.get_area_ratio(cube2) == 1

def test_relative_volume_diff():
    assert cube1.get_relative_volume_diff(cube2) == 0

def test_relative_area_diff():
    assert cube1.get_relative_area_diff(cube2) == 0

def test_area_diff():
    assert cube1.get_area_diff(cube2) == 0
    assert cube1.get_area_diff(cube2, flatten=True) == 0

def test_mean_surface_distance():
    assert cube1.get_mean_surface_distance(cube2) == 1
    assert cube2.get_mean_surface_distance(cube1) \
            == cube1.get_mean_surface_distance(cube2)

def test_rms_surface_distance():
    assert cube1.get_rms_surface_distance(cube2) == np.sqrt(5 / 3)

def test_hausdorff_distance():
    assert cube1.get_hausdorff_distance(cube2) == 2

def test_hausdorff_distance_flattened():
    assert cube1.get_hausdorff_distance(cube2, flatten=True) == 2

def test_force_volume():
    v_mask = cube1.get_volume(method="mask")
    assert cube1.get_volume(method="mask", force=True) == v_mask
    assert cube1.get_volume(method="contour", force=True) != v_mask
    v_contour = cube1.get_volume(method="contour")
    assert cube1.get_volume(method="mask", force=False) == v_contour
