'''Test ROI metrics and comparison metrics.'''

import numpy as np
from pytest import approx

from skrt.simulation import SyntheticImage


# Create fake image
centre1 = np.array([5, 4, 5])
centre2 = np.array([5, 6, 5])
side_length = 4
name1 = 'cube1'
name2 = 'cube2'
sim = SyntheticImage((10, 10, 10), origin=(0.5, 0.5, 0.5))
sim.add_cube(4, centre=centre1, name=name1)
sim.add_cube(4, centre=centre2, name=name2)
cube1 = sim.get_roi(name1)
cube2 = sim.get_roi(name2)


def test_mid_idx():
    assert abs(cube1.get_mid_idx() - centre1[2]) <= 1

def test_get_indices():
    assert cube1.get_indices() == [i for i in 
                                   range(int(centre1[2] - side_length / 2),
                                         int(centre1[2] + side_length / 2))
                                  ]

def test_on_slice():
    assert cube1.on_slice(view='x-y', idx=centre1[2] - 1)

def test_centroid_global():
    assert np.all(np.array(cube1.get_centroid()) == centre1)

def test_centroid_slice():
    assert np.all(np.array(cube1.get_centroid(single_slice=True, view='x-y')) == centre1[:2])

def test_centre():
    assert np.all(np.array(cube1.get_centre()) == centre1)

def test_centre_slice():
    assert np.all(np.array(cube1.get_centre(single_slice=True, view='x-y')) == centre1[:2])

def test_volume():
    assert cube1.get_volume() == side_length ** 3
    assert cube1.get_volume('voxels') == side_length ** 3

def test_area():
    assert cube1.get_area() == side_length ** 2
    assert cube1.get_area(sl=1) is None

def test_length():
    for ax in ['x', 'y', 'z']:
        assert cube1.get_length(ax=ax) == side_length
        assert cube1.get_length(units='voxels', ax=ax) == side_length

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

def test_dice():
    assert cube1.get_dice(cube2) == 0.5

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

def test_volume_from_contour():
    vol1 = cube1.get_volume(method="mask")
    vol2 = cube1.get_volume(method="contour")
    assert abs(vol1 - vol2) / vol1 < 0.1

def test_area_from_contour():
    area1 = cube1.get_area(method="mask")
    area2 = cube1.get_area(method="contour")
    assert abs(area1 - area2) / area1 < 0.1

def test_length_from_contour():
    for ax in ["x", "y", "z"]:
        len1 = cube1.get_length(ax=ax, method="mask")
        len2 = cube1.get_length(ax=ax, method="contour")
        assert abs(len1 - len2) / len1 < 0.1

def test_get_indices_from_contour():
    ind1 = cube1.get_indices(method="contour")
    ind2 = cube1.get_indices(method="mask")
    assert ind1 == ind2

def test_centroid_from_contour():
    c1 = cube1.get_centroid(method="mask")
    c2 = cube1.get_centroid(method="contour")
    for i in range(len(c1)):
        assert abs((c1[i] - c2[i]) / c1[i]) < 0.1

def test_2d_centroid_from_contour():
    for view in ["x-y", "y-z", "x-z"]:
        c1 = cube1.get_centroid(single_slice=True, method="mask", view=view)
        c2 = cube1.get_centroid(single_slice=True, method="contour", view=view)
        for i in range(len(c1)):
            assert abs((c1[i] - c2[i]) / c1[i]) < 0.1

def test_force_volume():
    v_mask = cube1.get_volume(method="mask")
    assert cube1.get_volume(method="mask", force=True) == v_mask
    assert cube1.get_volume(method="contour", force=True) != v_mask
    v_contour = cube1.get_volume(method="contour")
    assert cube1.get_volume(method="mask", force=False) == v_contour
