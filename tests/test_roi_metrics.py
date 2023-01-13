'''Test ROI metrics and comparison metrics.'''

import numpy as np
from pytest import approx
import random

from skrt import StructureSet
from skrt.simulation import SyntheticImage
from skrt.image import _plot_axes, _slice_axes, _axes
from skrt.structures import get_conformity_index

views = list(_slice_axes.keys())
methods = ["mask", "contour"]


# Create fake image
delta_y = 2
delta_y3 = -1
delta_z3 = 2
centre1 = np.array([5, 4, 5])
centre2 = np.array([5, 4 + delta_y, 5])
centre3 = np.array([5, 4 + delta_y3, 5 + delta_z3])
side_length = 4
side_length3 = 6
name1 = 'cube1'
name2 = 'cube2'
name3 = 'cube3'
sim = SyntheticImage((10, 10, 10), origin=(0.5, 0.5, 0.5))
sim.add_cuboid(side_length, centre=centre1, name=name1)
sim.add_cube(side_length, centre=centre2, name=name2)
sim.add_cube(side_length3, centre=centre3, name=name3)
cube1 = sim.get_roi(name1)
cube2 = sim.get_roi(name2)
cube3 = sim.get_roi(name3)

# Slice positions for cube1, cube3, their union, and their intersection.
slices_1 = cube1.get_slice_positions()
slices_3 = cube3.get_slice_positions()
slices_union13 = cube1.get_slice_positions(cube3, method="union")
slices_overlap13 = cube1.get_slice_positions(cube3, method="intersection")

def get_tests13(metric, method):
    """
    Define tests comparing cube1 and cube3.

    **Parameters:**
    metric: str
        Metric for which tests are to be defined.

    method: str
        Method for which tests are to be defined.
    """
    ssval0 = None
    if "dice" == metric:
        ssval = 2 * len(slices_1)**2 / (len(slices_1)**2 + len(slices_3)**2)
        ssval0 = 0
    elif "jaccard" == metric:
        ssval = len(slices_1)**2 / len(slices_3)**2
        ssval0 = 0
    elif "area_ratio" == metric:
        ssval = len(slices_3)**2 / len(slices_1)**2
    elif "area_diff" == metric:
        ssval = len(slices_1)**2 - len(slices_3)**2
    elif "abs_centroid" == metric:
        ssval = abs(delta_y3)
    elif "centroid" == metric:
        ssval = (0, delta_y3)
        ssval0 = (None, None)

    if "by_slice" == method:
        return {
                "left": {pos: (ssval if pos in slices_3 else ssval0)
                    for pos in slices_1},
                "right": {pos: (ssval if pos in slices_1 else ssval0)
                    for pos in slices_3},
                "union": {pos: (ssval if pos in slices_overlap13 else ssval0)
                    for pos in slices_union13},
                "intersection": {pos: ssval for pos in slices_overlap13},
            }

    elif "slice_mean" == method:
        if isinstance(ssval, tuple):
            return {
                    "left": [(len(slices_overlap13) * val) / len(slices_1)
                        for val in ssval],
                    "right": [(len(slices_overlap13) * val) / len(slices_3)
                        for val in ssval],
                    "union": [(len(slices_overlap13) * val) /
                        len(slices_union13) for val in ssval],
                    "intersection": [val for val in ssval],
                }
        else:
            return {
                    "left": (len(slices_overlap13) * ssval) / len(slices_1),
                    "right": (len(slices_overlap13) * ssval) / len(slices_3),
                    "union": (len(slices_overlap13) * ssval)
                    / len(slices_union13),
                    "intersection": ssval,
            }

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

def test_centroid_distance_by_slice():
    """Check slice-by-slice centroid distances."""
    for method, result in get_tests13("centroid", "by_slice").items():
        by_slice_result = cube1.get_centroid_distance(cube3, by_slice=method)
        for pos in set(result).union(set(by_slice_result)):
            assert tuple(by_slice_result[pos]) == result[pos]

def test_centroid_distance_slice_stat():
    """Check mean value of slice-by-slice centroid distances."""
    for method, result in get_tests13("centroid", "slice_mean").items():
        print(cube1.get_centroid_distance(cube3, by_slice=method,
            slice_stat="mean", value_for_none=0))
        assert (cube1.get_centroid_distance(cube3, by_slice=method,
            slice_stat="mean", value_for_none=0) == result)

def test_abs_centroid_distance():
    assert cube1.get_abs_centroid_distance(cube2) == np.sqrt(
        ((centre2 - centre1) ** 2).sum())
    assert cube1.get_abs_centroid_distance(cube2) \
            == cube2.get_abs_centroid_distance(cube1)

def test_abs_centroid_distance_by_slice():
    """Check slice-by-slice magnitudes of centroid distances."""
    for method, result in get_tests13("abs_centroid", "by_slice").items():
        assert (cube1.get_abs_centroid_distance(cube3, by_slice=method)
                == result)

def test_abs_centroid_distance_slice_stat():
    """Check mean value of slice-by-slice magnitudes of centroid distances."""
    for method, result in get_tests13("abs_centroid", "slice_mean").items():
        assert (cube1.get_abs_centroid_distance(cube3, by_slice=method,
            slice_stat="mean", value_for_none=0) == result)

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

def test_dice_by_slice():
    """Check slice-by-slice Dice scores."""
    for method, result in get_tests13("dice", "by_slice").items():
        assert cube1.get_dice(cube3, by_slice=method) == result

def test_dice_slice_stat():
    """Check mean value of slice-by-slice Dice scores."""
    for method, result in get_tests13("dice", "slice_mean").items():
        assert (cube1.get_dice(cube3, by_slice=method, slice_stat="mean")
            == result)

def test_jaccard_slice():
    assert cube1.get_jaccard(cube2, single_slice=True, view='x-y', 
                          idx=cube1.get_mid_idx()) ==  1/3

def test_jaccard_contour():
    d1 = cube1.get_jaccard(cube2, method="contour")
    d2 = cube1.get_jaccard(cube2, method="mask")
    assert abs(d1 - d2) / d1 < 0.1

def test_jaccard_contour_slice():
    d1 = cube1.get_jaccard(cube2, method="contour", single_slice=True)
    d2 = cube1.get_jaccard(cube2, method="mask", single_slice=True)
    assert abs(d1 - d2) / d1 < 0.1

def test_jaccard_flattened():
    assert cube1.get_jaccard(cube2, flatten=True) == 1/3

def test_jaccard_by_slice():
    """Check slice-by-slice Jaccard indices."""
    for method, result in get_tests13("jaccard", "by_slice").items():
        assert cube1.get_jaccard(cube3, by_slice=method) == result

def test_jaccard_slice_stat():
    """Check mean value of slice-by-slice Jaccard indices."""
    for method, result in get_tests13("jaccard", "slice_mean").items():
        assert (cube1.get_jaccard(cube3, by_slice=method, slice_stat="mean")
            == result)

def test_volume_ratio():
    assert cube1.get_volume_ratio(cube2) == 1

def test_area_ratio():
    assert cube1.get_area_ratio(cube2) == 1

def test_area_ratio_by_slice():
    """Check slice-by-slice area ratios."""
    for method, result in get_tests13("area_ratio", "by_slice").items():
        assert cube1.get_area_ratio(cube3, by_slice=method) == result

def test_area_ratio_slice_stat():
    """Check mean value of slice-by-slice area ratios."""
    for method, result in get_tests13("area_ratio", "slice_mean").items():
        assert (cube1.get_area_ratio(cube3, by_slice=method, slice_stat="mean",
            value_for_none=0) == result)

def test_relative_volume_diff():
    assert cube1.get_relative_volume_diff(cube2) == 0

def test_relative_area_diff():
    assert cube1.get_relative_area_diff(cube2) == 0

def test_area_diff():
    assert cube1.get_area_diff(cube2) == 0
    assert cube1.get_area_diff(cube2, flatten=True) == 0

def test_area_diff_by_slice():
    """Check slice-by-slice area differences."""
    for method, result in get_tests13("area_diff", "by_slice").items():
        assert cube1.get_area_diff(cube3, by_slice=method) == result

def test_area_diff_slice_stat():
    """Check mean value of slice-by-slice area differences."""
    for method, result in get_tests13("area_diff", "slice_mean").items():
        assert (cube1.get_area_diff(cube3, by_slice=method, slice_stat="mean",
            value_for_none=0) == result)

def test_mean_surface_distance():
    assert cube1.get_mean_surface_distance(cube2) == 1
    assert cube2.get_mean_surface_distance(cube1) \
            == cube1.get_mean_surface_distance(cube2)

def test_rms_surface_distance():
    if False:
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

def test_geometry_table_html():
    ss = sim.get_structure_set()
    html = ss.get_geometry(html=True)
    assert isinstance(html, str)

def test_comparison_table_html():
    ss = sim.get_structure_set()
    html = ss.get_comparison(html=True)
    assert isinstance(html, str)

def test_surface_distance():
    '''Test calculations of surface distance using concentric spheres.'''
    random.seed(1)
    n_x, n_y, n_z = (80, 80, 80)
    r_min = 10
    r_max = 35
    n_test = 2
    # Define acceptable level of agreement agreement
    # between true and calculated surface distances.
    # Don't expect exact agreement, because of voxelisation for calculations.
    precision = 2

    # Run test required number of times.
    for idx_test in range(n_test):

        # Create synthetic image,
        # then add two ROIs in the form of concentric spheres.
        sim = SyntheticImage((n_x, n_y, n_z))
        r_values = {}
        for idx in range(2):
            name = f"sphere{idx}"
            r = random.uniform(r_min, r_max)
            sim.add_sphere(radius=r, name=name, centre=sim.get_centre(),
                intensity=10)
            r_values[name] = r
        ss = sim.get_structure_set()
        spheres = list(ss.get_rois())
        assert len(spheres) == 2

        # Test calculations of unsigned symmetric distances.
        voxel_size = (1.0, 1.0, 1.0)
        delta_r1 = abs(r_values[spheres[1].name] - r_values[spheres[0].name])
        delta_r2 = spheres[0].get_mean_surface_distance(
                spheres[1], signed=False, connectivity=1, in_slice=False,
                voxel_size=(voxel_size), symmetric=True)
        assert delta_r1 == approx(delta_r2, abs=precision)

        sds = spheres[0].get_surface_distances(
                spheres[1], signed=False, connectivity=1, in_slice=False,
                voxel_size=(voxel_size), symmetric=True)
        assert sds.min() == approx(delta_r2, abs=precision)
        assert sds.max() == approx(delta_r2, abs=precision)

        # Test calculations of signed one-way distances.
        # Also test modification of the voxel size.
        voxel_size = (0.8, 0.8, 0.8)
        delta_r1 = r_values[spheres[1].name] - r_values[spheres[0].name]
        delta_r2 = spheres[0].get_mean_surface_distance(
                spheres[1], signed=True, connectivity=1, in_slice=False,
                voxel_size=(voxel_size), symmetric=False)
        assert delta_r1 == approx(delta_r2, abs=precision)
        delta_r3 = spheres[1].get_mean_surface_distance(
                spheres[0], signed=True, connectivity=1, in_slice=False,
                voxel_size=(voxel_size), symmetric=False)
        assert delta_r1 == approx(-delta_r3, abs=precision)

        sds = spheres[0].get_surface_distances(
                spheres[1], signed=True, connectivity=1, in_slice=False,
                voxel_size=(voxel_size), symmetric=False)
        assert sds.min() == approx(delta_r2, abs=precision)
        assert sds.max() == approx(delta_r2, abs=precision)

def test_mean_distance_to_conformity():
    '''Test calculation of mean distance to conformity.'''
    conformity = cube1.get_mean_distance_to_conformity(cube2)
    assert conformity.n_voxel == 2 * side_length**3 - delta_y * side_length**2
    assert conformity.voxel_size == cube1.voxel_size
    distances = np.array([dy + 1 for dy in range(delta_y)
            for idx in range(int(side_length)**2)])
    mean_distance = distances.sum() / conformity.n_voxel
    assert conformity.mean_under_contouring == mean_distance
    assert conformity.mean_over_contouring == mean_distance
    assert conformity.mean_distance_to_conformity == 2 * mean_distance

def test_conformity_index():
    """Test calculations of conformity index."""
    # Create test image, featuring overlapping cubes.
    sim2 = SyntheticImage((20, 20, 20), origin=(0.5, 0.5, 0.5))
    sim2.add_cube(8, centre=(10, 10, 8), name="cube1")
    sim2.add_cube(8, centre=(10, 10, 9), name="cube2")
    sim2.add_cube(8, centre=(10, 10, 11), name="cube3")
    sim2.add_cube(8, centre=(10, 10, 12), name="cube4")

    # For two ROIs, all conformity indices reduce to the Jaccard index.
    rois = [sim2.get_roi(f"cube{idx}") for idx in [1, 4]]
    jci = 4/12
    # Calculation for list of ROIs.
    ci = get_conformity_index(rois, "all")
    assert jci == ci.common
    assert jci == ci.gen
    assert jci == ci.pairs

    # For two ROIs, all conformity indices reduce to the Jaccard index.
    rois = [sim2.get_roi(f"cube{idx}") for idx in [2, 3]]
    jci = 6/10
    # Calculation for structure set.
    ci = StructureSet(rois).get_conformity_index(ci_type="all")
    assert jci == ci.common
    assert jci == ci.gen
    assert jci == ci.pairs

    # Conformity indices relative to the four ROIs defined.
    rois = [sim2.get_roi(f"cube{idx}") for idx in [1, 2, 3, 4]]
    ci_common = 4/12
    ci_pairs = (7/9 + 5/11 + 4/12 + 6/10 + 5/11 + 7/9) / 6
    ci_gen = (7 + 5 + 4 + 6 + 5 + 7) / (9 + 11 + 12 + 10 + 11 + 9)
    ci = StructureSet(rois).get_conformity_index(ci_type="all")
    assert ci.common == ci_common
    assert ci.gen == ci_gen
    assert ci.pairs == ci_pairs
    assert get_conformity_index(rois, "common") == ci_common
    assert get_conformity_index(rois, "pairs") == ci_pairs
    assert get_conformity_index(rois, "gen") == ci_gen

def test_intersection_union_size():
    """Check intersection, union, mean size for pairs of ROIs."""
    # Calculate expected values of intersection, union, mean size
    # for pair of cubes.
    intersection0 = ((side_length - delta_y) * side_length * side_length)
    union0 = ((side_length + delta_y) * side_length * side_length)
    size0 = side_length * side_length * side_length

    # Calculate intersection, union, mean size from contours.
    # (Contours created from cube masks, as here, have limited accuracy.)
    intersection, union, size = cube1.get_intersection_union_size(
            cube2, method="contour")
    precision = 2
    assert intersection == approx(intersection0, abs=precision)
    assert union == approx(union0, abs=precision)
    assert size == approx(size0, abs=precision)

    # Calculate intersection, union, mean size from masks.
    intersection, union, size = cube1.get_intersection_union_size(
            cube2, method="mask")
    assert intersection == intersection0
    assert union == union0
    assert size == size0

def test_intersection_union_size_split_roi():
    """Check intersection, union, mean size when one ROI is split."""

    # Create synthetic image, featuring cuboid and two non-overlapping cubes,
    # contained within the cuboid.
    sim1 = SyntheticImage((10, 10, 10), origin=(0.5, 0.5, 0.5))
    cuboid_sides = np.array([4, 4, 10])
    cube_side = cuboid_sides[0]
    sim1.add_cuboid(cuboid_sides, name="cuboid", centre=(5, 5, 5))
    sim1.add_cube(cube_side, name="cube1", centre=(5, 5, 2))
    sim1.add_cube(cube_side, name="cube2", centre=(5, 5, 8))

    # Extracts ROIs for cuboid, and for split_roi combining the two cubes.
    structure_set = sim1.get_structure_set()
    cuboid = structure_set.get_roi("cuboid")
    split_roi = StructureSet(
            structure_set.get_rois(["cube1", "cube2"])).combine_rois()

    # Calculate volumes of cuboid and split_roi from geometric properties.
    cuboid_volume_theory = np.prod(cuboid_sides)
    split_roi_volume_theory = 2 * cube_side**3

    # For different calculation methods, check cuboid-split_roi values
    # for intersection, union, mean size.
    # As the split_roi is contained within the cuboid:
    # - volume of union equals volume of cuboid;
    # - volume of intersection equals volume of split_roi.
    for method in ["mask", "contour"]:
        # Check ROI volumes.
        # Volumes from "mask" method should be the same as
        # the volumes calculated from geometric properties.  
        # Volumes from "contour" method should be slightly smaller,
        # because of corner rounding in converting from masks to contours.
        cuboid_volume = cuboid.get_volume(method=method)
        split_roi_volume = split_roi.get_volume(method=method)
        if "mask" == method:
            assert cuboid_volume == cuboid_volume_theory
            assert split_roi_volume == split_roi_volume_theory
        else:
            for volume_ratio in [cuboid_volume / cuboid_volume_theory,
                    split_roi_volume / split_roi_volume_theory]:
                assert volume_ratio < 1.000
                assert volume_ratio > 0.965

        # Check intersection, union and mean size
        intersection, union, size = cuboid.get_intersection_union_size(
                split_roi, method=method)
        assert intersection == split_roi_volume
        assert union == cuboid_volume
        assert size == 0.5 * (cuboid_volume + split_roi_volume)
