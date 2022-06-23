"""Tests for the ROI and StructureSet classes."""

import fnmatch
import math
import os
import random
import shutil
import pandas as pd
import pathlib
import pytest
import numpy as np
import matplotlib.colors

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from skrt.core import fullpath
from skrt.simulation import SyntheticImage
from skrt.structures import contour_to_polygon, polygon_to_contour, \
        StructureSet, ROI, interpolate_points_single_contour


# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# Create synthetic structure set
sim = SyntheticImage((100, 100, 100))
sim.add_cube(side_length=40, name="cube", centre=sim.get_centre(), intensity=1)
sim.add_sphere(radius=20, name="sphere", centre=sim.get_centre(), intensity=10)
structure_set = sim.get_structure_set()
cube = structure_set.get_roi("cube")

def test_structure_set_from_sythetic_image():
    assert isinstance(structure_set, StructureSet)

def test_write_nifti():
    nii_dir = "tmp/nii_structs"
    if os.path.exists(nii_dir):
        shutil.rmtree(nii_dir)
    structure_set.write(outdir="tmp/nii_structs")
    assert len(os.listdir(nii_dir)) == 2

def test_write_txt():
    txt_dir = "tmp/txt"
    if os.path.exists(txt_dir):
        shutil.rmtree(txt_dir)
    structure_set.write(outdir="tmp/txt", ext='.txt')
    assert len(os.listdir(txt_dir)) == 2

def test_get_rois():
    assert len(structure_set.get_rois()) == 2

def test_get_roi_names():
    names = structure_set.get_roi_names()
    assert len(names) == 2
    assert "cube" in names
    assert "sphere" in names

def test_get_dict():
    sdict = structure_set.get_roi_dict()
    assert len(sdict) == 2
    assert set(sdict.keys()) == set(["cube", "sphere"])

def test_get_roi():
    assert isinstance(cube, ROI)
    assert cube.name == "cube"
    # ROI from non-DICOM source
    assert cube.number == None

def test_structure_set_from_rois():
    """Test creation of structure set from ROIs."""

    structure_set2 = StructureSet(structure_set.get_rois())
    assert isinstance(structure_set2, StructureSet)
    sdict1 = structure_set.get_roi_dict()
    sdict2 = structure_set2.get_roi_dict()
    assert len(sdict1.keys()) > 0 
    for key in sdict1:
        assert key in sdict2

def test_rename():
    new_names = {
        "cube2": ["cube", "test"],
        "sphere2": "spher*"
    }
    structure_set.rename_rois(new_names)
    assert structure_set.get_rois()[0].get_name() != structure_set.get_rois()[0].get_name(original=True)
    assert set(structure_set.get_roi_names()) == set(new_names.keys())

    # Check that original names are kept
    n_match = 0
    for roi in structure_set:
        if roi.get_name() in new_names:
            match = False
            for old_name in new_names[roi.get_name()]:
                if fnmatch.fnmatch(roi.get_name(original=True).lower(), old_name.lower()):
                    match = True
                    n_match += 1
                    break
            assert match
    assert n_match == len(new_names.keys())

    old_names = {
        "cube": "cube2",
        "sphere": "sphere2"
    }
    structure_set.rename_rois(old_names)
    assert set(structure_set.get_roi_names()) == set(old_names.keys())

def test_copy_rename():
    """Test copying with ROI renaming."""

    new_names = {"cube3": "cube"}
    structure_set2 = structure_set.filtered_copy(new_names, name="copy", keep_renamed_only=True)
    assert len(structure_set2.get_rois()) == 1
    assert structure_set2.get_roi_names() == ["cube3"]
    assert structure_set2.name == "copy"
    structure_set.rename_rois({"cube": "cube3"})

def test_copy_remove():
    """Test copying with ROI removal."""

    structure_set2 = structure_set.filtered_copy(to_remove="cube")
    assert structure_set2.get_roi_names() == ["sphere"]

def test_copy_keep():
    """Test copying with keeping only certain ROIs."""

    structure_set2 = structure_set.filtered_copy(to_keep="sphere")
    assert structure_set2.get_roi_names() == ["sphere"]

def test_clone():
    """Test cloning; check that ROIs are fully copied."""

    sphere1 = structure_set.get_roi("sphere")
    sphere1.create_mask()
    structure_set2 = structure_set.clone()
    sphere2 = structure_set2.get_roi("sphere")
    sphere1.set_color("red")
    sphere2.set_color("blue")
    assert sphere1.color != sphere2.color
    assert sphere1 is not sphere2
    sphere2.transform(translation=(3, 3, 0))
    assert not np.all(sphere1.get_mask() == sphere2.get_mask())

def test_init_from_structure_set():
    sphere1 = structure_set.get_roi("sphere")
    sphere1.create_mask()
    structure_set2 = StructureSet(structure_set)
    sphere2 = structure_set2.get_roi("sphere")
    sphere1.set_color("red")
    sphere2.set_color("blue")
    assert sphere1.color != sphere2.color
    assert sphere1 is not sphere2
    sphere2.transform(translation=(3, 3, 0))
    assert not np.all(sphere1.get_mask() == sphere2.get_mask())

def test_filtered_copy_no_copy_data():
    """Test copying but using same data for ROIs."""

    # First check that data is copied correctly
    structure_set2 = structure_set.filtered_copy(copy_roi_data=True)
    assert structure_set2.get_rois()[0].get_mask() \
            is not structure_set.get_rois()[0].get_mask()

    # Check that data is passed by reference but new ROI object is made
    structure_set2 = structure_set.filtered_copy(copy_roi_data=False)
    assert structure_set2.get_rois()[0] is not structure_set.get_rois()[0]
    assert structure_set2.get_rois()[0].get_mask() \
            is structure_set.get_rois()[0].get_mask()

def test_read_nii():
    nii_dir = "tmp/nii_structs"
    structs_from_nii = StructureSet(nii_dir)
    assert len(structs_from_nii.get_rois()) == 2
    assert set(structure_set.get_roi_names()) \
            == set(structs_from_nii.get_roi_names())

def test_set_image():
    nii_dir = "tmp/nii_structs"
    structs_from_nii = StructureSet(nii_dir)
    sim2 = SyntheticImage((110, 110, 110))
    structs_from_nii.set_image(sim2)
    assert len(structs_from_nii.get_rois()) == 2
    for roi in structs_from_nii.get_rois():
        assert roi.get_mask().shape == sim2.get_data().shape

def test_plot_contour():
    cube.plot(plot_type="contour", show=False)

def test_plot_centroid():
    cube.plot(plot_type="centroid", show=False)

def test_plot_mask():
    cube.plot(plot_type="mask", show=False)

def test_plot_filled():
    cube.plot(plot_type="filled", show=False)

def test_get_geometry():
    geom = structure_set.get_geometry()
    assert isinstance(geom, pd.DataFrame)
    assert geom.shape[0] == 2

def test_get_comparison_pairs():
    """Get comparison pairs with self; should return each ROI paired with every
    other without duplicates."""

    pairs = structure_set.get_comparison_pairs()
    n_rois = len(structure_set.get_rois())
    assert len(pairs) == (n_rois ** 2 - n_rois) / 2
    assert len(pairs[0]) == 2

def test_get_comparison_pairs_with_other():
    """Get comparison pairs with another StructureSet; should return each ROI 
    paired with each ROI with a matching names."""

    structure_set2 = StructureSet("tmp/nii_structs")
    pairs = structure_set.get_comparison_pairs(structure_set2)
    assert len(pairs) == 2
    assert pairs[0][0].name == pairs[0][1].name
    assert pairs[1][0].name == pairs[1][1].name

def test_get_comparison_pairs_all():
    """Get comparison pairs with another StructureSet using the 'all' method; 
    should return every ROI in one set paired with every ROI in the other, 
    regardless of name match."""

    structure_set2 = StructureSet("tmp/nii_structs")
    pairs = structure_set.get_comparison_pairs(structure_set2, "all")
    assert len(pairs) == len(structure_set.get_rois()) \
            * len(structure_set2.get_rois())

def test_compare_with_own_consensus():
    """Get comparison of each ROI with consensus of all others."""

    pairs = structure_set.get_comparison_pairs(comp_type="consensus",
                                               consensus_type="sum")
    assert len(pairs) == len(structure_set.get_rois())
    assert pairs[0][1] == structure_set.get_sum(exclude=pairs[0][0].name)

def test_compare_with_other_consensus():
    """Get comparison of each ROI with consensus of another structure set."""

    structure_set2 = StructureSet("tmp/nii_structs")
    pairs = structure_set.get_comparison_pairs(structure_set2,
                                               comp_type="consensus",
                                               consensus_type="sum")
    assert len(pairs) == len(structure_set.get_rois())
    assert pairs[0][1] == structure_set2.get_sum()
    assert pairs[1][1] == structure_set2.get_sum()

def test_plot_consensus():
    to_exclude = structure_set.get_rois()[-1].name
    for include_image in [True, False]:
        structure_set.plot(consensus_type="overlap", 
                           exclude_from_consensus=to_exclude,
                           show=False,
                           include_image=include_image)

def test_get_comparison():
    comp = structure_set.get_comparison()
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape[0] == len(structure_set.get_comparison_pairs())

def test_plot_comparisons():
    plot_dir = "tmp/struct_plots"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    structure_set2 = StructureSet("tmp/nii_structs")
    structure_set2.plot_comparisons(outdir=plot_dir, show=False)
    assert len(os.listdir(plot_dir)) == len(structure_set2.get_comparison_pairs())
    
def compare_rois(roi0, roi1):
    '''Compare two rois.'''

    # Check basic characteristics
    assert roi0 is not roi1
    assert roi0.color == roi1.color
    assert roi0.get_centroid().all() == roi1.get_centroid().all()

    # Check number of planes.
    roi0_keys = list(roi0.get_contours().keys())
    roi1_keys = list(roi1.get_contours().keys())
    assert len(roi0_keys) == len(roi1_keys)

    roi0_keys.sort()
    roi1_keys.sort()

    for i in range(len(roi0_keys)):

        # Check plane z-coordinates.
        assert roi0_keys[i] == roi1_keys[i]

        # Check number of contours in plane.
        contours0 = roi0.get_contours()[roi0_keys[i]]
        contours1 = roi1.get_contours()[roi1_keys[i]]
        assert len(contours0) == len(contours1)

        # Check contour points.
        for j in range(len(contours0)):
            assert contours0[j].all() == contours1[j].all()

def test_write_dicom():
    '''Test writing of structure set to dicom.'''
    dcm_dir = "tmp/dcm_structs"
    if os.path.exists(dcm_dir):
        shutil.rmtree(dcm_dir)
    structure_set.write(outdir=dcm_dir, ext='dcm')
    assert len(os.listdir(dcm_dir)) == 1

def test_dicom_dataset():
    '''Check that structure set written to dicom matches original.'''
    dcm_dir = "tmp/dcm_structs1"
    if os.path.exists(dcm_dir):
        shutil.rmtree(dcm_dir)
    structure_set.write(outdir=dcm_dir, ext='dcm')
    structure_set_dcm = StructureSet(dcm_dir)
    assert structure_set.get_roi_names() == structure_set_dcm.get_roi_names()

    for name in structure_set.get_roi_names():
        roi0 = structure_set.get_roi(name)
        roi1 = structure_set_dcm.get_roi(name)
        compare_rois(roi0, roi1)

def test_write_roi_to_dicom():
    '''Check that individual rois written to dicom match originals.'''
    dcm_dir = "tmp/dcm_structs2"
    if os.path.exists(dcm_dir):
        shutil.rmtree(dcm_dir)
    rois = list(structure_set.get_rois())
    assert len(rois) == 2

    for i in range(len(rois)):
        rois[i].write(outdir=dcm_dir, ext='dcm', overwrite=True)
        assert len(os.listdir(dcm_dir)) == 1

        structure_set_dcm = StructureSet(dcm_dir)
        rois_dcm = list(structure_set_dcm.get_rois())

        assert len(rois_dcm) == 1

        compare_rois(rois[i], rois_dcm[0])

def test_write_roi_with_source_to_dicom():
    '''Check that adding rois to dicom file gives original structure set.'''
    dcm_dir = "tmp/dcm_structs3"
    if os.path.exists(dcm_dir):
        shutil.rmtree(dcm_dir)
    rois = list(structure_set.get_rois())
    assert len(rois) == 2

    rois[0].write(outdir=dcm_dir, ext='dcm', overwrite=True)
    assert len(os.listdir(dcm_dir)) == 1
    rois[1].write(outdir=dcm_dir, ext='dcm', header_source=dcm_dir,
            overwrite=False)
    filenames = os.listdir(dcm_dir)
    filenames.sort()
    assert len(filenames) == 2

    dcm_path = os.path.join(dcm_dir, filenames[-1])
    structure_set_dcm = StructureSet(dcm_path)
    rois_dcm = list(structure_set_dcm.get_rois())

    assert len(rois_dcm) == 2

    for name in structure_set.get_roi_names():
        roi0 = structure_set.get_roi(name)
        roi1 = structure_set_dcm.get_roi(name)
        compare_rois(roi0, roi1)

def test_init_from_roi():
    sphere1 = structure_set.get_roi("sphere")
    sphere2 = ROI(sphere1)
    sphere1.set_color("red")
    sphere2.set_color("blue")
    assert sphere1.color != sphere2.color

def test_roi_from_image_threshold():
    roi = ROI(sim.get_image(), mask_threshold=5)  
    assert roi.get_area() == sim.get_roi("sphere").get_area()

def test_roi_no_image_with_geom():
    """Check that an ROI object can be created from contours only plus 
    geometric info."""

    new_roi = ROI(cube.get_contours(),
                  origin=cube.get_origin(),
                  voxel_size=cube.get_voxel_size(),
                  shape=cube.get_mask().shape
                 )
    assert np.all(new_roi.get_mask() == cube.get_mask())

def test_roi_from_polygons():
    """Check that an ROI object can be created from shapely polygons."""

    new_roi = ROI(cube.get_polygons(),
                  origin=cube.get_origin(),
                  voxel_size=cube.get_voxel_size(),
                  shape=cube.get_mask().shape
                 )
    assert np.all(new_roi.get_mask() == cube.get_mask())

    df = StructureSet(new_roi).get_comparison(StructureSet(cube))
    # Dice score may not be exactly 1 because of rounding errors
    assert df['dice'][0] == pytest.approx(1.0, abs=1.e-3)

def test_null_roi():
    roi = ROI()
    assert(type(roi).__name__ == "ROI")
    assert(roi.affine is None)
    assert(roi.contours == {})
    assert(roi.custom_color is False)
    assert(roi.image is None)
    assert(roi.input_contours == None)
    assert(roi.kwargs == {})
    assert(roi.contours == {})
    assert(roi.loaded_mask is False)
    assert(roi.origin is None)
    assert(roi.original_name is None)
    assert(roi.source_type is None)
    assert(roi.shape is None)
    assert(roi.source is None)
    assert(roi.title is None)
    assert(roi.voxel_size is None)

def test_null_structure_set():
    ss = StructureSet()
    assert(ss.date == "")
    assert(ss.files == [])
    assert(ss.image is None)
    assert(ss.loaded is True)
    assert(ss.multi_label is False)
    assert(ss.names is None)
    assert(ss.path == "")
    assert(ss.rois == [])
    assert(ss.sources == [])
    assert(ss.subdir == "")
    assert(ss.time == "")
    assert(ss.timestamp =="")
    assert(ss.to_keep is None)
    assert(ss.to_remove is None)

def test_contour_to_polygon():
    # Create self-intersecting polygon
    contour = [(0, 0), (0, 2), (2, 2), (1.9, 2.1), (1.9, 0), (0, 0)]
    p1 = Polygon(contour)
    assert 'self-intersection' in explain_validity(p1).lower()
    p2 = contour_to_polygon(contour)
    assert 'self-intersection' not in explain_validity(p2).lower()

def test_contour_to_polygon_to_contour():
    '''Test converting from contour to polygon and back.'''
    for key, contours in cube.get_contours().items():
        for contour1 in contours:
            polygon = contour_to_polygon(contour1)
            contour2 = polygon_to_contour(polygon)
            n_point1 = len(contour1)
            n_point2 = len(contour2)
            assert n_point1 == n_point2
            assert contour1.all() == contour2.all()

def test_dummy_image():
    """Test setting an ROI's image to a dummy image with a given shape."""

    roi2 = cube.clone()
    nx = [20, 200]
    ny = [30, 150]
    for x, y in zip(nx, ny):
        roi2.set_image_to_dummy(shape=(x, y))
        assert roi2.image.get_data().shape[1] == x
        assert roi2.image.get_data().shape[0] == y

def test_dummy_image_from_voxel_size():
    """Test setting an ROI's image to a dummy image with given voxel sizes."""
    
    vx = 1.5
    vy = 1
    roi2 = cube.clone()
    roi2.set_image_to_dummy(voxel_size=(vx, vy))
    assert roi2.image.voxel_size[0] == vx
    assert roi2.image.voxel_size[1] == vy

def test_forced_mask_recreation():
    """Recreate a contour ROI's mask with a given shape or voxel size."""

    roi2 = ROI(cube.get_contours("x-y"))
    nx = [20, 200]
    ny = [30, 150]
    for x, y in zip(nx, ny):
        roi2.create_mask(force=True, shape=(x, y))
        assert roi2.get_mask().shape[1] == x
        assert roi2.get_mask().shape[0] == y

    vx = 1.5
    vy = 1
    roi2.create_mask(force=True, voxel_size=(vx, vy))
    assert roi2.mask.get_voxel_size()[0] == vx
    assert roi2.mask.get_voxel_size()[1] == vy

def test_overlap_level():
    """Create mask from contours using different overlap requirements."""

    # First set tight overlap level in __init__
    roi = ROI(sim.get_roi("sphere").get_contours(), overlap_level=1)
    roi.create_mask()

    # Mask area should be smaller than contour area with this requirement
    assert roi.get_area(method="mask") < roi.get_area(method="contour")

    # Force mask recreation with overlap_level set in create_mask()
    level = 0.1
    prev_mask = roi.get_mask()
    roi.create_mask(force=True, overlap_level=level)
    assert not np.all(roi.get_mask() == prev_mask)
    assert roi.overlap_level == level

    # Mask area should be larger than contour area with loose overlap
    assert roi.get_area(method="mask") > roi.get_area(method="contour")

def test_slice_number_conversions():
    """For an ROI with no image, check position-contour-index conversion works."""
    roi2 = ROI(cube.get_contours("x-y"))
    assert roi2.pos_to_idx(roi2.idx_to_pos(0, "z"), "z") == 0
    assert roi2.pos_to_slice(roi2.slice_to_pos(1, "z"), "z") == 1
    assert roi2.slice_to_idx(roi2.idx_to_slice(0, "z"), "z") == 0

                                  
def test_slice_number_conversions_image():
    """For an ROI with an image, check position-contour-index conversion works."""

    roi2 = ROI(cube.get_contours("x-y"))
    roi2.set_image_to_dummy()
    im = roi2.image
    assert roi2.pos_to_idx(0, "z") == im.pos_to_idx(0, "z")
    assert roi2.idx_to_pos(0, "z") == im.idx_to_pos(0, "z")
    assert roi2.pos_to_slice(0, "z") == im.pos_to_slice(0, "z")
    assert roi2.slice_to_pos(0, "z") == im.slice_to_pos(0, "z")
    assert roi2.slice_to_idx(0, "z") == im.slice_to_idx(0, "z")
    assert roi2.idx_to_slice(0, "z") == im.idx_to_slice(0, "z")

# Create structure set for transform tests
sim0 = SyntheticImage((100, 100, 40))
sim0.add_sphere(radius=5, centre=(20, 70, 30), name="sphere", intensity=10)
sim0.add_cube(side_length=5, centre=(30, 20, 10), name="cube", intensity=1)
ss0 = sim0.get_structure_set()

# Set limits for random-number generation for transform tests
dx_min, dx_max = (-15, 65)
dy_min, dy_max = (-15, 25)
xyc_min, xyc_max = (40, 60)
zc_min, zc_max = (15, 25)
dz_min, dz_max = (-5, 5)
theta_min, theta_max = (0, 360)

# Set level of agreement (mm) for post-transform centroid positions
precision = 0.5

def test_structure_set_translation():
    '''Test structure-set translation.'''
    n_test = 10
    random.seed(1)

    for i in range(n_test):
        # Randomly translate structure sets.
        dx = random.uniform(dx_min, dx_max)
        dy = random.uniform(dy_min, dy_max)
        dz = random.uniform(dz_min, dz_max)
        translation = [dx, dy, dz]
        # Translation using masks.
        ss1 = ss0.clone()
        ss1.transform(translation=translation)
        # Translation using contours.
        ss2 = ss0.clone()
        ss2.transform(translation=translation, force_contours=True)

        # Calculate expected centroid, and compare with transform results.
        for roi in ss0.get_rois():
            translated_centroid = [roi.get_centroid()[i] + translation[i]
                for i in range(3)]

            roi1 = ss1.get_roi(roi.name)
            roi2 = ss2.get_roi(roi.name)
            for i in range(3):
                assert roi1.get_centroid()[i] == pytest.approx(
                        translated_centroid[i], abs=precision)
                assert roi2.get_centroid()[i] == pytest.approx(
                        translated_centroid[i], abs=precision)

def test_structure_set_rotation():
    '''Test structure-set rotation.'''
    n_test = 10
    random.seed(1)

    for i in range(n_test):
        # Randomly rotate structure sets.
        xc = random.uniform(xyc_min, xyc_max)
        yc = random.uniform(xyc_min, xyc_max)
        zc = random.uniform(zc_min, zc_max)
        centre = (xc, yc, zc)
        angle = random.uniform(theta_min, theta_max)
        rad_angle = math.radians(angle)
        rotation = [0, 0, angle]
        # Rotation using masks.
        ss1 = StructureSet(ss0)
        ss1.transform(centre=centre, rotation=rotation)
        # Rotation using contours.
        ss2 = StructureSet(ss0)
        ss2.transform(centre=centre, rotation=rotation, force_contours=True)

        # Calculate expected centroid, and compare with transform results.
        for roi in ss0.get_rois():
            x0, y0, z0 = roi.get_centroid()
            dx = x0 - xc
            dy = y0 - yc
            xr = dx * math.cos(rad_angle) - dy * math.sin(rad_angle) + xc
            yr = dx * math.sin(rad_angle) + dy * math.cos(rad_angle) + yc
            zr = z0
            rotated_centroid = [xr, yr, zr]

            roi1 = ss1.get_roi(roi.name)
            roi2 = ss2.get_roi(roi.name)
            for i in range(3):
                assert roi1.get_centroid()[i] == pytest.approx(
                        rotated_centroid[i], abs=precision)
                assert roi2.get_centroid()[i] == pytest.approx(
                        rotated_centroid[i], abs=precision)

def test_get_extent():
    sim = SyntheticImage((10, 10, 10), origin=(0.5, 0.5, 0.5))
    sim.add_cuboid((4, 2, 6), name="cube")
    roi = sim.get_roi("cube")
    ext = roi.get_extents()
    assert ext[0] == [3, 7]
    assert ext[1] == [4, 6]
    assert ext[2] == [2, 8]
    ext = roi.get_extents(buffer=2)
    assert ext[0] == [1, 9]
    assert ext[1] == [2, 8]
    assert ext[2] == [0, 10]

# Make structure set with nearby structs for consensus calculation
sim2 = SyntheticImage((50, 50, 10))
sim2.add_sphere(radius=5, centre=(25, 25, 5), name="sphere1")
sim2.add_sphere(radius=5, centre=(26, 26, 5), name="sphere2")
ss2 = sim.get_structure_set()

def test_get_staple():
    """Test STAPLE function (only runs if SimpleITK is installed)"""
    try:
        staple = ss2.get_staple()
        assert 0 < staple.get_volume() < sum([roi.get_volume() 
                                              for roi in ss2.get_rois()])
    except ModuleNotFoundError:
        pass

def test_get_majority_vote():
    mv = ss2.get_majority_vote()
    assert 0 < mv.get_volume() < sum([roi.get_volume() 
                                          for roi in ss2.get_rois()])

def test_get_overlap():
    overlap = ss2.get_overlap()
    overlap_mask = ss2.get_rois()[0].get_mask()
    for roi in ss2.get_rois()[1:]:
        overlap_mask *= roi.get_mask()
    assert overlap_mask.sum() == overlap.get_volume(units="voxels")

def test_get_sum():
    overlap = ss2.get_overlap()
    roi_sum = ss2.get_sum()
    all_vol = sum([roi.get_volume() for roi in ss2.get_rois()])
    assert roi_sum.get_volume() == all_vol  - overlap.get_volume()

def test_multi_label():
    """Test reading a StructureSet from a single multi-label array."""

    array = np.zeros(structure_set.get_rois()[0].get_mask().shape)
    for i, roi in enumerate(structure_set):
        array[roi.get_mask()] = i + 1
    ss_multi = StructureSet(array, multi_label=True, 
                            names=structure_set.get_roi_names())
    assert len(ss_multi.get_rois()) == len(structure_set.get_rois())
    assert ss_multi.get_roi_names() == structure_set.get_roi_names()

#  def test_recolor():
    #  """Test recoloring ROIs with a list as they are loaded in."""

    #  colors = ["lime", "blue", "orange", "purple", "red"]
    #  ss = StructureSet("tmp/nii_structs", colors=colors)
    #  for i, roi in enumerate(ss.get_rois()):
        #  assert roi.color == matplotlib.colors.to_rbga(colors[i])

#  def test_recolor_from_dict():
    #  """Test recolor based on ROI names."""

    #  sim = SyntheticImage((50, 50, 10))
    #  sim.add_sphere(5, name="sphere1")
    #  sim.add_sphere(10, name="sphere2")
    #  colors = {"*1": "blue", "*2": "red"}
    #  ss = sim.get_structure_set()
    #  ss.recolor_rois(colors)
    #  assert ss.get_roi("sphere1").color == matplotlib.colors.to_rgba("blue")
    #  assert ss.get_roi("sphere2").color == matplotlib.colors.to_rgba("red")

def test_emptiness():

    # test emptiness of ROI created from mask
    empty_array = np.zeros((10, 10, 2))
    empty_roi = ROI(empty_array)
    assert empty_roi.empty

    # test non-empty array
    non_empty_array = empty_array.copy()
    non_empty_array[4:6, 4:6, :] = 1
    non_empty_roi = ROI(non_empty_array, mask_threshold=0.9)
    assert not non_empty_roi.empty

    # test empty ROI from contours
    empty_roi_contours = ROI(empty_roi.get_contours())
    assert empty_roi_contours.empty
    
    # test non-empty ROI from contours
    non_empty_roi_contours = ROI(non_empty_roi.get_contours())
    assert not non_empty_roi_contours.empty

def test_get_rois_ignore_empty():
    """Test ROI getting from StructureSet with and without empty ROIs"""

    empty_array = np.zeros((10, 10, 2))
    empty_roi = ROI(empty_array)
    non_empty_array = empty_array.copy()
    non_empty_array[4:6, 4:6, :] = 1
    non_empty_roi = ROI(non_empty_array, mask_threshold=0.9)
    ss = StructureSet([empty_roi, non_empty_roi])

    rois = ss.get_rois()
    assert len(rois) == 2

    rois = ss.get_rois(ignore_empty=True)
    assert len(rois) == 1

def test_dice():
    """Test calculation of Dice scores"""

    # Compare sphere with sphere increased in volume by different scale factors.
    for scale in [1., 1.1, 1.2, 1.3, 1.4, 1.5]:
        sphere1 = ROI(structure_set.get_roi("sphere"))
        sphere2 = ROI(structure_set.get_roi("sphere"))
        sphere2.transform(scale=scale, centre=sphere2.get_centroid())

        # Check that volume change is as expected.
        assert sphere1.get_volume() * scale**3 == pytest.approx(
                sphere2.get_volume(), rel=0.01)

        # Calculate Dice score based on roi volumes.
        dice_expected = 2 / (1 + sphere2.get_volume() / sphere1.get_volume())


        # Compare Dice scores with expectations, for two methods used.
        for method in ['contour', 'mask']:
            assert sphere1.get_dice(sphere2, method=method) == pytest.approx(
                    dice_expected, abs=0.001)
            assert sphere2.get_dice(sphere1, method=method) == pytest.approx(
                    dice_expected, abs=0.001)
            assert sphere2.get_dice(sphere1, method=method) == sphere1.get_dice(
                    sphere2, method=method)

def test_point_interpolation_single_contours():
    random.seed(1)

    sphere = structure_set.get_roi("sphere")

    # Interpolate points for individual contours,
    # then check that number of points is within 1 of the requested value.
    for z, contours in sphere.get_contours().items():
        for contour in contours:
            n_point = random.randint(10, 100)
            contour2 = interpolate_points_single_contour(contour, n_point)
            assert abs(n_point - len(contour2)) <= 1

def test_point_interpolation_roi():
    sphere1 = structure_set.get_roi("sphere")
    n_points = [4, 100]

    for n_point in n_points:
        sphere2 = sphere1.interpolate_points(n_point=n_point)

        if 4 == n_point:
            # Expect Dice score close to that for a circle with radius r
            # circumscribing a square with sides sqrt(2) * r
            dice_expected = 4 / (math.pi + 2)
            assert sphere1.get_dice(sphere2, method='contour') == pytest.approx(
                    dice_expected, abs=0.025)
        if 100 == n_point:
            # Expect dice score close to 1
            dice_expected = 1
            assert sphere1.get_dice(sphere2, method='contour') == pytest.approx(
                    dice_expected, abs=0.005)

        # Check that number of points for each contouri
        # is within 1 of the requested value.
        for z, contours2 in sphere2.get_contours().items():
            for contour2 in contours2:
                assert abs(n_point - len(contour2)) <=1

def test_roi_split():
    '''Test splitting of composite ROI into components.'''

    # Define non-composite ROIs.
    sim0 = SyntheticImage((100, 100, 100))
    sim0.add_cube(side_length=40, name="cube", centre=(30, 30, 50), intensity=1)
    sim0.add_sphere(radius=20, name="sphere", centre=(70, 70, 50), intensity=10)
    cube = sim0.get_roi('cube')
    sphere = sim0.get_roi('sphere')

    # Define composite ROI.
    sim = SyntheticImage((100, 100, 100))
    sim.add_cube(side_length=40, name="cube", centre=(30, 30, 50),
            intensity=1, group='my_group')
    sim.add_sphere(radius=20, name="sphere", centre=(70, 70, 50),
            intensity=10, group='my_group')
    my_group = sim.get_roi('my_group')
    ss = sim.get_structure_set()

    # Test splitting and name assignment.
    names = ['shape_1', 'shape_2']
    ss2 = my_group.split(names=names)
    assert len(ss2.get_rois()) == 2
    assert ss2.get_extent() == ss.get_extent()
    for i in range(2):
        assert ss2.rois[i].name == names[i]
    volume = 0
    for roi in ss2.get_rois():
        volume += roi.get_volume()
    assert volume == my_group.get_volume()

    # Test splitting with change in mask voxel size.
    ss3 = my_group.split(voxel_size=(0.5, 0.5))
    assert len(ss3.get_rois()) == 2
    assert ss3.get_extent() == ss.get_extent()
    volume = 0
    for roi in ss3.get_rois():
        volume += roi.get_volume()
    assert volume == my_group.get_volume()

    # Test splitting with ordering.
    for order in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        ss4 = my_group.split(order=order)
        rois4 = ss4.get_rois()
        assert len(rois4) == 2
        axis = 'xyz'.find(order[0])
        rois0 = (cube, sphere) if ('+' in order) else (sphere, cube)
        for i in range(2):
            assert (rois4[i].get_centroid()[axis]
                    == rois0[i].get_centroid()[axis])

def test_roi_split_in_two():
    '''Test splitting of composite ROI into two components.'''

    # Define non-composite ROIs.
    sim0 = SyntheticImage((100, 100, 100))
    sim0.add_cube(side_length=40, name="cube", centre=(30, 30, 50), intensity=1)
    sim0.add_sphere(radius=20, name="sphere", centre=(70, 70, 50), intensity=10)
    cube0 = ROI(sim0.get_roi('cube').get_contours(), image=sim0)
    sphere0 = ROI(sim0.get_roi('sphere').get_contours(), image=sim0)
    
    # Define composite ROI.
    sim = SyntheticImage((100, 100, 100))
    sim.add_cube(side_length=40, name="cube", centre=(30, 30, 50),
            intensity=1, group='my_group')
    sim.add_sphere(radius=20, name="sphere", centre=(70, 70, 50),
            intensity=10, group='my_group')
    my_group = ROI(sim.get_roi('my_group').get_contours(), image=sim)

    # Check that ROIs from splitting match originals
    for axis in ['x', 'y']:
        ss = my_group.split_in_two(axis=axis, v0=50, names=['cube', 'sphere'])
        assert len(ss.rois) == 2
        cube, sphere = ss.get_rois()

        assert np.all(cube.get_centroid() == cube0.get_centroid())
        assert np.all(sphere.get_centroid() == sphere0.get_centroid())
        assert (cube.get_volume() == cube0.get_volume())
        assert (sphere.get_volume() == sphere0.get_volume())

def test_combine_rois():
    '''Test combining of ROIs.'''

    # Define non-composite ROIs.
    sim = SyntheticImage((100, 100, 100))
    sim.add_cube(side_length=40, name="cube", centre=(30, 30, 50), intensity=1)
    sim.add_sphere(radius=20, name="sphere", centre=(70, 70, 50), intensity=10)
    cube = sim.get_roi('cube')
    sphere = sim.get_roi('sphere')
    ss = sim.get_structure_set()
    
    # Define composite ROI.
    cube_and_sphere = ss.combine_rois()

    assert (cube_and_sphere.get_volume() ==
            cube.get_volume() + sphere.get_volume())

    assert np.all(cube_and_sphere.get_mask() ==
            cube.get_mask() + sphere.get_mask())

def test_mask_image():
    '''Test retrieval of image objects repreenting ROI masks.'''

    # Define ROIs and structure_set.
    sim = SyntheticImage((100, 100, 100))
    sim.add_cube(side_length=40, name="cube", centre=(30, 30, 50), intensity=1)
    sim.add_sphere(radius=20, name="sphere", centre=(70, 70, 50), intensity=10)
    cube = sim.get_roi('cube')
    sphere = sim.get_roi('sphere')
    ss = sim.get_structure_set()
    
    # Check mask images for individual ROIs.
    ss_voxels = 0
    for roi in ss.get_rois():
        assert roi.get_mask().shape == roi.get_mask_image().get_data().shape
        roi_voxels = (roi.get_mask() > 0.5).sum()
        assert roi_voxels > 20 * 20 * 20
        ss_voxels += roi_voxels
        assert roi_voxels == (roi.get_mask_image().get_data() > 0.5).sum()

    # Check mask image for structure set.
    assert ss_voxels == (ss.get_mask_image().get_data() > 0.5).sum()

def test_pathlib_path():
    # Test passing of pathlib.Path.
    nii_dir = pathlib.Path("tmp/nii_structs")
    structs_from_nii = StructureSet(nii_dir)
    assert structs_from_nii.path == fullpath(nii_dir)

    for roi in structs_from_nii:
        roi_new = ROI(pathlib.Path(roi.path))
        assert roi_new.path == roi.path
