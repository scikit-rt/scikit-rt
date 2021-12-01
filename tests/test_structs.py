"""Tests for the ROI and StructureSet classes."""

import fnmatch
import os
import shutil
import pandas as pd
import pytest
import numpy as np

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from skrt.simulation import SyntheticImage
from skrt.structures import contour_to_polygon, StructureSet, ROI


# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# Create synthetic structure set
sim = SyntheticImage((100, 100, 40))
sim.add_cube(side_length=40, name="cube", intensity=1)
sim.add_sphere(radius=20, name="sphere", intensity=10)
structure_set = sim.get_structure_set()
roi = structure_set.get_roi("cube")


def test_structure_set_from_sythetic_image():
    assert isinstance(structure_set, StructureSet)

def test_write_nifti():
    nii_dir = "tmp/nii_structs"
    if os.path.exists(nii_dir):
        shutil.rmtree(nii_dir)
    structure_set.write(outdir="tmp/nii_structs")
    assert len(os.listdir(nii_dir)) == 2

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
    assert isinstance(roi, ROI)
    assert roi.name == "cube"

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
    assert structure_set.get_rois()[0].name != structure_set.get_rois()[0].original_name
    assert set(structure_set.get_roi_names()) == set(new_names.keys())

    # Check that original names are kept
    n_match = 0
    for roi in structure_set:
        if roi.name in new_names:
            match = False
            for old_name in new_names[roi.name]:
                if fnmatch.fnmatch(roi.original_name.lower(), old_name.lower()):
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
    structure_set2 = structure_set.clone()
    sphere2 = structure_set2.get_roi("sphere")
    sphere1.set_color("red")
    sphere2.set_color("blue")
    assert sphere1.color != sphere2.color

def test_filtered_copy_no_copy_data():
    """Test copying but using same data for ROIs."""

    # First check that data is copied correctly
    structure_set2 = structure_set.filtered_copy(copy_roi_data=True)
    assert structure_set2.get_rois()[0].get_mask() \
            is not structure_set.get_rois()[0].get_mask()

    # Check that data is passed by reference
    structure_set2 = structure_set.filtered_copy(copy_roi_data=False)
    assert structure_set2.get_rois()[0].get_mask() \
            is structure_set.get_rois()[0].get_mask()

def test_read_nii():
    nii_dir = "tmp/nii_structs"
    structs_from_nii = StructureSet(nii_dir)
    assert len(structs_from_nii.get_rois()) == 2
    assert set(structure_set.get_roi_names()) \
            == set(structs_from_nii.get_roi_names())

def test_get_geometry():
    geom = structure_set.get_geometry()
    assert isinstance(geom, pd.DataFrame)
    assert geom.shape[0] == 2

def test_get_comparison_pairs():
    pairs = structure_set.get_comparison_pairs()
    assert len(pairs) == 2
    assert len(pairs[0]) == 2

def test_get_comparison_pairs_with_other():
    structure_set2 = StructureSet("tmp/nii_structs")
    pairs = structure_set.get_comparison_pairs(structure_set2)
    assert len(pairs) == 2
    assert pairs[0][0].name == pairs[0][1].name
    assert pairs[1][0].name == pairs[1][1].name

def test_get_comparison():
    comp = structure_set.get_comparison()
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape[0] == 2

def test_plot_comparisons():
    plot_dir = "tmp/struct_plots"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    structure_set.plot_comparisons(outdir=plot_dir, show=False)
    assert len(os.listdir(plot_dir)) == 2
    
def test_write_dicom():
    pass

def test_dicom_dataset():
    pass

def test_roi_from_image_threshold():
    roi = ROI(sim, mask_level=5)  
    assert roi.get_area() == sim.get_roi("sphere").get_area()

def test_roi_no_image_with_geom():
    """Check that an ROI object can be created from contours only plus 
    geometric info."""

    new_roi = ROI(roi.get_contours(),
                  origin=roi.get_origin(),
                  voxel_size=roi.get_voxel_size(),
                  shape=roi.get_mask().shape
                 )
    assert np.all(new_roi.get_mask() == roi.get_mask())

def test_roi_from_polygons():
    """Check that an ROI object can be created from shapely polygons."""

    new_roi = ROI(roi.get_polygons(),
                  origin=roi.get_origin(),
                  voxel_size=roi.get_voxel_size(),
                  shape=roi.get_mask().shape
                 )
    assert np.all(new_roi.get_mask() == roi.get_mask())

    df = StructureSet(new_roi).get_comparison(StructureSet(roi))
    # Dice score may not be exactly 1 because of rounding errors
    assert df['Dice score'][0] == pytest.approx(1.0, abs=1.e-3)

def test_null_roi():
    roi = ROI()
    assert(type(roi).__name__ == "ROI")
    assert(roi.affine is None)
    assert(roi.contours == {})
    assert(roi.custom_color is False)
    assert(roi.image is None)
    assert(roi.input_contours is None)
    assert(roi.kwargs == {})
    assert(roi.loaded is False)
    assert(roi.loaded_contours is False)
    assert(roi.loaded_mask is False)
    assert(roi.origin is None)
    assert(roi.original_name is None)
    assert(roi.roi_source_type is None)
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
