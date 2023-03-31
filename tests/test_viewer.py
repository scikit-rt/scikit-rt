"""Test BetterViewer."""

import glob
import numpy as np
import pytest
import shutil
import os
from skrt.better_viewer import BetterViewer, options
#  from skrt.viewer import OrthogViewer
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from skrt.image import Image, ImageComparison
from skrt.simulation import SyntheticImage

# Create fake data
data = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)

def create_test_image():
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

im = create_test_image()

# Make temporary test dir
if not os.path.exists('tmp'):
    os.mkdir('tmp')

# Create a test nifti file
nii_file = 'tmp/tmp.nii'
im.write(nii_file)
im_nii = Image(nii_file)
data2 = (np.random.rand(30, 20, 20) * 1000).astype(np.uint16)
im2 = Image(data2)


# Decorator to close matplotlib figures (prevents too many figures from existing)
def close_after(func):
    def do_then_close():
        func()
        plt.close("all")
    return do_then_close


@close_after
def test_single_image():
    qv = BetterViewer(nii_file, show=False)
    assert len(qv.viewers) == 1


@close_after
def test_single_image_object():
    qv = BetterViewer(im, show=False)
    assert len(qv.viewers) == 1


@close_after
def test_viewer_from_image():
    im.view(show=False)

@close_after
def test_view_from_image_with_multi():
    qv = im.view(images=[im, im], show=False)
    assert len(qv.viewers) == 3
    qv = im.view(images=im, show=False)
    assert len(qv.viewers) == 2

@close_after
def test_multiple_image():
    qv = BetterViewer([im, im], show=False)
    assert len(qv.viewers) == 2
    assert not len(qv.slider_boxes)


@close_after
def test_different_size_images():
    qv = BetterViewer([im, im2], show=False)
    assert len(qv.viewers) == 2
    assert len(qv.slider_boxes) == 2


@close_after
def test_init_idx():
    init_sl = 10
    qv = BetterViewer(im, init_sl=init_sl, scale_in_mm=False,
                     show=False)
    assert qv.viewers[0].ui_slice.value == init_sl


@close_after
def test_custom_intensity():
    v = (-500, 100)
    qv = BetterViewer(im, intensity=v, show=False)
    assert qv.viewers[0].ui_intensity.value == v


@close_after
def test_figsize():
    figsize = 10
    qv = BetterViewer(im, figsize=figsize, show=False)
    assert int(qv.fig.get_size_inches()[1]) == int(figsize)


@close_after
def test_init_views():
    for view in ["x-y", "y-z", "x-z"]:
        bv = BetterViewer(im, init_view=view, show=False)
        assert bv.view == view


@close_after
def test_suptitle():
    title = "test"
    qv = BetterViewer(im, suptitle=title, show=False)
    assert qv.suptitle == title


#  @close_after
#  def test_translation():
    #  BetterViewer(['data/MI_Translation/ct_relapse.nii',
                 #  'data/MI_Translation/result.0.nii'],
                #  translation=True, show=False)


@close_after
def test_cb():
    qv = BetterViewer([im, im], comparison="chequerboard", show=False)
    assert len(qv.comparison) == 1
    assert qv.ui_cb in qv.all_ui
    assert qv.ui_overlay not in qv.all_ui


@close_after
def test_overlay():
    qv = BetterViewer([im, im], comparison="overlay", show=False)
    assert len(qv.comparison) == 1
    assert qv.ui_overlay in qv.all_ui
    assert qv.ui_cb not in qv.all_ui


@close_after
def test_difference():
    qv = BetterViewer([im, im], comparison="difference", show=False)
    assert len(qv.comparison) == 1
    assert qv.ui_invert in qv.all_ui
    assert qv.ui_cb not in qv.all_ui
    assert qv.ui_overlay not in qv.all_ui


@close_after
def test_multicomp():
    qv = BetterViewer([im, im], comparison=True, show=False)
    assert len(qv.comparison) == 1
    assert qv.ui_multicomp in qv.all_ui
    assert qv.ui_invert in qv.all_ui
    assert qv.ui_cb in qv.all_ui
    assert qv.ui_overlay in qv.all_ui


@close_after
def test_many_comparisons():
    qv = BetterViewer([im, im], 
                      comparison=["chequerboard", "overlay", "difference"], 
                      show=False)
    assert len(qv.comparison) == 3


@close_after
def test_comparison_only():
    qv = BetterViewer([im, im], comparison=True, comparison_only=True, 
                      show=False)
    assert len(qv.comparison) == 1
    assert len(qv.fig.get_axes()) == 1


@close_after
def test_view_comparison():
   comp = ImageComparison(im, im2)
   comp.view(show=False)


@close_after
def test_titles():
    title = ["test1", "test2"]
    qv = BetterViewer([im, im2], title=title, show=False)
    assert qv.viewers[0].image.ax.title.get_text() == title[0]
    assert qv.viewers[1].image.ax.title.get_text() == title[1]

def make_sim():
    """Make synthetic image containing two ROIs."""

    sim = SyntheticImage((100, 100, 10))
    sim.add_sphere(20, name="sphere")
    sim.add_cube(10, name="cube")
    return sim

def make_structure_set():
    """Make a structure set containing two ROIs."""

    sim = make_sim()
    return sim.get_structure_set()

@close_after
def test_unique_naming():
    """Check that multiple ROIs with the same name are assigned unique names."""

    ss1 = make_structure_set()
    ss2 = make_structure_set()
    assert ss1.get_roi_names() == ss2.get_roi_names()
    bv = ss1.image.view(rois=[ss1, ss2], show=False)
    roi_names = bv.viewers[0].roi_names
    assert len(set(roi_names)) == 4
    for name in ss2.get_roi_names():
        assert f"{name} 1" in roi_names
        assert f"{name} 2" in roi_names

@close_after
def test_unique_naming_from_structure_set():
    """Check that multiple ROIs with the same name are assigned unique names
    based on StructureSet names."""

    ss1 = make_structure_set()
    ss1.name = "ss1"
    ss2 = make_structure_set()
    ss2.name = "ss2"
    assert ss1.get_roi_names() == ss2.get_roi_names()
    bv = ss1.image.view(rois=[ss1, ss2], show=False)
    roi_names = bv.viewers[0].roi_names
    assert len(set(roi_names)) == 4
    for ss in [ss1, ss2]:
        for name in ss.get_roi_names():
            assert f"{name} ({ss.name})" in roi_names

@close_after
def test_roi_comparison():
    """Test comparison two StructureSets with some overlapping ROI names."""

    sim1 = make_sim()
    sim2 = make_sim()
    sim1.add_sphere(10, name="1")
    sim2.add_sphere(10, name="2")
    ss1 = sim1.get_structure_set()
    ss2 = sim2.get_structure_set()

    bv = ss1.image.view(rois=[ss1, ss2], compare_rois=True, show=False)
    viewer = bv.viewers[0]
    assert viewer.ui_roi_comp_table.value
    assert len(viewer.comparison_pairs) == 2
    assert len(viewer.rois) == 4   # Non-matching ROIs should be ignored

    # Test comparison without ignoring non-matching ROI names
    bv = ss1.image.view(rois=[ss1, ss2], compare_rois=True, show=False,
                       show_compared_rois_only=False)
    viewer = bv.viewers[0]
    assert len(viewer.comparison_pairs) == 2
    assert len(viewer.rois) == 6
    
@close_after
def test_roi_comparison_single_structure_set():
    """Test comparison within a single StructureSet."""

    sim = make_sim()
    sim.add_sphere(20, name="other")
    ss = sim.get_structure_set()

    bv = ss.image.view(rois=ss, compare_rois=True, show=False)
    viewer = bv.viewers[0]
    assert viewer.ui_roi_comp_table.value
    assert len(viewer.comparison_pairs) == 3
    assert len(viewer.rois) == 3

def test_set_viewer_options():
    """Test that options.set_viewer_options() returns a non-empty dictionary."""
    viewer_options = options.set_viewer_options()
    assert viewer_options and isinstance(viewer_options, dict)
    key = list(viewer_options.keys())[0]
    viewer_options = options.set_viewer_options(to_exclude=key)
    assert key not in viewer_options
