"""
Test legacy QuickViewer code.

Tests migrated from tests/test_quickviewer.py at:
    https://github.com/hlpullen/quickviewer
"""

import glob
import numpy as np
import pytest
import os
import shutil
from skrt.image import Image
from skrt.simulation import SyntheticImage
from skrt.viewer.viewer import QuickViewer
from skrt.viewer.viewer import OrthogViewer
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Make temporary test directory, and quickviewer subdirectory.
if not os.path.exists('tmp'):
    os.mkdir('tmp')
if not os.path.exists('tmp/qv'):
    os.mkdir('tmp/qv')
if os.path.exists('tmp/qv/structs'):
    shutil.rmtree('tmp/qv/structs')
os.mkdir('tmp/qv/structs')


# Create fake data
def create_test_image(data, voxel_size, origin):
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

def make_sim():
    """Make synthetic image containing ROIs."""

    sim = SyntheticImage((100, 100, 10))
    sim.add_sphere(20, name="sphere1")
    sim.add_sphere(25, name="sphere2")
    sim.add_cube(10, name="cube1")
    sim.add_cube(15, name="cube2")
    return sim

data1 = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)
im1 = create_test_image(data1, voxel_size, origin)
im1_path = 'tmp/qv/tmp1.nii'
im1.write(im1_path)

data2 = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
im2 = create_test_image(data2, voxel_size, origin)
im2_path = 'tmp/qv/tmp2.nii'
im2.write(im2_path)

data3 = (np.random.rand(30, 40, 30) * 1000).astype(np.uint16)
im3 = create_test_image(data3, voxel_size, origin)
im3_path = 'tmp/qv/tmp3.nii'
im3.write(im3_path)

sim = make_sim()
sim_path = 'tmp/qv/sim.nii'
sim.write(sim_path)
ss = sim.get_structure_set()
ss_path = 'tmp/qv/structs'
ss.write(outdir=ss_path, ext='.nii.gz')

def close_after(func):
    def do_then_close():
        func()
        plt.close("all")
    return do_then_close


@close_after
def test_single_image():
    qv = QuickViewer(im1_path, show=False)
    assert len(qv.viewer) == 1


@close_after
def test_invalid_image():
    fake_im = "fake.nii"
    if os.path.exists(fake_im):
        os.remove(fake_im)
    qv = QuickViewer([im1_path, fake_im], show=False)
    assert len(qv.viewer) == 1
    qv = QuickViewer(fake_im, show=False)
    assert not len(qv.viewer)


@close_after
def test_duplicate_image():
    qv = QuickViewer([im1_path, im1_path], show=False)
    assert len(qv.viewer) == 2
    assert not len(qv.slider_boxes)


@close_after
def test_multiple_images():
    qv = QuickViewer([im1_path, im2_path], show=False)
    assert len(qv.viewer) == 2
    assert not len(qv.slider_boxes)


@close_after
def test_different_size_images():
    qv = QuickViewer([im1_path, im3_path], show=False)
    assert len(qv.viewer) == 2
    assert len(qv.slider_boxes) == 2


@close_after
def test_init_idx():
    init_sl = 10
    qv = QuickViewer(im1_path, init_sl=init_sl, scale_in_mm=False,
                     show=False)
    assert qv.viewer[0].ui_slice.value == init_sl


@close_after
def test_custom_hu():
    v = (-500, 100)
    qv = QuickViewer(im1_path, hu=v, show=False)
    assert qv.viewer[0].ui_hu.value == v


@close_after
def test_figsize():
    QuickViewer(im1_path, figsize=10, show=False)


@close_after
def test_init_views():
    QuickViewer(im1_path, init_view="x-y", show=False)
    QuickViewer(im1_path, init_view="x-z", show=False)
    QuickViewer(im1_path, init_view="y-z", show=False)


@close_after
def test_suptitle():
    title = "test"
    qv = QuickViewer(im1_path, suptitle=title, show=False)
    assert qv.suptitle == title


@close_after
def test_translation():
    QuickViewer([im1_path, im2_path],
                translation=True, show=False)


@close_after
def test_cb():
    qv = QuickViewer([im1_path, im2_path],
                     show_cb=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_overlay():
    qv = QuickViewer([im1_path, im2_path],
                     show_overlay=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_diff():
    qv = QuickViewer([im1_path, im2_path],
                show_diff=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_comparison_only():
    qv = QuickViewer([im1_path, im2_path],
                     show_cb=True, show_diff=True, show_overlay=True,
                     comparison_only=True, show=False)
    assert len(qv.comparison) == 3


@close_after
def test_titles():
    title = ["test1", "test2"]
    qv = QuickViewer([im1_path, im1_path], title=title, show=False)
    assert qv.viewer[0].im.title == title[0]
    assert qv.viewer[1].im.title == title[1]


@close_after
def test_mask():
    QuickViewer(im1_path, mask=im2_path, show=False)


@close_after
def test_dose():
    opacity = 0.3
    cmap = "gray"
    qv = QuickViewer(im1_path,
                     dose=im2_path,
                     dose_kwargs={"cmap": cmap},
                     dose_opacity=opacity, show=False)
    assert qv.viewer[0].ui_dose.value == opacity


@close_after
def test_share_slider():
    qv = QuickViewer([im1_path, im2_path], share_slider=False,
                      show=False)
    assert len(qv.viewer) == 2
    assert len(qv.slider_boxes) == 2


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_structs():

    # Test directory
    qv = QuickViewer(sim_path, structs=ss_path, show=False)
    assert len(qv.viewer[0].im.structs) == len(os.listdir(ss_path))

    # Test list of files
    qv = QuickViewer(
        sim_path,
        structs=[
            f"{ss_path}/cube1.nii.gz",
            f"{ss_path}/sphere1.nii.gz"
        ], show=False)
    assert len(qv.viewer[0].im.structs) == 2

    # Test wildcard directory
    qv = QuickViewer(sim_path, structs="tmp/qv/str*", show=False)
    assert len(qv.viewer[0].im.structs) == len(os.listdir(ss_path))

    # Test wildcard files
    qv = QuickViewer(sim_path, show=False,
                     structs=[f"{ss_path}/*cube*", f"{ss_path}/*sphere*"])
    assert len(qv.viewer[0].im.structs) == 4


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_struct_colours():

    colour = 'cyan'

    # Test with wildcard filename
    qv = QuickViewer(sim_path, structs=f"{ss_path}/*sphere*",
                     struct_colours={"*sphere*": colour},
                     show=False)
    assert len(qv.viewer[0].im.structs) == 2
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)

    # Test with structure name
    qv = QuickViewer(sim_path, structs=f"{ss_path}/*cube1*",
                     struct_colours={"cube1": colour},
                     show=False)
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)

    # Test with wildcard structure name
    qv = QuickViewer(sim_path, structs=f"{ss_path}/cube*",
                     struct_colours={"cube*": colour}, show=False)
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_struct_mask():
    opacity = 0.6
    qv = QuickViewer(sim_path, structs=f"{ss_path}/*cube*",
                     struct_plot_type="mask", struct_opacity=opacity,
                     show=False)
    assert qv.viewer[0].ui_struct_opacity.value == opacity


@close_after
def test_zoom():
    QuickViewer(im1_path, zoom=2, show=False)


@close_after
def test_downsample():
    QuickViewer(im1_path, downsample=(5, 4, 2), show=False)


@close_after
def test_jacobian():
    opacity = 0.2
    qv = QuickViewer(im1_path,
                jacobian=im2_path,
                jacobian_opacity=opacity, show=False)
    assert qv.viewer[0].ui_jac_opacity.value == opacity


# Not currently able to generate synthetic data
# corresponding to deformation field.
"""
@close_after
def test_df_grid():
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                df="data/MI_BSpline30/deformationField.nii", show=False)
    assert qv.viewer[0].ui_df.value == "grid"


@close_after
def test_df_quiver():
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                df="data/MI_BSpline30/deformationField.nii",
                df_plot_type="quiver", show=False)
    assert qv.viewer[0].ui_df.value == "quiver"
"""


@close_after
def test_save():
    output = "tmp/qv/test_march2.pdf"
    if os.path.isfile(output):
        os.remove(output)
    QuickViewer(im1_path, save_as=output, show=False)
    assert os.path.isfile(output)


@close_after
def test_orthog_view():
    qv = QuickViewer(im1_path, orthog_view=True, show=False)
    assert isinstance(qv.viewer[0], OrthogViewer)


@close_after
def test_plots_per_row():
    qv = QuickViewer([im1_path, im1_path], plots_per_row=1,
                     show=False)
