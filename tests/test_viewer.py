"""Test BetterViewer."""

import glob
import numpy as np
import pytest
import shutil
import os
from skrt.better_viewer import BetterViewer
#  from skrt.viewer import OrthogViewer
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from skrt.image import Image, ImageComparison

# Create fake data
data = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)

def create_test_image():
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

im = create_test_image()

# Make temporary test dir
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Create a test nifti file
nii_file = 'tmp/tmp.nii'
im.write(nii_file)
im_nii = Image(nii_file)
data2 = (np.random.rand(30, 20, 20) * 1000).astype(np.uint16)
im2 = Image(data2)


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
def test_custom_hu():
    v = (-500, 100)
    qv = BetterViewer(im, hu=v, show=False)
    assert qv.viewers[0].ui_hu.value == v


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
    assert qv.viewers[0].im.ax.title.get_text() == title[0]
    assert qv.viewers[1].im.ax.title.get_text() == title[1]


#  @close_after
#  def test_mask():
    #  BetterViewer("data/ct.nii", mask=("data/structs/RTSTRUCT_CT_20140715_113632"
                                     #  "_002_alterio_pcs.nii.gz"), show=False)


#  @close_after
#  def test_dose():
    #  opacity = 0.3
    #  cmap = "gray"
    #  qv = BetterViewer("data/MI_BSpline30/result.0.nii",
                     #  dose="data/MI_BSpline30/spatialJacobian.nii",
                     #  dose_kwargs={"cmap": cmap},
                     #  dose_opacity=opacity, show=False)
    #  assert qv.viewers[0].ui_dose.value == opacity


#  @close_after
#  def test_share_slider():
    #  qv = BetterViewer(['data/MI_Translation/ct_relapse.nii',
                      #  'data/MI_Translation/result.0.nii'], share_slider=False,
                      #  show=False)
    #  assert len(qv.viewers) == 2
    #  assert len(qv.slider_boxes) == 2


#  @close_after
#  @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
#  def test_structs():

    #  # Test directory
    #  qv = BetterViewer("data/ct.nii", structs="data/structs", show=False)
    #  assert len(qv.viewers[0].im.structs) == len(os.listdir("data/structs")) - 1

    #  # Test list of files
    #  qv = BetterViewer(
        #  "data/ct.nii",
        #  structs=[
            #  "data/structs/RTSTRUCT_CT_20140715_113632_002_mpc.nii.gz",
            #  "data/structs/RTSTRUCT_CT_20140715_113632_002_right_smg.nii.gz"
        #  ], show=False)
    #  assert len(qv.viewers[0].im.structs) == 2

    #  # Test wildcard directory
    #  qv = BetterViewer("data/ct.nii", structs="data/str*", show=False)
    #  assert len(qv.viewers[0].im.structs) == len(os.listdir("data/structs")) - 1

    #  # Test wildcard files
    #  qv = BetterViewer("data/ct.nii", show=False,
                     #  structs=["data/structs/*parotid*", "data/structs/*mpc*"])
    #  assert len(qv.viewers[0].im.structs) == 3


#  @close_after
#  @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
#  def test_struct_colours():

    #  colour = 'cyan'

    #  # Test with wildcard filename
    #  qv = BetterViewer("data/ct.nii", structs="data/structs/*parotid*",
                     #  struct_colours={"*parotid*": colour},
                     #  show=False)
    #  assert len(qv.viewers[0].im.structs) == 2
    #  assert qv.viewers[0].im.structs[0].color == to_rgba(colour)

    #  # Test with structure name
    #  qv = BetterViewer("data/ct.nii", structs="data/structs/*right_parotid*",
                     #  struct_colours={"right parotid": colour},
                     #  show=False)
    #  assert qv.viewers[0].im.structs[0].color == to_rgba(colour)

    #  # Test with wildcard structure name
    #  qv = BetterViewer("data/ct.nii", structs="data/structs/*right_parotid*",
                     #  struct_colours={"*parotid": colour}, show=False)
    #  assert qv.viewers[0].im.structs[0].color == to_rgba(colour)


#  @close_after
#  @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
#  def test_struct_mask():
    #  opacity = 0.6
    #  qv = BetterViewer("data/ct.nii", structs="data/structs/*mpc*",
                     #  struct_plot_type="mask", struct_opacity=opacity,
                     #  show=False)
    #  assert qv.viewers[0].ui_struct_opacity.value == opacity


#  @close_after
#  def test_zoom():
    #  BetterViewer("data/ct.nii", zoom=2, show=False)


#  @close_after
#  def test_downsample():
    #  BetterViewer("data/ct.ni", downsample=(5, 4, 2), show=False)


#  @close_after
#  def test_jacobian():
    #  opacity = 0.2
    #  qv = BetterViewer("data/MI_BSpline30/result.0.nii",
                #  jacobian="data/MI_BSpline30/spatialJacobian.nii",
                #  jacobian_opacity=opacity, show=False)
    #  assert qv.viewers[0].ui_jac_opacity.value == opacity


#  @close_after
#  def test_df_grid():
    #  qv = BetterViewer("data/MI_BSpline30/result.0.nii",
                #  df="data/MI_BSpline30/deformationField.nii", show=False)
    #  assert qv.viewers[0].ui_df.value == "grid"


#  @close_after
#  def test_df_quiver():
    #  qv = BetterViewer("data/MI_BSpline30/result.0.nii",
                #  df="data/MI_BSpline30/deformationField.nii",
                #  df_plot_type="quiver", show=False)
    #  assert qv.viewers[0].ui_df.value == "quiver"


#  @close_after
#  def test_save():
    #  output = "data/test_march2.pdf"
    #  #  if os.path.isfile(output):
        #  #  os.remove(output)
    #  BetterViewer("data/ct.nii", save_as=output, show=False)
    #  #  assert os.path.isfile(output)


#  @close_after
#  def test_orthog_view():
    #  qv = BetterViewer("data/ct.nii", orthog_view=True, show=False)
    #  assert isinstance(qv.viewers[0], OrthogViewer)


#  @close_after
#  def test_plots_per_row():
    #  qv = BetterViewer(["data/ct.nii", "data/ct.nii"], plots_per_row=1,
                     #  show=False)
