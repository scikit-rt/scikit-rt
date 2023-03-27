"""Test functionality of the Dose class."""

from pathlib import Path
import pytest

import numpy as np

from skrt import Dose, Image, ROI
from skrt.core import fullpath
from skrt.simulation import SyntheticImage

# Make random arrays to use for an image and a dose map
im_array = np.random.rand(100, 150, 30)
dose_array = np.random.rand(100, 150, 30)

# Make temporary test dir
tmp_path = Path('tmp')
if not tmp_path.exists():
    tmp_path.mkdir()

def get_synthetic_data(dose_cube=1, dose_sphere=10):
    # Create synthetic structure set
    sim = SyntheticImage((100, 100, 40))
    sim.add_cube(side_length=30, centre=(30, 30, 20), name="cube",
            intensity=dose_cube)
    sim.add_sphere(radius=20, centre=(70, 70, 20), name="sphere",
            intensity=dose_sphere)
    structure_set = sim.get_structure_set()
    # Create dose object from the synthetic image
    dose = Dose(sim)
    return (dose, structure_set)

def test_create_dose_from_array():
    dose = Dose(dose_array)
    assert np.all(dose.get_data() == dose_array)

def test_set_dose_image():
    dose = Dose(dose_array)
    im = Image(im_array)
    dose.set_image(im)
    assert dose.image == im
    assert dose in im.doses

def test_plot_dose():
    dose = Dose(dose_array)
    dose.plot(include_image=False, show=False)
    dose.plot(include_image=True, show=False)

def test_plot_dose_from_image():
    im = Image(im_array)
    dose = Dose(dose_array)
    dose.set_image(im)
    im.plot(dose=0, show=False)
    im.plot(dose=dose, show=False)

def test_view():
    dose = Dose(dose_array)
    bv = dose.view(include_image=False, show=False)
    assert bv.viewers[0].image == dose

def test_view_with_image():
    im = Image(im_array)
    dose = Dose(dose_array)
    dose.set_image(im)
    opacity = 0.8
    bv = dose.view(include_image=True, show=False, dose_opacity=opacity)
    assert np.all(bv.viewers[0].image.get_data() == dose.get_data())
    assert np.all(bv.viewers[0].image.image.get_data() == im.get_data())
    assert bv.viewers[0].ui_dose.value == opacity

def test_view_from_image():
    im = Image(im_array)
    dose = Dose(dose_array)
    bv = im.view(dose=dose, show=False)
    assert bv.viewers[0].dose == dose

def test_array_to_dcm():
    dose = Dose(dose_array)
    dcm_dir = str(tmp_path / 'dose_dcm')
    dose.write(dcm_dir, modality='RTDOSE')
    dose_dcm = Dose(dcm_dir)
    assert dose.data.shape == dose_dcm.data.shape
    assert np.all(dose.affine == dose_dcm.affine)
    assert np.abs(dose.get_data() - dose_dcm.get_data()).max() < 0.005

def test_dcm_to_dcm():
    dose = Dose(dose_array)
    dcm1_dir = str(tmp_path / 'dose_dcm1')
    dose.write(dcm1_dir, modality='RTDOSE')
    dose_dcm1 = Dose(dcm1_dir)
    dcm2_dir = str(tmp_path / 'dose_dcm2')
    series_description = 'Dose study'
    header_extras = {'SeriesDescription' : series_description}
    dose_dcm1.write(dcm2_dir, modality='RTDOSE', header_extras=header_extras)
    dose_dcm2 = Dose(dcm2_dir)
    assert dose_dcm1.data.shape == dose_dcm2.data.shape
    assert np.all(dose_dcm1.affine == dose_dcm2.affine)
    assert np.abs(dose_dcm1.get_data() - dose_dcm2.get_data()).max() < 0.005
    assert dose_dcm2.get_dicom_dataset().SeriesDescription == series_description

def test_plot_dvh():
    dose, structure_set = get_synthetic_data()
    cube = structure_set.get_roi('cube')
    dose_max = dose.get_max_dose_in_rois(structure_set.get_rois())
    pdf = Path('tmp/dvh.pdf')
    pdf.unlink(missing_ok=True)

    # Check that pdf-file doesn't initially exist, but is created.
    assert not pdf.exists()
    ax = dose.plot_dvh(rois=[cube, structure_set], fname=str(pdf))
    assert pdf.exists()

    # Check data bounds
    x1, x2 = ax.get_xbound()
    y1, y2 = ax.get_ylim()
    dx, dy = ax.margins()
    x_range = (x2 - x1) / (1 + 2 * dx)
    ddx = x_range * dx
    assert x1 + ddx == 0
    assert x2 - ddx == dose_max
    assert y1 == 0
    assert y2 - dy == 1

    # Check that expected rois are included in the dose-volume histogram.
    roi_names = structure_set.get_roi_names()
    labels = [text.get_text() for text in ax.get_legend().texts]
    assert len(roi_names) == len(labels)
    while roi_names:
        roi_name = roi_names.pop()
        assert roi_name in labels
        labels.remove(roi_name)

def test_pathlib_path():
    # Test passing of pathlib.Path.
    dcm_dir = Path(tmp_path) / 'dose_dcm'
    dose = Dose(dcm_dir)
    assert dose.path == fullpath(dcm_dir)

def test_get_dose_in_roi_3d():
    """Test retrieval of dose array representing dose to ROI."""
    # Define synthetic dose field and associated structure set.
    doses = {"dose_cube": 10, "dose_sphere": 100}
    dose, structure_set = get_synthetic_data(**doses)

    # Introduce size mismatch between dose array and ROI masks.
    dxyz = np.array((20, 20, 4))
    dose.resize(image_size=list(np.array(dose.get_n_voxels() - dxyz)),
                                keep_centre=True)
    assert (list(np.array(dose.get_n_voxels() + dxyz))
                 == structure_set.image.get_n_voxels())
    dose_data = dose.get_data()

    # Loop over ROIs.
    for key, roi_dose in doses.items():
        # Obtain dose array with non-zero values only in region of ROI.
        roi_name = key.split("_")[-1]
        roi = structure_set[roi_name]
        dose_in_roi = dose.get_dose_in_roi_3d(roi)

        # Check that dose values and number of voxels in ROI are as expected.
        assert dose_in_roi.max() == roi_dose
        assert np.all(dose_in_roi[dose_in_roi > 0] == roi_dose)
        # Expect only approximate agreement between number of voxels with
        # non-zero dose and number of voxels in ROI,
        # because of ROI resizing in skrt.dose.Dose.get_dose_in_roi_3d().
        assert ((dose_in_roi > 0).sum()
                == pytest.approx(roi.get_volume("voxels"), rel=0.02))

def test_get_biologically_effective_dose():
    """Test calculation of biologically effective dose."""
    # Define synthetic dose field and associated structure set.
    doses = {"dose_cube": 10, "dose_sphere": 100}
    dose, structure_set = get_synthetic_data(**doses)
    dose_data = dose.get_data()

    # Obtain dose object representing biologically effective dose (BED),
    # using long and short method names.
    alpha_beta_ratios = {"cube": 3, "sphere": 5}
    n_fraction = 20
    bed1 = dose.get_biologically_effective_dose(structure_set,
            alpha_beta_ratios, n_fraction)
    bed2 = dose.get_bed(structure_set, alpha_beta_ratios, n_fraction)

    small_number = 1e-8
    # Loop over ROIs.
    for key, roi_dose in doses.items():
        # Obtain BED array with non-zero values only in region of ROI.
        roi_name = key.split("_")[-1]
        roi = structure_set[roi_name]
        for bed in [bed1, bed2]:
            bed_in_roi = bed.get_dose_in_roi_3d(roi)

            # Check that BED values and number of voxels in ROI are as expected.
            roi_bed = roi_dose * (1 + roi_dose /
                    (n_fraction * alpha_beta_ratios[roi_name]))
            assert bed_in_roi.max() == pytest.approx(roi_bed, small_number)
            assert np.all(bed_in_roi[bed_in_roi > 0] == pytest.approx(roi_bed))
            assert (bed_in_roi > 0).sum() == roi.get_volume("voxels")
