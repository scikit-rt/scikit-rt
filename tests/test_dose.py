"""Test functionality of the Dose class."""

from pathlib import Path
import pytest

import numpy as np

from skrt import Dose, Image
from skrt.simulation import SyntheticImage

# Make random arrays to use for an image and a dose map
im_array = np.random.rand(100, 150, 30)
dose_array = np.random.rand(100, 150, 30)

# Make temporary test dir
tmp_path = Path('tmp')
if not tmp_path.exists():
    tmp_path.mkdir()

def get_synthetic_data():
    # Create synthetic structure set
    sim = SyntheticImage((100, 100, 40))
    sim.add_cube(side_length=40, name="cube", intensity=1)
    sim.add_sphere(radius=20, name="sphere", intensity=10)
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
