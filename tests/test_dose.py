"""Test functionality of the Dose class."""

from pathlib import Path
import pytest

import numpy as np

from skrt import Dose, Image

# Make random arrays to use for an image and a dose map
im_array = np.random.rand(100, 150, 30)
dose_array = np.random.rand(100, 150, 30)

# Make temporary test dir
tmp_path = Path('tmp')
if not tmp_path.exists():
    tmp_path.mkdir()

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
    assert bv.viewers[0].image == im
    assert bv.viewers[0].dose == dose
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
    dose_dcm1.write(dcm2_dir, modality='RTDOSE')
    dose_dcm2 = Dose(dcm2_dir)
    assert dose_dcm1.data.shape == dose_dcm2.data.shape
    assert np.all(dose_dcm1.affine == dose_dcm2.affine)
    assert np.abs(dose_dcm1.get_data() - dose_dcm2.get_data()).max() < 0.005
