"""Test functionality of the Dose class."""

import numpy as np

from skrt import Dose, Image

# Make random arrays to use for an image and a dose map
im_array = np.random.rand(100, 150, 30)
dose_array = np.random.rand(100, 150, 30)


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
