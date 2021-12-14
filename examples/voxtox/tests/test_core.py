"""Test prototype image class."""

import os
import shutil

import numpy as np
import pydicom
import pytest

from skrt.image import Image

from voxtox.core import couch_shifts_group, get_couch_shifts, \
        rotations_element, translations_element

# Create fake data
data = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)
# Translations in mm, rotations in degrees
translations0 = list(20 * np.random.rand(3) - 10)
rotations0 = list(10 * np.random.rand(3) - 5)
# Storage order for VoxTox data
dx, dy, dz = translations0
translations1 = [dx, -dz, dy]
rotations1 = rotations0

def create_test_image():
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

im = create_test_image()

# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# Create a test dicom file
dcm_file = "tmp/tmp_dcm"
im.write(dcm_file)
im_dcm = Image(dcm_file)

# Add couch shifts to the dicom dataset
block = im_dcm.dicom_dataset.private_block(
        couch_shifts_group, 'Couch translations and rotations', create=True)
translations_offset = translations_element - 0x1000
block.add_new(translations_offset, 'DS', [f'{x:.3f}' for x in translations1])
rotations_offset = rotations_element - 0x1000
block.add_new(rotations_offset, 'DS', [f'{x:.3f}' for x in rotations1])

def test_couch_shifts():
    '''Check that couch shifts retrieved are the same as the ones set'''

    translations2, rotations2 = get_couch_shifts(im_dcm) 
    for i in [0, 1, 2]:
        assert(translations0[i] == pytest.approx(translations2[i], abs=1.e-3))
        assert(rotations0[i] == pytest.approx(rotations2[i], abs=1.e-3))
