"""Test prototype image class."""

import os

import random

import numpy as np
import pytest

from voxtox.image import Image
from skrt.simulation import SyntheticImage

from voxtox.core import COUCH_SHIFTS_GROUP, get_couch_shifts, \
        ROTATIONS_ELEMENT, TRANSLATIONS_ELEMENT

random.seed(14)

# Origin and voxel size for image data
voxel_size = [2, 2, 3]
shape = [41, 51, 21]
origin = [-0.5 * ((shape[i] - 1) * voxel_size[i]) for i in range(3)]

def create_test_image():
    sim = SyntheticImage(shape=shape, voxel_size=voxel_size, origin=origin,
        intensity=0)
    sim.add_cube(side_length=6, centre=(6, -6, 12), name="cube", intensity=100)
    image = Image(path=sim.get_data().astype(np.float64),
            voxel_size=voxel_size, origin=origin)
    return image

image = create_test_image()

# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# Create a test dicom file
dcm_file = "tmp/tmp_dcm"
image.write(dcm_file)
im_dcm = Image(dcm_file)
im_dcm.data = im_dcm.get_data().astype(np.float64)

# Define private block and offsets for adding couch shifts to the dicom dataset
block = im_dcm.dicom_dataset.private_block(
        COUCH_SHIFTS_GROUP, 'Couch translations and rotations', create=True)
translations_offset = TRANSLATIONS_ELEMENT - 0x1000
rotations_offset = ROTATIONS_ELEMENT - 0x1000

def test_apply_couch_shifts():
    '''
    Check effect of applying and reversing couch shifts.

    Rounding errors can result in twice-transformed image
    not being identical to the original.
    '''
# Translations in mm, rotations in degrees
    n_test = 100
    small_number = 750
    big_number = 5000
    n_zero = 0

    for i in range(n_test):
        sign = 1 if random.random() < 0.5 else -1
        translations = [random.uniform(5, 10) * sign for i in range(3)]
        rotations = [0, 0, random.uniform(-5, 5)]
        block.add_new(
                translations_offset, 'DS', [f'{x:.3f}' for x in translations])
        block.add_new(rotations_offset, 'DS', [f'{x:.3f}' for x in rotations])

        im1 = Image(im_dcm)
        im1.apply_couch_shifts()
        abs_diff = np.abs(im1.get_data() - im_dcm.get_data())
        assert abs_diff.sum() > big_number

        im1.apply_couch_shifts(reverse=True)
        abs_diff = np.abs(im1.get_data() - im_dcm.get_data())
        assert abs_diff.sum() < small_number

        if abs_diff.sum() ==0:
            n_zero += 1

    assert (n_zero / n_test) > 0.8
