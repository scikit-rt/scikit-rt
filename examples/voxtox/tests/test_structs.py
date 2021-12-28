"""Tests of VoxTox extensions to ROI and StructureSet classes."""

import math
import os
from pathlib import Path
import random

from pytest import approx

from skrt.image import Image
from skrt.simulation import SyntheticImage

from voxtox.core import COUCH_SHIFTS_GROUP, get_couch_shifts, \
        ROTATIONS_ELEMENT, TRANSLATIONS_ELEMENT
from voxtox.structures import ROI, StructureSet

random.seed(14)

# Make temporary test dir.
tmp_dir = Path('tmp')
tmp_dir.mkdir(exist_ok=True)

# Create synthetic structure set.
sim = SyntheticImage((100, 100, 40))
sim.add_cube(side_length=40, name="cube", intensity=1)
sim.add_sphere(radius=20, name="sphere", intensity=10)
structure_set = StructureSet()
sim.get_structure_set().clone_attrs(structure_set)
structure_set.rois = []
structure_set.sources = []
for roi in sim.get_structure_set().get_rois():
    roi_voxtox = ROI()
    roi.clone_attrs(roi_voxtox)
    structure_set.add_roi(roi_voxtox)

for name in structure_set.get_roi_names():
    roi_path = tmp_dir / f'{name}.txt'
    roi_path.unlink(missing_ok=True)

# Create a test dicom file
dcm_file = "tmp/tmp_dcm"
im = Image(sim)
im.write(dcm_file)
im_dcm = Image(dcm_file)
im_dcm.data = im_dcm.get_data()

# Define private block and offsets for adding couch shifts to the dicom dataset
block = im_dcm.dicom_dataset.private_block(
        COUCH_SHIFTS_GROUP, 'Couch translations and rotations', create=True)
translations_offset = TRANSLATIONS_ELEMENT - 0x1000
rotations_offset = ROTATIONS_ELEMENT - 0x1000

def test_structure_set_write_point_file():

    assert list(sim.get_structure_set().get_roi_names()) \
            == list(structure_set.get_roi_names())

    # Write structure set as point cloud.
    structure_set.write(outdir=str(tmp_dir), point_cloud=True)

    for name in structure_set.get_roi_names():
        # Check that the ROI point cloud was written.
        assert (tmp_dir / f'{name}.txt').exists()

        # Load the point-cloud data,
        # then check that the contours obtain are the same as the originals.
        ss1 = StructureSet(sources=str(tmp_dir), image=sim)

        # Check centroids.
        assert structure_set.get_roi(name).get_centroid().all() == \
                ss1.get_roi(name).get_centroid().all()

        # Check number of planes.
        ss0_keys = list(structure_set.get_roi(name).get_contours().keys())
        ss1_keys = list(ss1.get_roi(name).get_contours().keys())
        assert len(ss0_keys) == len(ss1_keys)

        ss0_keys.sort()
        ss1_keys.sort()

        for i in range(len(ss0_keys)):

            # Check plane z-coordinates.
            assert ss0_keys[i] == ss1_keys[i]

            # Check number of contours in plane.
            contours0 = structure_set.get_roi(name).get_contours()[ss0_keys[i]]
            contours1 = ss1.get_roi(name).get_contours()[ss1_keys[i]]
            assert len(contours0) == len(contours1)

            # Check contour points.
            for j in range(len(contours0)):
                assert contours0[j].all() == contours1[j].all()

def test_apply_couch_shifts():
    '''
    Check effect of applying and reversing couch shifts.
    '''
    # Number of sets of couch shifts to test.
    n_test = 10

    # Constraints on absolute translations and on rotation
    dxyz_min = 5
    dxyz_max = 50
    roll_min = 0
    roll_max = 360

    for i_test in range(n_test):
        # Add random couch shifts to DICOM dataset.
        sign = 1 if random.random() < 0.5 else -1
        translations = [random.uniform(dxyz_min, dxyz_max) * sign
                for i in range(3)]
        rotations = [0, 0, random.uniform(roll_min, roll_max)]
        dx, dy, dz = translations
        rad_angle = math.radians(rotations[2])
        block.add_new(
                translations_offset, 'DS', [f'{x:.3f}' for x in [dx, -dz, dy]])
        block.add_new(rotations_offset, 'DS', [f'{x:.3f}' for x in rotations])

        # Create structure set from original, and update image settings.
        ss1 = structure_set.clone()
        im1 = Image(im_dcm)
        ss1.set_image(im1)
        
        # Apply couch shifts.
        ss2 = ss1.clone()
        ss2.apply_couch_shifts()

        # Apply reverse couch shifts.
        ss3 = ss2.clone()
        ss3.apply_couch_shifts(reverse=True)

        for name in structure_set.get_roi_names():

            # Check centroids.
            centroid1 = list(ss1.get_roi(name).get_centroid(method='contour'))
            centroid2 = list(ss2.get_roi(name).get_centroid(method='contour'))
            centroid3 = list(ss3.get_roi(name).get_centroid(method='contour'))
            for i in range(3):
                assert centroid1[i] != approx(centroid2[i], abs=0.01)
                assert centroid1[i] == approx(centroid3[i], abs=0.01)

            # Check number of planes.
            ss1_keys = list(ss1.get_roi(name).get_contours().keys())
            ss2_keys = list(ss2.get_roi(name).get_contours().keys())
            ss3_keys = list(ss3.get_roi(name).get_contours().keys())
            assert len(ss1_keys) == len(ss2_keys)
            assert len(ss1_keys) == len(ss3_keys)

            ss1_keys.sort()
            ss2_keys.sort()
            ss3_keys.sort()

            for i in range(len(ss1_keys)):

                # Check plane z-coordinates.
                assert ss1_keys[i] == approx(ss3_keys[i], abs=0.01)
                assert ss2_keys[i] - ss1_keys[i] == approx(dz, abs=0.01)

                # Check number of contours in plane.
                contours1 = ss1.get_roi(name).get_contours()[ss1_keys[i]]
                contours2 = ss2.get_roi(name).get_contours()[ss2_keys[i]]
                contours3 = ss3.get_roi(name).get_contours()[ss3_keys[i]]
                assert len(contours1) == len(contours2)
                assert len(contours1) == len(contours3)

                # Check contour points.
                for j in range(len(contours1)):
                    assert len(contours1[j]) == len(contours2[j])
                    assert len(contours1[j]) == len(contours3[j])
                    for k in range(len(contours1[j])):
                        x1, y1 = contours1[j][k]
                        x2, y2 = contours2[j][k]
                        x3, y3 = contours3[j][k]

                        # Check that applying couch shifts
                        # gives expected coordinate changes
                        xt = (x1 + dx)  * math.cos(rad_angle) \
                                - (y1 + dy) * math.sin(rad_angle)
                        yt = (x1 + dx)  * math.sin(rad_angle) \
                                + (y1 + dy) * math.cos(rad_angle)
                        assert x2 == approx(xt, abs=0.01)
                        assert y2 == approx(yt, abs=0.01)

                        # Check that applying then reversing couch shifts
                        # returns original coordinates
                        assert x1 == approx(x3, abs=0.01)
                        assert y1 == approx(y3, abs=0.01)
