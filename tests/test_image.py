"""Test Image class."""

from pathlib import Path

import math
import os
import random
import pytest
import numpy as np
import pandas as pd
import shutil
import pydicom

from pydicom._storage_sopclass_uids import\
        PositronEmissionTomographyImageStorage

from skrt.core import Defaults, File, fullpath
from skrt.image import (Image, checked_crop_limits, get_alignment_translation,
                        get_geometry, get_image_comparison_metrics,
                        get_mask_bbox, get_translation_to_align,
                        match_images, match_image_voxel_sizes, rescale_images)
from skrt.simulation import SyntheticImage
from skrt.structures import ROI

try:
    import mahotas
    has_mahotas = True
except ModuleNotFoundError:
    has_mahotas = False

# Decorator for tests requiring mahotas
def needs_mahotas(func):
    def wrapper():
        if not has_mahotas:
            return
        else:
            func()
    return wrapper

# Create fake data
def create_test_image(shape, voxel_size, origin, data_type='rand', factor=1000):
    if 'rand' == data_type:
        n1, n2, n3 = shape
        data = (np.random.rand(n1, n2, n3) * factor).astype(np.uint16)
    elif 'zeros' == data_type:
        data = np.zeros(shape).astype(np.uint16)
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

shape = (40, 50, 20)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)
affine = np.array(
        [
            [voxel_size[0], 0, 0, origin[0]],
            [0, voxel_size[1], 0, origin[1]],
            [0, 0, voxel_size[2], origin[2]],
            [0, 0, 0, 1],
            ]
        )

im = create_test_image(shape, voxel_size, origin)

data = im.data

# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

# Create a test nifti file
nii_file = "tmp/tmp.nii"
im.write(nii_file)
im_nii = Image(nii_file)

# Create a test dicom file
dcm_file = "tmp/tmp_dcm"
im.write(dcm_file, header_extras={'RescaleSlope': 1})
im_dcm = Image(dcm_file)


############################
# Test reading and writing #
############################
def test_array_reading():
    """Check voxel sizes, origin, and affine matrix are correctly set for a 
    numpy array."""

    assert all([voxel_size[i] == im.voxel_size[i] for i in range(3)])
    assert all([origin[i] == im.origin[i] for i in range(3)])
    assert all([voxel_size[i] == im.affine[i, i] for i in range(3)])
    assert all([origin[i] == im.affine[i, 3] for i in range(3)])
    assert im.data.shape == data.shape
    assert im.n_voxels[0] == data.shape[1]
    assert im.n_voxels[1] == data.shape[0]
    assert im.n_voxels[2] == data.shape[2]

def test_dicom_array():
    """Check dicom array is the same as the input array."""

    ddata, daffine = im.get_dicom_array_and_affine()
    assert np.all(ddata == data)
    assert np.all(daffine.astype(int) == im.affine)

def test_nifti_array():
    """Check nifti array is correctly transposed wrt input array."""

    ndata, naffine = im.get_nifti_array_and_affine()
    assert ndata.shape[1] == data.shape[0]
    assert ndata.shape[0] == data.shape[1]
    assert ndata.shape[2] == data.shape[2]
    assert naffine[1, 3] != origin[1]

def test_array_to_npy():
    """Check numpy array can be saved to a .npy file and read in correctly."""

    outname = "tmp/tmp.npy"
    im.write(outname)
    im_npy = Image(outname, affine=im.affine)
    assert np.all(im_npy.data == im.data)
    assert np.all(im_npy.affine == im.affine)
    assert os.path.exists(outname.replace("npy", "txt"))

def test_array_to_nifti_npy():
    """Check numpy array is correctly saved in nifti-style."""

    outname = "tmp/tmp_nii.npy"
    im.write(outname, nifti_array=True)
    im_npy_nifti = Image(outname, nifti_array=True, voxel_size=voxel_size,
                         origin=origin)
    ndata, naffine = im.get_nifti_array_and_affine()
    print(naffine)
    print(im_npy_nifti.affine)
    assert ndata.shape == im_npy_nifti.data.shape
    assert np.all(ndata == im_npy_nifti.data)
    assert np.all(im._saffine == im_npy_nifti._saffine)
    assert np.all(im.get_standardised_data() 
                  == im_npy_nifti.get_standardised_data())

def test_array_to_nifti():
    """Check numpy array is correctly saved to nifti."""

    # Data and affine should be the same as nifti data and affine in 
    # original image
    ndata, naffine = im.get_nifti_array_and_affine()
    assert np.all(ndata == im_nii.data)
    assert np.all(naffine == im_nii.affine)

    # Standarised data and affine should be the same for both
    im.standardise_data()
    im_nii.standardise_data()
    assert np.all(im._sdata == im_nii._sdata)
    assert np.all(im._saffine == im_nii._saffine)

def test_nifti_to_nifti():
    """Check a nifti file can be written and read correctly."""

    # Write nifti image to second nifti file
    nii2 = "tmp/tmp2.nii"
    im_nii.write(nii2)
    im_nii2 = Image(nii2)

    # Check data and affine matrix is the same
    assert np.all(im_nii.data == im_nii2.data)
    assert np.all(im_nii.affine == im_nii2.affine)

def test_nifti_to_npy():
    """Check that a nifti file is correctly written to a numpy file."""

    # Write to numpy file
    npy = "tmp/tmp2.npy"
    im_nii.write(npy)
    affine_dcm = im_nii.get_dicom_array_and_affine()[1]
    im_npy = Image(npy, affine=affine_dcm)

    # Check nifti array matches
    ndata, naffine = im_npy.get_nifti_array_and_affine()
    assert np.all(naffine == im_nii.affine)
    assert np.all(ndata == im_nii.data)

    # Check standardised data matches
    im_nii.standardise_data()
    im_npy.standardise_data()
    assert np.all(im_nii._saffine == im_npy._saffine)
    assert np.all(im_nii._sdata == im_npy._sdata)

def test_dcm_to_nifti():
    """Check that a dicom file is correctly written to nifti."""

    # Write dicom to nifti
    nii = "tmp/tmp_dcm2nii.nii"
    im_dcm.write(nii)
    im_dcm2nii = Image(nii)

    # Check nifti array is the same
    ndata, naffine = im_dcm.get_nifti_array_and_affine()
    assert np.all(naffine == im_dcm2nii.affine)
    assert np.all(ndata == im_dcm2nii.data)

    # Check standardised data is the same
    im_dcm.standardise_data()
    im_dcm2nii.standardise_data()
    assert np.all(im_dcm._saffine == im_dcm2nii._saffine)
    assert np.all(im_dcm._sdata == im_dcm2nii._sdata)

def test_dcm_to_dcm():
    """Check that a dicom file is correctly written to dicom."""

    # Write to dicom
    dcm = "tmp/tmp_dcm2"
    series_description = 'Image series'
    study_description = 'Image study'
    # Test that tag can be with or without spaces
    header_extras = {
            'RescaleSlope' : 1,
            'SeriesDescription' : series_description,
            'Study Description' : study_description}
    im_dcm.write(dcm, header_extras=header_extras)
    im_dcm2 = Image(dcm)
    assert im_dcm2.get_dicom_dataset().SeriesDescription == series_description
    assert im_dcm2.get_dicom_dataset().StudyDescription == study_description

    assert im_dcm.data.shape == im_dcm2.data.shape
    assert np.all(im_dcm.affine == im_dcm2.affine)
    assert np.all(im_dcm.data == im_dcm2.data)

def test_nifti_to_dcm():
    """Check that a nifti file can be written to dicom using a fresh header."""

    # Write nifti to dicom, setting modality as PT
    dcm = "tmp/tmp_nii2dcm"
    im_nii.write(dcm, modality='PT', header_extras={'RescaleSlope': 1})
    im_nii2dcm = Image(dcm)

    # Check standardised data and affine are the same
    im_nii.standardise_data()
    im_nii2dcm.standardise_data()
    assert np.all(im_nii._saffine == im_dcm._saffine)
    assert np.all(im_nii._sdata == im_dcm._sdata)

    # Check nifti array and affine are the same
    ndata, naffine = im_nii2dcm.get_nifti_array_and_affine()
    assert np.all(im_nii.affine == naffine)
    assert np.all(im_nii.data == ndata)

    # Check modality-related information correctly set
    ds = im_nii2dcm.get_dicom_dataset()
    assert ds.file_meta.MediaStorageSOPClassUID == (
            PositronEmissionTomographyImageStorage)
    assert ds.SOPClassUID == PositronEmissionTomographyImageStorage
    assert ds.Modality == 'PT'

def test_dcm_to_nifti_to_dcm():
    """Check that a nifti file can be written to dicom using the header of 
    the dicom that was used to create that nifti file."""

    # Write dicom to nifti
    nii = "tmp/tmp_dcm2nii.nii"
    im_dcm.write(nii)
    im_dcm2nii = Image(nii)

    # Write nifti to new dicom
    dcm = "tmp/tmp_nii2dcm"
    im_dcm2nii.write(dcm, header_source=im_dcm.source)
    im_dcm2 = Image(dcm)

    # Check data is the same
    assert np.all(im_dcm.affine == im_dcm2.affine)
    assert np.all(im_dcm.data == im_dcm2.data)

def test_array_to_dcm():
    """check numpy array is correctly saved to dicom."""

    # Data and affine should be the same for numpy array and dicom
    assert im.data.shape == im_dcm.data.shape
    assert np.all(im.affine == im_dcm.affine)
    assert np.all(im.data == im_dcm.data)

def test_resampling():
    """Test z axis voxel size resampling."""

    # Create inital image
    init_shape = [100, 100, 30]
    init_voxel_size = [1, 1, 5]
    im1 = Image(np.random.rand(*init_shape), voxel_size=init_voxel_size)
    im2 = Image(np.random.rand(*init_shape), voxel_size=init_voxel_size)

    # Resample
    im2.resample((None, None, init_voxel_size[2] / 2))
    assert im2.voxel_size[2] == init_voxel_size[2] / 2
    assert im2.data.shape[2] == init_shape[2] * 2

    # Resample to original shape and check data is approximately the same
    im2.resample(init_voxel_size)
    assert [abs(int(i)) for i in im2.voxel_size] == init_voxel_size
    assert list(im2.get_data().shape) == init_shape

def test_resampling_3d():
    """Test resampling to different voxel size in three dimensions."""

    # Create inital images
    init_shape = (31, 40, 25)
    init_voxel_size = (1, 2, 3)
    im0 = Image(np.random.rand(*init_shape), voxel_size=init_voxel_size)

    # Resample to different multiples of original voxel size in each direction
    small_number = 1.e-6
    for factor in [1 / 3, 1 / 2, 2, 5]:
        voxel_size = [x * factor for x in init_voxel_size]
        im1 = Image(im0)
        im1.resample(voxel_size)
        for i in range(3):
            assert im1.voxel_size[i] == im0.voxel_size[i] * factor
            assert abs(im1.data.shape[i] - im0.data.shape[i] / factor) < 1
            assert im1.origin[i] == im0.origin[i] + \
                   (im1.voxel_size[i] -  im0.voxel_size[i]) / 2
            assert im1.image_extent[i][0] == im0.image_extent[i][0]
            if factor < 1:
                assert im1.image_extent[i][1] == pytest.approx(
                        im0.image_extent[i][1], small_number)
            else:
                assert im1.image_extent[i][1] == pytest.approx(
                        im0.image_extent[i][1], im0.voxel_size[i])

def test_resize_and_match_size():
    """Test image resizing and matching to reference."""

    # Create inital images
    shape_1 = (40, 40, 40)
    voxel_size_1 = (1, 2, 3)
    origin_1 = (-100, -100, -100)
    resize_centre_1 = (-75, -80, 0)
    shape_2 = (50, 50, 50)
    voxel_size_2 = (1, 2, 3)
    origin_2 = (-150, -120, -80)
    resize_centre_2 = (-140, -30, 20)

    im1 = Image(np.random.rand(*shape_1), voxel_size=voxel_size_1,
            origin=origin_1)
    im1.set_geometry()

    im2 = Image(np.random.rand(*shape_2), voxel_size=voxel_size_2,
            origin=origin_2)
    im2.set_geometry()

    # Resize im1
    for image0, image2, image_size, origin, voxel_size, resize_centre in [
            (im1, im2, shape_2, origin_2, voxel_size_2, resize_centre_1),
            (im2, im1, shape_1, origin_1, voxel_size_1, resize_centre_2)]:
        image_size = list(image_size)
        origin = list(origin)
        voxel_size = list(voxel_size)
        for i in range(4):
            # Include check that None values are handled.
            if i < 3:
                image_size[i] = None
                origin[i] = None
                voxel_size[i] = None
            image1 = Image(image0)
            image1.resize(image_size, origin, voxel_size)
            for i in range(3):
                if image_size[i] is None:
                    image3 = image1
                else:
                    image3 = image2
                assert image1.voxel_size[i] == image3.voxel_size[i]
                assert image1.n_voxels[i] == image3.n_voxels[i]
                assert image1.image_extent[i] == image3.image_extent[i]
                assert image1.origin[i] == image3.origin[i]

            # Check case where centre position is fixed.
            image1 = Image(image0)
            image1.resize(image_size, origin, voxel_size, keep_centre=True)
            for i in range(3):
                if image_size[i] is None:
                    image3 = image1
                else:
                    image3 = image2
                assert image1.voxel_size[i] == image3.voxel_size[i]
                assert image1.n_voxels[i] == image3.n_voxels[i]
                assert image1.get_length(i) == image3.get_length(i)
                assert image1.get_centre()[i] == image0.get_centre()[i]

            # Check case where centre position is chosen.
            image1 = Image(image0)
            image1.resize(image_size, origin, voxel_size, centre=resize_centre)
            for i in range(3):
                if image_size[i] is None:
                    image3 = image1
                else:
                    image3 = image2
                assert image1.voxel_size[i] == image3.voxel_size[i]
                assert image1.n_voxels[i] == image3.n_voxels[i]
                assert image1.get_length(i) == image3.get_length(i)
                assert image1.get_centre()[i] == resize_centre[i]

    # Resize im1 to im2
    for image0, image2 in [(im1, im2), (im2, im1)]:
        image1 = Image(image0)
        image1.match_size(image2)
        assert image1.voxel_size == image2.voxel_size
        assert image1.data.shape == image2.data.shape
        assert image1.image_extent == image2.image_extent
        assert image1.origin == image2.origin
        image_diff = np.absolute(image1.data - image2.data)
        assert np.count_nonzero(image_diff > 0.5) == pytest.approx(
                0.5 * image_diff.size, rel=0.02)

def test_match_images():
    """Minimal test of image matching."""
    # Define image shapes.
    shape1 = [40, 40, 20]
    shape2 = [10, 10, 55]

    # Create test images.
    im1 = SyntheticImage(shape1).get_image()
    im2 = SyntheticImage(shape2).get_image()

    # Check that image sizes are unchanged for alignment set to False.
    im1a, im2a = match_images(im1, im2, alignment=False)
    assert im1a.get_size() == im1.get_size()
    assert im2a.get_size() == im2.get_size()

    # Check that image sizes are mached for alignment not set to False
    alignments = [None, "_centre_", "_top_", "_bottom_"]
    for alignment in alignments:
        im1a, im2a = match_images(im1, im2, alignment=alignment)
        assert im1a.get_size() == im2a.get_size()

def test_match_image_voxel_sizes():
    """Test resampling images to match voxel sizes."""

    # Define image voxel sizes and shapes.
    vs1 = [1, 1, 4]
    vs2 = [2, 2, 2]
    vs3 = [3, 3, 3]
    shape1 = [40, 40, 20]
    shape2 = [10, 10, 55]

    # Create test images.
    im1 = SyntheticImage(shape1, voxel_size=vs1).get_image()
    im2 = SyntheticImage(shape2, voxel_size=vs2).get_image()

    # Test different options for matching voxel sizes.
    for vs_in, vs_out in (
            ("dz_max", vs1), ("dz_min", vs2), (vs3, vs3), (vs3[0], vs3)):
        im1a, im2a = match_image_voxel_sizes(im1.clone(), im2.clone(), vs_in)
        # Check that voxel sizes are as expected.
        assert im1a.get_voxel_size() == vs_out
        assert im2a.get_voxel_size() == vs_out
        # Check that image centres have stayed approximately the same.
        assert im1a.get_centre() == pytest.approx(im1.get_centre(), 0.5)
        assert im2a.get_centre() == pytest.approx(im2.get_centre(), 0.5)

def test_clone():
    """Test cloning an image."""

    im_cloned = im.clone()
    assert np.all(im.get_affine() == im_cloned.get_affine())
    assert np.all(im.get_data() == im_cloned.get_data())
    assert np.all(im.get_standardised_data() 
                  == im_cloned.get_standardised_data())

    # Check that changing the new array doesn"t change the old one
    assert im.data is not im_cloned.data
    im.data[0, 0, 0] = 1
    im_cloned.data[0, 0, 0] = 2
    assert im.data[0, 0, 0] != im_cloned.data[0, 0, 0]

    # Check that changing the new date doesn"t change the old one
    im_cloned.date = "new_date"
    assert im.date != im_cloned.date

    # Check that changing the old file list doesn"t change the new one
    im.files.append(File())
    assert len(im.files) == (len(im_cloned.files) + 1)

    # Check that adding an attribute to the old image doesn"t add it to the new
    im.user_addition = {"uno": 1, "due": 2, "tre": 3}
    assert not hasattr(im_cloned, "user_addition")

    # Check the recloning includes the added attribute
    im_cloned = Image(im)
    assert hasattr(im_cloned, "user_addition")

    # Check that changing the new dictionary doesn"t change the old one
    im_cloned.user_addition["quattro"] = 4
    assert "quattro" not in im.user_addition

def test_clone_no_copy():
    """Test cloning an image without copying its data."""

    im_cloned = im.clone(copy_data=False)
    assert im.data is im_cloned.data

def test_init_from_image():
    """Test cloning an image using the Image initialiser."""

    im = create_test_image(shape, voxel_size, origin)
    im_cloned = Image(im)
    assert np.all(im.get_affine() == im_cloned.get_affine())
    assert np.all(im.get_data() == im_cloned.get_data())
    assert np.all(im.get_standardised_data()
                  == im_cloned.get_standardised_data())

def test_dicom_dataset():
    """Check that dicom dataset property is assigned."""

    assert isinstance(im_dcm.get_dicom_dataset(), pydicom.dataset.FileDataset)
    for i in range(1, 1 + len(im_dcm.files)):
        assert isinstance(
                im_dcm.get_dicom_dataset(i), pydicom.dataset.FileDataset)

def test_dicom_filepath():
    """Check that filepaths are retrieved correctly."""

    i = 0
    for dicom_file in im_dcm.files:
        i += 1
        assert im_dcm.get_dicom_filepath(i) == dicom_file.path

def test_dicom_dataset_slice():
    """Check that a dataset can be retrieved for a specific slice."""

    idx = 2
    ds = im_dcm.get_dicom_dataset(idx=2)
    assert ds.ImagePositionPatient[2] == im_dcm.idx_to_pos(idx, "z")

def test_null_image():
    im_null = Image()
    assert(type(im).__name__ == "Image")
    assert(im_null.affine is None)
    assert(im_null.data is None)
    assert(im_null.date == "")
    assert(im_null.downsampling is None)
    assert(im_null.files == [])
    assert(not im_null.nifti_array)
    assert(im_null.origin == [0, 0, 0])
    assert(im_null.path == "")
    assert(im_null.source == "")
    assert(im_null.source_type is None)
    assert(im_null.subdir == "")
    assert(im_null.time == "")
    assert(im_null.timestamp == "")
    assert(im_null.title is None)
    assert(im_null.voxel_size == [1, 1, 1])

def test_plot():
    plot_out = "tmp/plot.pdf"
    im.plot(show=False, save_as=plot_out)
    assert os.path.isfile(plot_out)
    os.remove(plot_out)

def get_random_voxel(im, fraction=1):
    '''
    Obtain centre coordinates of random voxel in image

    Parameters
    ----------
    im : skrt.image.Image
        Image object for which coordinates of random voxel are to be retrieved
    fraction : float, default = 1
        Fraction of image extents to be considered.
    '''
    xyz_voxel = []
    for i in range(3):
        v1, v2 = [im.image_extent[i][j] * fraction for j in [0, 1]]
        v = random.uniform(v1, v2)
        # Obtain coordinates corresponding vo voxel centre
        iv = im.pos_to_idx(v, i)
        v = im.idx_to_pos(iv, i)
        xyz_voxel.append(v)
    return xyz_voxel

def test_translation():
    '''Test translation - track movement of single bright voxel.'''
    shape=(31, 31, 31)
    voxel_size=(1, 1, 3)
    origin=(-15, -15, -45)
    im0 = create_test_image(shape, voxel_size, origin, 'zeros')
    random.seed(1)
    # Intensity value for bright voxel.
    v_test = 1000
    # Number of translations to test.
    n_test = 20

    for i in range(n_test):
        im1 = Image(im0)
        # Obtain coordinates for original and translated point.
        x0, y0, z0 = get_random_voxel(im1)
        x1, y1, z1 = get_random_voxel(im1)
        ix0 = im1.pos_to_idx(x0, 'x')
        iy0 = im1.pos_to_idx(y0, 'y')
        iz0 = im1.pos_to_idx(z0, 'z')
        ix1 = im1.pos_to_idx(x1, 'x')
        iy1 = im1.pos_to_idx(y1, 'y')
        iz1 = im1.pos_to_idx(z1, 'z')
        im1.data[iy0][ix0][iz0] = v_test

        # Determine translation.
        translation = [x1-x0, y1-y0, z1-z0]

        # Check that bright voxel is in expected position in original image.
        assert np.sum(im1.data >= 0.9 * v_test) == 1
        assert im1.data[iy0][ix0][iz0] == v_test

        # Translate image
        im1_a = Image.clone(im1)
        im1_a.transform(translation=translation, order=0)

        # Check that bright voxel is in expected position in translated image.
        assert np.sum(im1_a.data >= 0.9 * v_test) == 1
        assert im1_a.data[iy1][ix1][iz1] == v_test

        # Translate origin
        im1_b = Image.clone(im1)
        im1_b.transform(translation=translation, order=0)

        # Check that bright voxel is in expected position after translating origin.
        assert np.sum(im1_b.data >= 0.9 * v_test) == 1
        assert im1_b.data[iy1][ix1][iz1] == v_test

def get_plane_data(iplane, xyzc, xyz0, xyz1, min_length=1000,
        min_scale=0.8, max_scale=1.2):
    '''
    Extract plane data for checking rotations.

    Parameters
    ----------
    iplane: int
        Plane identifier:
            0 - perpendicular to x-axis;
            1 - perpendicular to y-axis;
            2 - perpendicular to z-axis;

    xyzc: tuple
        (x, y, z) coordinates of point of rotation.

    xyz0: tuple
        (x, y, z) coordinates of unrotated point.

    xyz1: tuple
        (x, y, z) coordinates of rotated point.

    min_length: float, default=1000
        Minimum length (mm) of displacement vectors.

    min_scale: float, default=0.8
        Minimum ratio of displacement vector lengths.

    max_scale: float, default=1.2
        Maximum ratio of displacement vector lengths.
    '''

    # Obtain coordinates in plane for centre of rotation and rotated point.
    centre = list(xyzc)
    centre[iplane] = xyz0[iplane]
    xyzr = list(xyz1)
    xyzr[iplane] = xyz0[iplane]

    # Calculate displacement vectors and their lengths.
    vec0 = np.array(xyz0) - np.array(centre)
    vecr = np.array(xyzr) - np.array(centre)
    len0 = np.linalg.norm(vec0)
    lenr = np.linalg.norm(vecr)

    scale = None
    if min(len0, lenr) >= min_length[iplane]:
        scale = lenr / len0
        if (scale < min_scale) or (scale > max_scale):
            scale = None

    return(centre, xyzr, vec0, vecr, scale)
    
def test_scale_and_rotation():
    '''Test scale and rotation - track movement of single bright voxel.'''

    shape = (31, 31, 31)
    voxel_size = (1, 1, 3)
    origin = (-15, -15, -45)
    im0 = create_test_image(shape, voxel_size, origin, 'zeros')
    random.seed(1)

    # Intensity value for bright voxel.
    v_test = 1000

    # Set constraints to reduce changes of bright voxel
    # being rotated out of the image region.
    #
    # Fraction of image extents to be considered
    # when choosing original and rotated bright voxel.
    fraction = 0.4
    #
    # For displacement vectors from centre of rotation to rotated and
    # original bright, define minimum length (min_length),
    # and minimum and maximum length ratios (min_scale, max_scale).
    min_length = [2 * x for x in voxel_size]
    min_scale = 0.8
    max_scale = 1.2

    # Number of rotations to test.
    n_test = 20

    # Number of cases, in each projection, where single bright voxel
    # found in expected location after rotation.  Interpolation and rouding
    # errors may result in more than one bright pixel, or in bright voxel
    # being displaced.
    n_good = [0, 0, 0]

    # Minimum accepted fraction of cases where single bright voxel found
    # in expected location after rotation.
    min_fraction_good = 0.95

    for i in range(n_test):
        # Obtain coordinates for original and rotated point,
        # and for centre of rotation, ensuring that constraints are respected.
        point_ok = False
        while not point_ok:
            xyzc = get_random_voxel(im0)
            xyz0 = get_random_voxel(im0, fraction)
            xyz1 = get_random_voxel(im0, fraction)
            point_ok = True
            for j in range(3):
                centre, xyzr, vec0, vecr, scale = get_plane_data(
                        j, xyzc, xyz0, xyz1, min_length, min_scale, max_scale)
                if scale is None:
                    point_ok = False
                    break

        # Consider rotations in individual planes.
        for j in range(3):

            # Obtain in-plane coordinates and set bright voxel.
            centre, xyzr, vec0, vecr, scale = get_plane_data(
                    j, xyzc, xyz0, xyz1, min_length, min_scale, max_scale)
            im1 = Image(im0)
            ix0 = im1.pos_to_idx(xyz0[0], 'x')
            iy0 = im1.pos_to_idx(xyz0[1], 'y')
            iz0 = im1.pos_to_idx(xyz0[2], 'z')
            ix1 = im1.pos_to_idx(xyzr[0], 'x')
            iy1 = im1.pos_to_idx(xyzr[1], 'y')
            iz1 = im1.pos_to_idx(xyzr[2], 'z')
            im1.data[iy0][ix0][iz0] = v_test

            # Determine rotation angle as the difference between
            # the positive angles of rotation for rotated and original points.
            rotation = [0, 0, 0]
            k1 = j + 1 if j + 1 < 3 else 0
            k2 = k1 + 1 if k1 + 1 < 3 else 0
            theta0 = math.atan2(vec0[k2], vec0[k1]) % (2. * math.pi)
            thetar = math.atan2(vecr[k2], vecr[k1]) % (2. * math.pi)
            rotation[j] = math.degrees(thetar - theta0)

            # Check that bright voxel is in expected position before transform.
            iyy = np.where(im1.data == im1.data.max())[0][0]
            ixx = np.where(im1.data == im1.data.max())[1][0]
            izz = np.where(im1.data == im1.data.max())[2][0]
            assert np.sum(im1.data >= 0.9 * v_test) == 1
            assert im1.data[iy0][ix0][iz0] == v_test

            # Perform transform.
            im1.transform(centre=centre, scale=scale, rotation=rotation,
                    order=0)

            # Allow up to 3 bright voxels after rotation
            # (possible interpolation/rounding errors).
            assert np.sum(im1.data >= 0.9 * v_test) >=1
            assert np.sum(im1.data >= 0.9 * v_test) <=3
            if np.sum(im1.data >= 0.9* v_test) ==1:
                if im1.data[iy1][ix1][iz1] == v_test:
                    n_good[j] += 1
            # Allow bright voxel to be displaced by 1 in any direction
            # from expected position after rotation
            # (possible interpolation/rouding errors).
            iy2 = np.where(im1.data == im1.data.max())[0][0]
            ix2 = np.where(im1.data == im1.data.max())[1][0]
            iz2 = np.where(im1.data == im1.data.max())[2][0]
            assert abs(ix2 - ix1) < 2
            assert abs(iy2 - iy1) < 2
            assert abs(iz2 - iz1) < 2
    
    # Require single bright voxel in expected position after rotation
    # in at least some fraction of cases.
    for i in range(3):
        assert n_good[i] >= min_fraction_good

def test_crop_about_point():
    """Test cropping of image about point."""
    # Create test image.
    sim1 = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5))
    im1 = sim1.get_image()

    # Test null cropping.
    im2 = im1.clone()
    im2.crop_about_point()
    assert im1.get_extents() == im2.get_extents()

    # Crop image about point.
    point = (4, 8, 2)
    xyz_lims = [(-3, 5), (-6, -1), (0, 7)]
    im2.crop_about_point(point, *xyz_lims)

    # Check that extents of cropped image are as expected.
    extents = im2.get_extents()
    for i_ax, lims in enumerate(xyz_lims):
        assert tuple([lim + point[i_ax] for lim in lims]) == extents[i_ax]

def test_crop_by_amounts():
    """Test cropping of image by specified amounts."""
    # Create test image.
    sim1 = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5))
    im1 = sim1.get_image()

    # Test null cropping.
    im2 = im1.clone()
    im2.crop_by_amounts()
    assert im1.get_extents() == im2.get_extents()

    # Define amounts by which to crop.
    dxyz = [(1, 2), (3, 1), (4, 0)]
    im2.crop_by_amounts(*dxyz)

    for i_ax, reductions in enumerate(dxyz):
        dv1, dv2 = reductions
        assert im1.get_extents()[i_ax][0] + dv1 == im2.get_extents()[i_ax][0]
        assert im1.get_extents()[i_ax][1] - dv2 == im2.get_extents()[i_ax][1]

def test_crop_to_roi():
    """Text cropping to ROI and StructureSet."""
    # Create test image, featuring cuboid.
    sim = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5), noise_std=100)
    sim.add_cuboid((4, 2, 6), name="cuboid")
    ss = sim.get_structure_set()
    roi = sim.get_roi("cuboid")

    for obj in [roi, ss]:
        # Check that image extents, after cropping to an ROI or StructureSet,
        # are the same as the extents of the ROI or StructureSet.
        im = sim.get_image().clone()
        im.crop_to_roi(obj)
        assert np.all(obj.get_extents()
                      == [list(extent) for extent in im.get_extents()])

        # Check that image extents, after cropping about the centre
        # of an ROI or StructureSet, are as expected from the
        # centre coordinates of the ROI or StructureSet plus margins.
        im = sim.get_image().clone()
        crop_margins = (2, 1, (-3, 3))
        im.crop_to_roi(obj, crop_margins=crop_margins, crop_about_centre=True)
        centre = obj.get_centre()
        margins = checked_crop_limits(crop_margins)
        assert np.all(im.get_centre() == centre)
        extents = [(centre[idx] + margins[idx][0],
                    centre[idx] + margins[idx][1]) for idx in range(3)]
        assert np.all(extents == im.get_extents())

def test_crop_to_image():
    # Create test images.
    sim1 = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5))
    im1 = sim1.get_image()
    sim2 = SyntheticImage((12, 14, 12), origin=(5.5, 5.5, 5.5))
    im2 = sim2.get_image()

    # Perform cropping, after aligning image centres.
    im2.crop_to_image(im1, alignment="_centre_")

    # Check that images have the same lengths along each axis.
    for idx in range(3):
        assert im1.get_length(idx) == im2.get_length(idx)

    # Check that cropping has shifted origin by expected amount.
    assert im2.get_origin() == [6.5, 6.5, 6.5]

def test_dicom_dicom_slice():
    shape_single = (shape[0], shape[1], 1)
    im = create_test_image(shape_single, voxel_size, origin)
    dcm_file = "tmp/single_dcm"
    im.write(dcm_file)
    im_dcm = Image(f'{dcm_file}/1.dcm')
    assert im_dcm.get_data().shape == shape_single

@needs_mahotas
def test_create_foreground_mask():
    sim = SyntheticImage((100, 100, 40))
    sim.add_cube(side_length=5, name="cube", centre=(25, 60, 12), intensity=60)
    sim.add_sphere(radius=10, name="sphere", centre=(80, 20, 12), intensity=50)

    # Should detect sphere in first loop and cube in second
    for intensity in [50, 60]:
        threshold = intensity - 5
        mask1 = sim.get_foreground_mask(threshold=threshold).get_data()
        assert mask1.dtype == bool

        nx, ny, nz = sim.get_n_voxels()
        mask2 = np.zeros((ny, nx, nz), dtype=np.uint32)
        mask2[sim.get_data() == intensity] = True

        assert mask1.shape == mask2.shape
        assert mask1.min() == 0
        assert mask1.max() == 1
        assert mask1.min() == mask2.min()
        assert mask1.max() == mask2.max()
        assert mask1.sum() == mask2.sum()
    assert np.all(mask1 == mask2)

@needs_mahotas
def test_get_foreground_roi():
    """Test creation of ROI representing image foreground."""
    # Create synthetic image, featuring cube and sphere.
    sim = SyntheticImage((100, 100, 40))
    cube_length = 5
    sphere_radius = 10

    shapes = {
            "cube": [cube_length, (25, 60, 12), 60, cube_length**3],
            "sphere": [sphere_radius, (80, 20, 12), 50,
                       (4. / 3.) * math.pi * sphere_radius**3],
            }

    for name, values in shapes.items():
        length, centre, intensity, volume = values
        if "cube" == name:
            sim.add_cube(length, centre, intensity, name=name)
        elif "sphere" == name:
            sim.add_sphere(length, centre, intensity, name=name)

    # Should detect sphere as foreground in first loop and cube in second.
    for name, values in shapes.items():
        length, centre, intensity, volume = values
        threshold = intensity - 5
        roi = sim.get_foreground_roi(threshold=threshold, name=name)
        assert isinstance(roi, ROI)
        assert roi.name == name
        assert roi.get_volume() == pytest.approx(volume, rel=0.005)
        assert tuple(roi.get_centre()) == centre

    # For case where no name is specified, check that default is assigned.
    roi = sim.get_foreground_roi(threshold=threshold)
    assert roi.name == Defaults().foreground_name

@needs_mahotas
def test_mask_bbox():
    """Test calculation of mask bounding box."""

    # Create synthetic image featuring a sphere.
    sim = SyntheticImage((100, 100, 40))
    centre = (80, 20, 12)
    radius = 10
    sim.add_sphere(radius=radius, name="sphere", centre=centre, intensity=50)

    # Create foreground mask, and obtain it's bounding box.
    mask = sim.get_foreground_mask(threshold=45)
    bbox = get_mask_bbox(mask)

    # Test that the bounding box is consistent with the sphere dimensions,
    # allowing tolreance of +/-1.
    for idx1 in range(3):
        for idx2 in range(2):
            assert abs(abs(bbox[idx1][idx2] - centre[idx1]) - radius) <= 1

@needs_mahotas
def test_foreground_bbox():
    """Test calculation of foreground bounding box."""

    # Create synthetic image featuring a sphere.
    sim = SyntheticImage((100, 100, 40))
    centre = (80, 20, 12)
    radius = 10
    sim.add_sphere(radius=radius, name="sphere", centre=centre, intensity=50)

    # Obtain bounding box of foreground.
    bbox = sim.get_foreground_bbox(threshold=45)

    # Test that the bounding box is consistent with the sphere dimensions,
    # allowing tolreance of +/-1.
    for idx1 in range(3):
        for idx2 in range(2):
            assert abs(abs(bbox[idx1][idx2] - centre[idx1]) - radius) <= 1

@needs_mahotas
def test_foreground_bbox_centre_and_widths():
    """Test calculation of centre and widths for foreground bounding box."""

    # Create synthetic image featuring a sphere.
    sim = SyntheticImage((100, 100, 40))
    centre = (80, 20, 12)
    radius = 10
    sim.add_sphere(radius=radius, name="sphere", centre=centre, intensity=50)

    # Calculate centre and widths of foreground bounding box.
    bb_centre, bb_widths = sim.get_foreground_bbox_centre_and_widths(
            threshold=45)

    # Test that the centre and widths of the bounding box are consistent
    # with the sphere dimensions, allowing tolreance of +/-1.
    for idx in range(3):
        assert abs(bb_centre[idx] - centre[idx]) <= 1
        assert abs(bb_widths[idx] - 2 * radius) <= 1

@needs_mahotas
def test_create_intensity_mask():
    """
    Test creation of intensity masks.
    """
    # Create synthetic image (default intensity -1024),
    # featuring cube and sphere of higher intensities.
    sim = SyntheticImage((100, 100, 40))
    sim.add_cube(
            side_length=5, name="cube", centre=(25, 60, 12), intensity=60.4)
    sim.add_sphere(
            radius=10, name="sphere", centre=(80, 20, 12), intensity=50.6)

    # Create expected masks given different minimum and maximum intensities.
    nx, ny, nz = sim.get_n_voxels()
    ones = np.ones((ny, nx, nz), dtype=bool)
    masks = [
            ((None, None), ones),
            ((50.59, 50.61), ones * (sim.get_data() == 50.6)),
            ((60, None), ones * (sim.get_data() == 60.4)),
            ((None, 55),  1 - ones * (sim.get_data() == 60.4)),
            ]

    # Check intensitiy masks against expected masks.
    for intensities, mask2 in masks:
        vmin, vmax = intensities
        mask1 = sim.get_intensity_mask(vmin, vmax).get_data()
        mask1[mask1 > 0] = 1

        assert mask1.shape == mask2.shape
        # If there are no constraints on vmin and vmax,
        # all elements of the intensity mask should be True
        # (minimum and maximum both 1).
        # Otherwise, the minimum should be 0 and the maximum should be 1.
        if vmin is None and vmax is None:
            assert mask1.min() == 1
        else:
            assert mask1.min() == 0
        assert mask1.max() == 1
        assert mask1.min() == mask2.min()
        assert mask1.max() == mask2.max()
        assert mask1.sum() == mask2.sum()

        # Check that intensity mask is voxel for voxel identical
        # to the created mask.
        assert np.all(mask1 == mask2)

@needs_mahotas
def test_translation_to_align():
    """Test calculation of translation to align pair of images."""

    # Create synthetic images featuring a sphere.
    # shapes = [(100, 100, 40), (80, 80, 50)]
    # origins = [(-50, 40, -20), (0, -10, 30)]
    # centres = [(30, 80, 0), (60, 40, 50)]
    # radii = (10, 15)
    shapes = [(50, 50, 20), (40, 40, 25)]
    origins = [(-25, 20, -10), (0, -5, 15)]
    centres = [(15, 40, 0), (30, 20, 25)]
    radii = (5, 8)
    intensity=50
    sims = []
    for idx in range(len(shapes)):
        sims.append(SyntheticImage(shapes[idx], origin=origins[idx]))
        sims[idx].add_sphere(radius=radii[idx], name="sphere",
                centre=centres[idx], intensity=intensity)

    # Define expected translations for whole-image alignment:
    # (1) upper coordinates; (2) centre coordinates; (3) lower coordinates.
    translations = {
            1 : tuple([origins[1][idx] - origins[0][idx] for idx in range(3)]),
            2 : tuple([origins[1][idx] + 0.5 * shapes[1][idx]
                - origins[0][idx] - 0.5 * shapes[0][idx] for idx in range(3)]),
            3 : tuple([origins[1][idx] + shapes[1][idx]
                - origins[0][idx] - shapes[0][idx] for idx in range(3)]),
            }

    # Check translations for whole-image alignment.
    for alignment, translation in translations.items():
        # Perform alignment based on default alignment type.
        assert translation == get_translation_to_align(
                sims[0], sims[1], None, alignment, None)
        # Perform alignment based on dictionary of alignment types.
        alignments = {axis: alignment for axis in ("x", "y", "z")}
        assert translation == get_translation_to_align(
                sims[0], sims[1], alignments, None, None)

    # Define expected translations for foreground alignment:
    # (1) upper coordinates; (2) centre coordinates; (3) lower coordinates.
    translations = {
            1 : tuple([centres[1][idx] - radii[1] - centres[0][idx] + radii[0]
                for idx in range(3)]),
            2 : tuple([centres[1][idx] - centres[0][idx] for idx in range(3)]),
            3 : tuple([centres[1][idx] + radii[1] - centres[0][idx] - radii[0]
                for idx in range(3)]),
            }

    # Check translations for foreground alignment.
    for alignment, translation in translations.items():
        # Perform alignment based on default alignment type.
        assert translation == get_translation_to_align(
                sims[0], sims[1], None, alignment, intensity - 5)
        alignments = {axis: alignment for axis in ("x", "y", "z")}
        # Perform alignment based on dictionary of alignment types.
        assert translation == get_translation_to_align(
                sims[0], sims[1], alignments, None, intensity - 5)

def test_get_alignment_translation():
    """Test calculations of alignment translations."""
    # Create test images.
    im1 = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5)).get_image()
    im2 = SyntheticImage((12, 14, 12), origin=(5.5, 5.5, 5.5)).get_image()

    # Translations for centre alignment.
    dx_centre, dy_centre, dz_centre = im2.get_centre() - im1.get_centre()
    # Translation along z for top alignment.
    dz_top = im2.get_extents()[2][1] - im1.get_extents()[2][1]
    # Translation along z for bottom alignment.
    dz_bottom = im2.get_extents()[2][0] - im1.get_extents()[2][0]

    # Define alignments to test, and expected translations.
    alignments = [
            (None, None),
            ({}, (dx_centre, dy_centre, dz_centre)),
            ((dx_centre, dy_centre, dz_top), (dx_centre, dy_centre, dz_top)),
            ({"x": 2, "y": 2, "z": 2} , (dx_centre, dy_centre, dz_centre)),
            ({"z": 3} , (dx_centre, dy_centre, dz_top)),
            ({"z": 1} , (dx_centre, dy_centre, dz_bottom)),
            ("_centre_" , (dx_centre, dy_centre, dz_centre)),
            ("_top_" , (dx_centre, dy_centre, dz_top)),
            ("_bottom_" , (dx_centre, dy_centre, dz_bottom)),
            ]

    for alignment, translation in alignments:
        assert get_alignment_translation(im1, im2, alignment) == translation

def test_same_geometry():
    # Test identification of geometry differences
    shape1 = (50, 50, 10)
    voxel_size1 = (1, 1, 3)
    origin1 = (-62., -62., -10.)

    # Compare image with itself
    im1 = create_test_image(shape1, voxel_size1, origin1)
    assert im1.has_same_geometry(im1)

    # Accept differences within tolerance
    max_diff = 0.001
    origin2 = (-62.0008, -61.9999, -10.0003)
    im2 = create_test_image(shape1, voxel_size1, origin2)
    assert im1.has_same_geometry(im2, max_diff)

    # Reject differences outside tolerance
    origin3 = (-62., -62., -12.002)
    im3 = create_test_image(shape1, voxel_size1, origin3)
    assert not im1.has_same_geometry(im3, max_diff)


    # Compare with image that has equivalent geometry,
    # but different orientation.
    im4 = im1.astype("nii")
    assert not im1.has_same_geometry(im4)
    assert im1.has_same_geometry(im4, standardise=True)

def test_astype():
    # Test conversions between pydicom/DICOM representations
    # and nibabel/NIfTI representations
    assert np.all(im_nii.get_affine() == im_dcm.astype('nii').get_affine())
    assert np.all(im_dcm.get_affine() == im_nii.astype('dcm').get_affine())
    assert np.all(im_nii.get_data() == im_dcm.astype('nii').get_data())
    assert np.all(im_dcm.get_data() == im_nii.astype('dcm').get_data())
    assert im_dcm.astype('unknown') is None
    assert im_nii.astype('unknown') is None

def test_get_geometry():
    """Test consistency of geometry definitions."""
    # Check that null geometry returned for null inputs.
    assert get_geometry(None, None, None) == (None, None, None)

    # Check consistency of origin, voxel size, affine matrix.
    assert np.all(get_geometry(None, voxel_size, origin)[0] == affine)
    assert get_geometry(affine, None, None)[1] == list(voxel_size)
    assert get_geometry(affine, None, None)[2] == list(origin)

def test_get_affine():
    """Check non-standardised and standardised affine matrix."""
    # Check that initial image has affine matrix in standardised form.
    assert np.all(im.get_affine() == affine)

    # Create NIfTI-format image, which will have non-standardised orientation.
    im2 = im.astype("nii")
    # Check non-standardised and standardised affine matrix.
    assert not np.all(im2.get_affine() == affine)
    assert np.all(im2.get_affine(standardise=True) == affine)

def test_get_origin():
    """Check non-standardised and standardised origin."""
    # Check that initial image has standardised origin.
    assert im.get_origin() == list(origin)

    # Create NIfTI-format image, which will have non-standardised orientation.
    im2 = im.astype("nii")
    # Check non-standardised and standardised origin.
    assert im2.get_origin() != list(origin)
    assert im2.get_origin(standardise=True) == list(origin)

def test_get_voxel_size():
    """Check non-standardised and standardised voxel_size."""
    # Check that initial image has standardised voxel size.
    assert im.get_voxel_size() == list(voxel_size)

    # Create NIfTI-format image, which will have non-standardised orientation.
    im2 = im.astype("nii")
    # Check non-standardised and standardised voxel_size.
    assert im2.get_voxel_size() != list(voxel_size)
    assert [abs(dxyz) for dxyz in im2.get_voxel_size()] == list(voxel_size)
    assert im2.get_voxel_size(standardise=True) == list(voxel_size)

def test_apply_banding():
    # Test banding.
    im1 = Image(im)
    bands = {300: 100, 700: 500, 1e10: 900}
    im1.apply_banding(bands)
    values = sorted(list(bands.keys()))
    for i in range(len(values)):
        if i:
            v1 = values[i - 1]
        else:
            v1 = -1e10
        v2 = values[i]
        v_band = bands[v2]
        assert (((im.get_data() > v1) & (im.get_data() <= v2)).sum()
                == (im1.get_data() == v_band).sum())

def test_apply_selective_banding():
    # Test selective banding.

    # Define test image.
    im1 = Image(im)
    image_data = im1.get_data()
    print(image_data.min(), image_data.max())

    # Define banding, and apply to image copy.
    unbanded = [(600, 800)]
    bands = {100: (None, 300), 500: (300, 600), 900: (800, None)}
    im2 = im1.clone()
    im2.apply_selective_banding(bands)
    banded_data = im2.get_data()

    # Check that band values are correctly assigned.
    for v_band, values in sorted(bands.items()):
        v1 = values[0] if values[0] is not None else image_data.min() - 1
        v2 = values[1] if values[1] is not None else image_data.max() + 1
        assert np.all(banded_data[(image_data > v1) & (image_data <= v2)]
                == v_band)

    # Check that values in unbanded range(s) are unchanged.
    for v1, v2 in unbanded:
        assert np.all(banded_data[(image_data > v1) & (image_data <= v2)]
                == image_data[(image_data > v1) & (image_data <= v2)])

def test_addition():
    # Define test image.
    shape = (50, 50, 10)
    voxel_size = (1, 1, 3)
    origin = (-62., -62., -10.)
    im1 = create_test_image(shape, voxel_size, origin)

    # Test addition.
    im2 = im1 + im1
    
    assert im2.get_n_voxels() == im1.get_n_voxels()
    assert im2.get_voxel_size() == im1.get_voxel_size()
    assert im2.get_origin() == im1.get_origin()
    assert np.all(im2.get_data() == 2 * im1.get_data())

    # Test in-place addition.
    im3 = im1.clone()
    im3 += im1
    
    assert im3.get_n_voxels() == im1.get_n_voxels()
    assert im3.get_voxel_size() == im1.get_voxel_size()
    assert im3.get_origin() == im1.get_origin()
    assert np.all(im3.get_data() == 2 * im1.get_data())

def test_subtraction():
    # Define test image.
    shape = (50, 50, 10)
    voxel_size = (1, 1, 3)
    origin = (-62., -62., -10.)
    im1 = create_test_image(shape, voxel_size, origin)

    # Test subtraction.
    im2 = im1 - im1
    
    assert im2.get_n_voxels() == im1.get_n_voxels()
    assert im2.get_voxel_size() == im1.get_voxel_size()
    assert im2.get_origin() == im1.get_origin()
    assert np.all(im2.get_data() == np.zeros(shape))

    # Test in-place subtraction.
    im3 = im1.clone()
    im3 -= im1
    
    assert im3.get_n_voxels() == im1.get_n_voxels()
    assert im3.get_voxel_size() == im1.get_voxel_size()
    assert im3.get_origin() == im1.get_origin()
    assert np.all(im3.get_data() == np.zeros(shape))

def test_unary_opeartions():
    # Define test image.
    shape = (50, 50, 10)
    voxel_size = (1, 1, 3)
    origin = (-62., -62., -10.)
    im1 = create_test_image(shape, voxel_size, origin)

    # Test unary positive.
    im2 = +im1
    
    assert im2.get_n_voxels() == im1.get_n_voxels()
    assert im2.get_voxel_size() == im1.get_voxel_size()
    assert im2.get_origin() == im1.get_origin()
    assert np.all(im2.get_data() == im1.get_data())

    # Test unary negative.
    im3 = -im1
    
    assert im3.get_n_voxels() == im1.get_n_voxels()
    assert im3.get_voxel_size() == im1.get_voxel_size()
    assert im3.get_origin() == im1.get_origin()
    assert np.all(im3.get_data() == -im1.get_data())

def test_multiplication_by_scalar():
    # Define test image.
    shape = (50, 50, 10)
    voxel_size = (1, 1, 3)
    origin = (-62., -62., -10.)
    im1 = create_test_image(shape, voxel_size, origin)

    # Test multiplication by scalar.
    scalar = 5.2
    for i in [0, 1, 2]:
        # Test left multiplication.
        if 0 == i:
            im2 =  scalar * im1
        # Test right multiplication.
        elif 1 == i:
            im2 =  im1 * scalar
        # Test in-place multiplication.
        elif 2 == i:
            im2 = im1.clone()
            im2 *= scalar
        
        assert im2.get_n_voxels() == im1.get_n_voxels()
        assert im2.get_voxel_size() == im1.get_voxel_size()
        assert im2.get_origin() == im1.get_origin()
        assert np.all(im2.get_data() == scalar * im1.get_data())

def test_division_by_scalar():
    # Define test image.
    shape = (50, 50, 10)
    voxel_size = (1, 1, 3)
    origin = (-62., -62., -10.)
    im1 = create_test_image(shape, voxel_size, origin)

    # Test division by scalar.
    scalar = 5.2
    for i in [0, 1]:
        # Test standard division.
        if 0 == i:
            im2 =  im1 / scalar
        # Test in-place division.
        elif 1 == i:
            im2 = im1.clone()
            im2 /= scalar
        
        assert im2.get_n_voxels() == im1.get_n_voxels()
        assert im2.get_voxel_size() == im1.get_voxel_size()
        assert im2.get_origin() == im1.get_origin()
        assert np.all(im2.get_data() == im1.get_data() / scalar) 

def test_pathlib_path():
    # Test passing of pathlib.Path.
    im_tmp = Image(Path(dcm_file))
    assert im_tmp.path == fullpath(dcm_file)

def test_get_size():
    # Test calculation of image size.
    for idx in range(3):
        assert (im.get_n_voxels()[idx] * im.get_voxel_size()[idx]
                == im.get_size()[idx])

def test_get_volume():
    # Test calculation of image volume.
    assert np.prod(shape) == im.get_volume("voxels")
    assert np.prod(shape) * np.prod(voxel_size) == im.get_volume("mm")
    assert (np.prod(shape) * np.prod(voxel_size) / 1000) == im.get_volume("ml")

def test_get_centroid():
    """Test calculation of centroid of above-threshold voxels."""

    # Create synthetic image, featuring sphere.
    # The maximum intensity for the image is shared by the voxels of the sphere,
    # so the calculated centroid should correspond to the sphere centroid.
    #shape = [100, 60, 80]
    #origin = [-40, 40, -100]
    #idx_centre = [80, 15, 35]
    shape = [50, 30, 40]
    origin = [-20, 20, -50]
    idx_centre = [40, 8, 17]
    sl_centre = [idx_centre[0] + 1, idx_centre[1] + 1, shape[2] - idx_centre[2]]
    pos_centre = [origin[idx] + idx_centre[idx] for idx in range(3)]
    sim = SyntheticImage(shape, origin=origin)
    sim.add_sphere(radius=10, name="sphere", centre=pos_centre, intensity=50)

    # Check centroid for each view and coordinate system.
    fraction = 0.9
    for idx, view in enumerate(["y-z", "x-z", "x-y"]):
        assert sim.get_centroid_idx(view, fraction) ==  idx_centre[idx]
        assert sim.get_centroid_slice(view, fraction) ==  sl_centre[idx]
        assert sim.get_centroid_pos(view, fraction) ==  pos_centre[idx]

def test_get_translation_to_align_image_rois():
    """Test calculation of translation to align image ROIs."""

    # Create synthetic images featuring a sphere.
    shapes = [(100, 100, 40), (80, 80, 50)]
    origins = [(-50, 40, -20), (0, -10, 30)]
    centres = [(30, 80, 0), (60, 40, 50)]
    radii = (10, 15)
    intensity=50
    ims = []
    for idx in range(len(shapes)):
        sim = SyntheticImage(shapes[idx], origin=origins[idx])
        sim.add_sphere(radius=radii[idx], name="sphere",
                centre=centres[idx], intensity=intensity)
        ss = sim.get_structure_set()
        ims.append(Image(sim))
        ss.set_image(ims[idx])

    # Initialise random-number seed.
    np.random.seed(1)

    # Check that translations for aligning sphere centres are as expected.
    for z1, z2 in [(None, None), np.random.uniform(0, 1, 2)]:
        dz1 = 0.5 if z1 is None else z1 - 0.5
        dz2 = 0.5 if z2 is None else z2 - 0.5
        t1 = [centres[1][idx] - centres[0][idx] for idx in range(3)]
        t2 = ims[0].get_translation_to_align_image_rois(ims[1],
                "sphere", "sphere", z1, z2)
        assert t1[0] == t2[0]
        assert t1[1] == t2[1]
        assert ((t1[2] + (dz2 * radii[1] - dz1 * radii[0]))
                == pytest.approx(t2[2], 1e6))

def test_dcm_single_file():
    """
    Check that an Image can be loaded given a single file from a directory.
    """
    # Create a list of all files in directory, then randomly choose one
    # to pass to the Image constructor.
    paths = list(Path(dcm_file).glob("*.dcm"))
    assert len(paths) > 0
    path = random.choice(paths)
    im = Image(path)

    # Check that the shape of the loaded image is as expected.
    assert im.get_n_voxels()[2] == len(paths)
    assert im.get_n_voxels() == im_dcm.get_n_voxels()
    assert str(Path(im.path).parent) == im_dcm.path

def test_dcm_wildcards():
    """
    Check that an Image can be loaded from a path including wildcards.
    """
    im = Image("tmp/t*_dcm/*.d?m")
    assert im.get_n_voxels() == im_dcm.get_n_voxels()
    assert im.path == im_dcm.path

def test_dcm_list():
    """
    Check that an Image can be loaded from a list of file paths.
    """
    # Create a list of all files in directory, then randomly choose one
    # to pass to the Image constructor.
    paths = list(Path(dcm_file).glob("*.dcm"))
    assert len(paths) > 0
    im = Image(paths)

    # Check that the shape of the loaded image is as expected.
    assert im.get_n_voxels()[2] == len(paths)
    assert im.get_n_voxels() == im_dcm.get_n_voxels()
    assert im.path == im_dcm.path

    n_path = 6
    selected_paths = set()
    while len(selected_paths) < n_path:
        selected_paths.add(random.choice(paths))
    im = Image(selected_paths)

    # Check that the shape of the loaded image is as expected.
    assert im.get_n_voxels()[2] == len(selected_paths)
    assert im.get_n_voxels()[0: 2] == im_dcm.get_n_voxels()[0: 2]
    assert im.path == im_dcm.path

def test_checked_crop_limits():
    """Test checking of crop limits."""

    null = None
    value = 4
    limits1 = ((-2, 7), (8, 37), (-20, 29))
    limits2 = ((-2, 7), 4, (-20, 29))
    limits3 = ((-2, 7), (-4, 4), (-20, 29))
    limits4 = ((-2, 7), -4, (-20, 29))
    limits5 = ((-2, 7), (4, -4), (-20, 29))

    # Define tuple pairing inputs and expected outputs.
    crop_limits = (
            (null, 3 * (null,)),
            (value, tuple((-value, value) for idx in range(3))),
            (-value, tuple((value, -value) for idx in range(3))),
            (limits1, limits1),
            (limits2, limits3),
            (limits4, limits5),
            )

    # Check that behaviour is as expected.
    for in_value, out_value in crop_limits:
        assert checked_crop_limits(in_value) == out_value

def test_get_mutual_information():
    """Test calculation of mutual information and variants."""

    small_number = 1.e-6

    # Check mutual information for some simple cases.
    im1 = Image(np.array([[1, 1],[1, 1]]))
    im2 = Image(np.array([[1, 2],[3, 4]]))
    # Intensities of image 1 give no information on intensities of image 2:
    # MI = [(1) * log((1)/(1))] = 0; NMI = 1; IQR = 0; RD = 1..
    variants = [("mi", 0), ("nmi", 1), ("iqr", 0), ("rajski", 1)]
    for variant, value in variants:
        for base in [None, 2, 10]:
            assert (im1.get_mutual_information(im2, base=base, variant=variant)
                    == value)

    im1 = Image(np.array([[1, 2],[3, 4]]))
    im2 = Image(np.array([[5, 6],[7, 8]]))
    # Intensities of image 1 linearly related to intensities of image 2:
    # MI = 4 * [(1/4) * log((1/4)/(1/16))] = log(4); NMI = 2; IQR = 1; RD = 0.
    variants = [("mi", None), ("nmi", 2), ("iqr", 1), ("rajski", 0)]
    for variant, value in variants:
        for base in [None, 2, 10]:
            test_value = value if value is not None else pytest.approx(
                    math.log(4, (base or math.e)), small_number)
            assert (im1.get_mutual_information(im2, base=base, variant=variant)
                    == test_value)

def test_rescale_images():
    """Test rescaling of image greyscale values."""
    
    # Create test image, and obtain greyscale characteristics.
    im = create_test_image(shape, voxel_size, origin)
    im.data = im.data - (0.5 * im.get_max())
    u_min = im.get_min(force=True)
    u_max = im.get_max(force=True)
    du = u_max - u_min
    u_sum = im.data.sum()

    # Rescale image.
    v_min = 0.
    v_max = 100.
    constant = 50.
    dv = v_max - v_min
    im2 = rescale_images(im, v_min, v_max, constant, clone=False)[0]

    # Check greyscale characteristics of rescaled image.
    assert im2 is im
    assert im2.get_min(force=True) == v_min
    assert im2.get_max(force=True) == v_max
    assert ((u_min + ((im2.data - v_min) * (du / dv))).sum()
            == pytest.approx(u_sum, rel=0.001))

    # Check greyscale value after rescaling,
    # when initial greyscale values are all the same.
    v_fill = 10
    im.data.fill(v_fill)
    im2 = rescale_images((im,), v_min, v_max, constant, clone=True)[0]
    assert im2 is not im
    assert np.all(im2.data == constant)
    assert np.all(im2.data != im.data)

    # Check that original image is returned
    # when lower or upper bound for rescaling is None,
    for v_min2, v_max2 in [(None, v_max), (v_min, None)]:
        im2 = rescale_images([im], v_min2, v_max2, constant, clone=True)[0]
        assert im2 is im
        assert np.all(im2.data == v_fill)

def test_get_relative_structural_content():
    """Test calculation of relative structural content."""
    # For image compared with itself,
    # check that relative structural content is 1.
    im1 = create_test_image(shape, voxel_size, origin)
    assert im1.get_relative_structural_content(im1) == 1

    # For image of zeros, compared with another image,
    # check that relative structural content is 0.
    im2 = create_test_image(shape, voxel_size, origin, "zeros")
    assert im2.get_relative_structural_content(im1) == 0

def test_get_fidelity():
    """Test calculation of fidelity."""
    # For image compared with itself, check that fidelity is 1.
    im1 = create_test_image(shape, voxel_size, origin)
    assert im1.get_fidelity(im1) == 1

    # For image of zeros, compared with another image,
    # check that fidelity is 0.
    im2 = create_test_image(shape, voxel_size, origin, "zeros")
    assert im2.get_fidelity(im1) == 0

def test_get_correlation_quality():
    """Test calculation of correlation quality."""
    # For image compared with itself, check that correlation_quality is 1.
    im1 = create_test_image(shape, voxel_size, origin)
    assert im1.get_correlation_quality(im1) == 1

    # For image of zeros, compared with another image,
    # check that correlation quality is 0.
    im2 = create_test_image(shape, voxel_size, origin, "zeros")
    assert im2.get_correlation_quality(im1) == 0

def test_get_quality():
    """Test quality of image with respect to another image."""
    # Create test images, and perform comparison.
    im1 = create_test_image(shape, voxel_size, origin)
    im2 = create_test_image(shape, voxel_size, origin)

    metrics = {
            "relative_structural_content" : 1,
            "fidelity" : 0.5,
            "correlation_quality" : 0.75
            }

    # Check that only None values returned for null or invalid metrics.
    for metric in [[], "", "unknown_metric"]:
        scores = im1.get_quality(im2, metric)
        assert isinstance(scores, dict)
        assert len(scores) == len(metrics)
        assert all([score is None for score in scores.values()])

    # Check that scores are as expected for images with random intensity values.
    scores = im1.get_quality(im2, list(metrics))
    assert isinstance(scores, dict)
    assert len(scores) == len(metrics)
    for metric, score in metrics.items():
        assert scores[metric] == pytest.approx(score, abs=0.02)

def test_get_image_comparison_metrics():
    """Check that get_image_comparison_metrics() returns a list of strings."""
    metrics = get_image_comparison_metrics()
    assert isinstance(metrics, list)
    for metric in metrics:
        assert isinstance(metric, str)

def test_get_comparison():
    """Test evaluation of image-comparison metrics."""
    # Create test images, and perform comparison.
    im1 = create_test_image(shape, voxel_size, origin)
    im2 = create_test_image(shape, voxel_size, origin)
    metrics = get_image_comparison_metrics()
    comparison = im1.get_comparison(im2, metrics=metrics)

    # Check that resulting DataFrame is as expected.
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape[0] == 1
    assert comparison.shape[1] == len(metrics)
    assert list(comparison.columns) == metrics

    # Check that exception is raised for unknown metric.
    with pytest.raises(RuntimeError) as error_info:
        comparison = im1.get_comparison(im2, metrics=["unknown_metric"])
    assert "Metric unknown_metric not recognised" in str(error_info.value)
