"""Test Image class."""

import math
import os
import random
import pytest
import numpy as np
import shutil
import pydicom

from skrt.core import File
from skrt.image import Image
from skrt.simulation import SyntheticImage


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
im.write(dcm_file)
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
    series_description = 'Image study'
    header_extras = {'SeriesDescription' : series_description}
    im_dcm.write(dcm, header_extras=header_extras)
    im_dcm2 = Image(dcm)
    assert im_dcm2.get_dicom_dataset().SeriesDescription == series_description

    assert im_dcm.data.shape == im_dcm2.data.shape
    assert np.all(im_dcm.affine == im_dcm2.affine)
    assert np.all(im_dcm.data == im_dcm2.data)

def test_nifti_to_dcm():
    """Check that a nifti file can be written to dicom using a fresh header."""

    # Write nifti to dicom
    dcm = "tmp/tmp_nii2dcm"
    im_nii.write(dcm)
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
    shape_2 = (50, 50, 50)
    voxel_size_2 = (1, 2, 3)
    origin_2 = (-150, -120, -80)

    im1 = Image(np.random.rand(*shape_1), voxel_size=voxel_size_1,
            origin=origin_1)
    im1.set_geometry()
    im2 = Image(np.random.rand(*shape_2), voxel_size=voxel_size_2,
            origin=origin_2)
    im2.set_geometry()

    # Resize im1
    for image0, image2, image_size, origin, voxel_size in [
            (im1, im2, shape_2, origin_2, voxel_size_2),
            (im2, im1, shape_1, origin_1, voxel_size_1)]:
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

def test_crop_to_roi():
    sim = SyntheticImage((10, 12, 10), origin=(0.5, 0.5, 0.5), noise_std=100)
    sim.add_cuboid((4, 2, 6), name="cuboid")
    roi = sim.get_roi("cuboid")
    im = sim.get_image()
    im.crop_to_roi(roi)
    for i in range(2):
        assert set(roi.get_extents()[i]) == set(im.image_extent[i])

def test_dicom_dicom_slice():
    shape_single = (shape[0], shape[1], 1)
    im = create_test_image(shape_single, voxel_size, origin)
    dcm_file = "tmp/single_dcm"
    im.write(dcm_file)
    im_dcm = Image(f'{dcm_file}/1.dcm')
    assert im_dcm.get_data().shape == shape_single

def test_create_foreground_mask():
    sim = SyntheticImage((100, 100, 40))
    sim.add_cube(side_length=5, name="cube", centre=(25, 60, 12), intensity=60)
    sim.add_sphere(radius=10, name="sphere", centre=(80, 20, 12), intensity=50)

    # Should detect sphere in first loop and cube in second
    for intensity in [50, 60]:
        threshold = intensity - 5
        mask1 = sim.create_foreground_mask(threshold=threshold).get_data()
        mask1 = mask1.astype(np.uint32)
        mask1[mask1 > 0] =1

        nx, ny, nz = sim.get_n_voxels()
        mask2 = np.zeros((ny, nx, nz), dtype=np.uint32)
        mask2[sim.get_data() == intensity] = 1

        assert mask1.shape == mask2.shape
        assert mask1.min() == 0
        assert mask1.max() == 1
        assert mask1.min() == mask2.min()
        assert mask1.max() == mask2.max()
        assert mask1.sum() == mask2.sum()
    assert np.all(mask1 == mask2)
