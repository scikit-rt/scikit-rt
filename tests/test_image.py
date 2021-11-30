"""Test prototype image class."""

import os
import numpy as np
import shutil
import pydicom

from skrt.core import File
from skrt.image import Image


# Create fake data
data = (np.random.rand(40, 50, 20) * 1000).astype(np.uint16)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)

def create_test_image():
    im = Image(data, voxel_size=voxel_size, origin=origin)
    return im

im = create_test_image()

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
    im_dcm.write(dcm)
    im_dcm2 = Image(dcm)

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

    im = create_test_image()
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
    im = Image()
    assert(type(im).__name__ == "Image")
    assert(im.affine is None)
    assert(im.data is None)
    assert(im.date == "")
    assert(im.downsampling is None)
    assert(im.files == [])
    assert(not im.nifti_array)
    assert(im.origin == (0, 0, 0))
    assert(im.path == "")
    assert(im.source == "")
    assert(im.source_type is None)
    assert(im.subdir == "")
    assert(im.time == "")
    assert(im.timestamp == "")
    assert(im.title is None)
    assert(im.voxel_size == (1, 1, 1))
