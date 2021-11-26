'''Test prototype image class.'''

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
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Create a test nifti file
nii_file = 'tmp/tmp.nii'
im.write(nii_file)
im_nii = Image(nii_file)

# Create a test dicom file
dcm_file = 'tmp/tmp_dcm'
im.write(dcm_file)
im_dcm = Image(dcm_file)


############################
# Test reading and writing #
############################

def test_array_to_npy():
    '''Check numpy array can be saved to a .npy file and read in correctly.'''

    outname = 'tmp/tmp.npy'
    im.write(outname)
    im_npy = Image(outname, affine=im.affine)
    assert np.all(im_npy.data == im.data)
    assert np.all(im_npy.affine == im.affine)
    assert os.path.exists(outname.replace('npy', 'txt'))

def test_array_to_nifti_npy():
    '''Check numpy array is correctly saved in nifti-style.'''

    outname = 'tmp/tmp_nii.npy'
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

def test_nifti_to_npy():
    '''Check that a nifti file is correctly written to a numpy file.'''

    # Write to numpy file
    npy = 'tmp/tmp2.npy'
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
