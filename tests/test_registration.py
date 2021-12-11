"""Test the skrt.registration.Registration class."""
import numpy as np
import os
import shutil
import pytest

from skrt.simulation import SyntheticImage
from skrt.reg2 import Registration


# Directory to store test registration data
reg_dir = "tmp/reg"
if os.path.exists(reg_dir):
    shutil.rmtree(reg_dir)
    
# Test images
sim1 = SyntheticImage((10, 12, 8))
sim1.add_cube(2, centre=(4, 4, 4))
im1 = sim1.get_image()
sim2 = SyntheticImage((11, 11, 11))
sim2.add_cube(2, centre=(6, 6, 6))
im2 = sim2.get_image()


def test_setup_with_images():
    """Test creation of a new Registration object with images."""

    reg = Registration(reg_dir, im1, im2, overwrite=True)
    assert os.path.exists(reg.fixed_path)
    assert os.path.exists(reg.moving_path)
    assert np.all(reg.fixed_image.get_standardised_data() == im1.get_standardised_data())
    assert np.all(reg.moving_image.get_standardised_data() == im2.get_standardised_data())

def test_init_with_pfiles():
    """Test creation of a new Registration object with images and parameter
    files."""

    pfiles =  ["pfiles/MI_Translation.txt", "pfiles/MSD_Rigid.txt"]
    reg = Registration(reg_dir, im1, im2, pfiles=pfiles, overwrite=True)
    assert len(reg.steps) == len(pfiles)
    assert os.path.isfile(reg.steps_file)
    assert len(open(reg.steps_file).readlines()) == len(pfiles)
    for outdir in reg.outdirs.values():
        assert os.path.isdir(outdir)
    for pfile in reg.pfiles.values():
        assert os.path.isfile(pfile)

def test_load_existing():
    """Test loading an existing registration object that has a fixed and 
    moving image and registration steps."""

    reg = Registration(reg_dir)
    
    # Check fixed and moving images were loaded
    assert os.path.exists(reg.fixed_path)
    assert os.path.exists(reg.moving_path)
    assert np.all(reg.fixed_image.get_standardised_data() == im1.get_standardised_data())
    assert np.all(reg.moving_image.get_standardised_data() == im2.get_standardised_data())

    # Check registration steps were loaded
    assert len(reg.steps) == 2
    for outdir in reg.outdirs.values():
        assert os.path.isdir(outdir)
    for pfile in reg.pfiles.values():
        assert os.path.isfile(pfile)

def test_load_overwrite():
    """Test loading with overwrite=True; check that this removes existing 
    images and registration steps."""

    reg = Registration(reg_dir, overwrite=True)
    assert not os.path.exists(reg.fixed_path)
    assert not os.path.exists(reg.moving_path)
    assert len(reg.steps) == 0
    assert len(reg.pfiles) == 0
    assert len(reg.outdirs) == 0

def test_add_pfiles():
    """Test adding a registration step to an existing Registration object."""

    reg = Registration(reg_dir, im1, im2, overwrite=True)
    assert len(reg.steps) == 0
    reg.add_pfile("pfiles/MI_Translation.txt")
    assert len(reg.steps) == 1
    assert len(reg.pfiles) == 1
    assert len(reg.outdirs) == 1

def test_clear_registrations():
    """Test removing all registration steps and their outputs."""

    reg = Registration(reg_dir)
    assert len(reg.steps)
    old_outdirs = reg.outdirs
    assert len(old_outdirs)
    reg.clear()
    assert len(reg.steps) == 0
    assert len(reg.pfiles) == 0
    assert len(reg.outdirs) == 0
    for old in old_outdirs.values():
        assert not os.path.exists(old)

def test_adjust_parameters():
    """Test adjustment of an elastix parameter file."""
    
    pfile = "pfiles/MI_Translation.txt"
    init_text = open(pfile).read()
    assert "(DefaultPixelValue 0)" in init_text
    from skrt.reg2 import adjust_parameters
    new_file = "tmp/tmp_pfile.txt"
    adjust_parameters(pfile, new_file, {"DefaultPixelValue": 10})
    assert os.path.exists(new_file)
    text = open(new_file).read()
    assert "(DefaultPixelValue 10)" in text
    os.remove(new_file)

def test_run_registration():
    """Test running of a multi-step registration."""

    pfiles = ["pfiles/MI_Translation.txt", "pfiles/MSD_Rigid.txt"]
    reg = Registration(
        reg_dir, 
        overwrite=True, 
        fixed=im1, 
        moving=im2,
        pfiles=pfiles
    )
    assert len(reg.tfiles) == 0

    reg.register()
    assert len(reg.tfiles) == len(pfiles)
    assert len(reg.transformed_images) == len(pfiles)
    for step in reg.steps:
        assert reg.already_performed(step)
    assert reg.get_transformed_image().get_standardised_data().shape \
            == im1.get_standardised_data().shape
    reg.view_result(show=False)

def test_load_completed_registration():
    """Test loading existing registration results."""

    reg2 = Registration(reg_dir)
    assert len(reg2.tfiles) == 2
    assert len(reg2.transformed_images) == 2
    for step in reg2.steps:
        assert reg2.already_performed(step)

def test_transform_image():
    """Test transforming an Image object using the result of a registration."""

    reg = Registration(reg_dir)
    sim3 = SyntheticImage(sim2.shape)
    sim4 = reg.transform_image(sim3)
    assert sim4.get_standardised_data().shape \
            != sim3.get_standardised_data().shape
    assert sim4.get_standardised_data().shape \
            == sim1.get_standardised_data().shape

    # Test transforming with an earlier step
    sim5 = reg.transform_image(sim3, step=0)
    assert not(np.all(sim5.get_standardised_data() 
                      == sim4.get_standardised_data()))
    assert sim5.get_standardised_data().shape \
            == sim1.get_standardised_data().shape

def test_transform_roi():
    """Test transforming an ROI using the result of a registration."""

    reg = Registration(reg_dir)
    sim3 = SyntheticImage(sim2.shape)
    sim3.add_cube(4, name="cube")
    roi = sim3.get_roi("cube")
    roi2 = reg.transform_roi(roi)
    assert roi.get_contours() != roi2.get_contours()
    assert roi.get_mask().get_standardised_data().shape \
            == sim1.get_standardised_data().shape
