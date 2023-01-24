"""Test the skrt.registration.Registration class."""
import numpy as np
import os
from pathlib import Path
import shutil
import pytest
import subprocess

from skrt import Image
from skrt.simulation import SyntheticImage
from skrt.registration import (Registration, get_default_pfiles,
        get_default_pfiles_dir, read_parameters)

from test_structs import compare_rois

# Define directory containing default parameter files.
pfiles_dir = get_default_pfiles_dir()

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
try:
    import mahotas
    has_mahotas = True
except ModuleNotFoundError:
    has_mahotas = False

if has_mahotas:
    mask1 = im1.get_foreground_mask(threshold=-10)
    mask2 = im2.get_foreground_mask(threshold=-10)

# Check for elastix executable
try:
    subprocess.check_output("elastix")
    has_elastix = True
except:
    has_elastix = False

# Decorator for tests requiring elastix functionality
def needs_elastix(func):
    def wrapper():
        if not has_elastix:
            return
        else:
            func()
    return wrapper

# Decorator for tests requiring mahotas
def needs_mahotas(func):
    def wrapper():
        if not has_mahotas:
            return
        else:
            func()
    return wrapper

def test_setup_with_images():
    """Test creation of a new Registration object with images."""

    reg = Registration(reg_dir, im1, im2, overwrite=True)
    assert os.path.exists(reg.fixed_path)
    assert os.path.exists(reg.moving_path)
    assert np.all(reg.fixed_image.get_standardised_data() == im1.get_standardised_data())
    assert np.all(reg.moving_image.get_standardised_data() == im2.get_standardised_data())

@needs_mahotas
def test_setup_with_masks():
    """Test creation of a new Registration object with images and masks."""

    reg = Registration(reg_dir, im1, im2, mask1, mask2, overwrite=True)
    assert os.path.exists(reg.fixed_path)
    assert os.path.exists(reg.moving_path)
    assert os.path.exists(reg.fixed_mask_path)
    assert os.path.exists(reg.moving_mask_path)

    array1 = (Image(reg.fixed_path).get_data() > -10)
    array2 = (Image(reg.moving_path).get_data() > -10)
    array3 = (Image(reg.fixed_mask_path).get_data() > 0)
    array4 = (Image(reg.moving_mask_path).get_data() > 0)
    assert array1.sum() == array3.sum()
    assert np.all(array1 == array3)
    assert array2.sum() == array4.sum()
    assert np.all(array2 == array4)

def test_init_with_pfiles():
    """Test creation of a new Registration object with images and parameter
    files."""

    pfiles = get_default_pfiles(basename_only=False)
    assert len(pfiles) > 0
    reg = Registration(reg_dir, im1, im2, pfiles=pfiles, overwrite=True)
    assert len(reg.steps) == len(pfiles)
    assert os.path.isfile(reg.steps_file)
    assert len(open(reg.steps_file).readlines()) == len(pfiles)
    for outdir in reg.outdirs.values():
        assert os.path.isdir(outdir)
    for pfile in reg.pfiles.values():
        assert os.path.isfile(pfile)

def test_init_pfiles_custom_names():
    """Test loading of parameter files with a dict containing custom names."""

    pfiles = {
        "translation": pfiles_dir / "MI_Translation.txt", 
        "affine": pfiles_dir / "MI_Affine.txt"
    }
    reg = Registration(reg_dir, im1, im2, pfiles=pfiles, overwrite=True)
    assert reg.steps == list(pfiles.keys())
    for step in pfiles:
        assert os.path.isdir(os.path.join(reg.path, step))

def test_load_existing():
    """Test loading an existing registration object that has a fixed and 
    moving image and registration steps."""

    reg = Registration(reg_dir)
    
    # Check fixed and moving images were loaded
    assert os.path.exists(reg.fixed_path)
    assert os.path.exists(reg.moving_path)
    assert np.all(reg.fixed_image.get_standardised_data() 
                  == im1.get_standardised_data())
    assert np.all(reg.moving_image.get_standardised_data() 
                  == im2.get_standardised_data())

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

def test_add_files():
    """Test adding a registration step to an existing Registration object."""

    reg = Registration(reg_dir, im1, im2, overwrite=True)
    assert len(reg.steps) == 0
    reg.add_file(pfiles_dir / "MI_Translation.txt")
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

@needs_elastix
def test_run_registration():
    """Test running of a multi-step registration."""

    # Define masks that keep visible the entire image volumes.
    im1_mask = Image(im1)
    im1_mask.data = np.ones(im1.get_data().shape)
    im2_mask = Image(im2)
    im2_mask.data = np.ones(im2.get_data().shape)

    pfiles = [pfiles_dir / "MI_Translation.txt", pfiles_dir / "MI_Affine.txt"]
    reg = Registration(
        reg_dir, 
        overwrite=True, 
        fixed=im1, 
        moving=im2,
        fixed_mask=im1_mask,
        moving_mask=im2_mask,
        pfiles=pfiles
    )
    assert len(reg.tfiles) == 0

    reg.register()
    assert len(reg.tfiles) == len(pfiles)
    assert len(reg.transformed_images) == len(pfiles)
    for step in reg.steps:
        assert reg.is_registered(step)
    assert reg.get_transformed_image().get_standardised_data().shape \
            == im1.get_standardised_data().shape
    reg.view_result(show=False)

@needs_elastix
def test_load_completed_registration():
    """Test loading existing registration results."""

    reg2 = Registration(reg_dir)
    assert len(reg2.tfiles) == 2
    assert len(reg2.transformed_images) == 2
    for step in reg2.steps:
        assert reg2.is_registered(step)

@needs_elastix
def test_transform_image():
    """Test transforming an Image object using the result of a registration."""

    reg = Registration(reg_dir)
    sim3 = SyntheticImage(sim2.shape)
    sim4 = reg.transform(sim3)
    assert sim4.get_standardised_data().shape \
            != sim3.get_standardised_data().shape
    assert sim4.get_standardised_data().shape \
            == sim1.get_standardised_data().shape

    # Test transforming with an earlier step
    sim5 = reg.transform(sim3, step=0)
    assert not(np.all(sim5.get_standardised_data() 
                      == sim4.get_standardised_data()))
    assert sim5.get_standardised_data().shape \
            == sim1.get_standardised_data().shape

@needs_elastix
def test_transform_roi():
    """Test transforming an ROI using the result of a registration."""

    reg = Registration(reg_dir)
    sim3 = SyntheticImage(sim2.shape)
    sim3.add_cube(4, name="cube")
    roi = sim3.get_roi("cube")
    roi2 = reg.transform(roi)
    assert roi2.get_contours() != roi.get_contours()
    assert roi2.get_mask(standardise=True).shape \
            == sim1.get_data(standardise=True).shape

@needs_elastix
def test_transform_structure_set():

    reg = Registration(reg_dir)
    sim3 = SyntheticImage(sim2.shape)
    sim3.add_cube(4, name="cube")
    sim3.add_sphere(2, name="sphere")
    ss = reg.transform(sim3.get_structure_set())
    assert len(ss.rois) == len(sim3.get_structure_set().rois)

@needs_elastix
def test_transform_points_structure_set():

    tfile_lines = [
            '(Transform "TranslationTransform")',
            '(NumberOfParameters 3)',
            '(TransformParameters 0 0 0)',
            '(InitialTransformParametersFileName "NoInitialTransform")',
            '(HowToCombineTransforms "Compose")',
            '(FixedImageDimension 3)',
            '(MovingImageDimension 3)',
            ]

    reg_dir = Path('tmp/transform')
    if reg_dir.exists():
        shutil.rmtree(str(reg_dir))
    reg_dir.mkdir()

    tfile_path = reg_dir / 'zero_translation.txt'
    tfile_path.unlink(missing_ok=True)
    with open(tfile_path, 'w') as tfile:
        for line in tfile_lines:
            tfile.write(f'{line}\n')

    reg = Registration(reg_dir, tfiles={'zero_translation': str(tfile_path)})
    sim3 = SyntheticImage(sim2.shape)
    sim3.add_cube(4, name="cube")
    sim3.add_sphere(2, name="sphere")
    ss1 = sim3.get_structure_set()
    ss2 = reg.transform(ss1, transform_points=True)
    assert len(ss1.rois) == 2
    assert len(ss1.rois) == len(ss2.rois)
    assert ss1.get_roi_names() == ss2.get_roi_names()

    for name in ss1.get_roi_names():
        roi1 = ss1.get_roi(name)
        roi2 = ss2.get_roi(name)
        compare_rois(roi1, roi2)

def test_read_parameters():
    """Test reading elastix parameter file into dict."""

    from skrt.registration import read_parameters
    params = read_parameters(pfiles_dir / "MI_Translation.txt")
    assert params["NumberOfResolutions"] == 4
    assert params["UseDirectionCosines"]
    assert isinstance(params["UseDirectionCosines"], bool)
    assert params["HowToCombineTransforms"] == "Compose"
    assert params["RequiredRatioOfValidSamples"] == 0.05
    assert isinstance(params["ImagePyramidSchedule"], list)
    assert params["ImagePyramidSchedule"][0] == 8

def test_write_parameters():
    """Test writing dict of parameters to elastix parameter file."""

    test_file = "tmp/test_params.txt"
    params = {
        "float": 0.4,
        "list": [0.4, 0.2, 0.1],
        "int": 6,
        "int_list": [2, 5, 2],
        "string": "test",
        "true": True,
        "false": False
    }
    from skrt.registration import write_parameters, read_parameters
    write_parameters(test_file, params)
    params2 = read_parameters(test_file)
    assert params == params2
    os.remove(test_file)

def test_adjust_parameters():
    """Test adjustment of an elastix parameter file."""
    
    pfile = pfiles_dir / "MI_Translation.txt"
    init_text = open(pfile).read()
    assert "(DefaultPixelValue 0)" in init_text
    from skrt.registration import adjust_parameters
    new_file = "tmp/tmp_pfile.txt"
    adjust_parameters(pfile, new_file, {"DefaultPixelValue": 10})
    assert os.path.exists(new_file)
    text = open(new_file).read()
    assert "(DefaultPixelValue 10)" in text
    os.remove(new_file)

@needs_elastix
def test_shift_parameters():
    """Test shifting translation parameters by a given amount."""

    from skrt.registration import read_parameters, shift_translation_parameters
    reg = Registration("tmp/reg")
    input_file = reg.tfiles[reg.steps[0]]
    init = read_parameters(input_file)["TransformParameters"]
    dx, dy, dz = 5, 7, 3
    outfile = "tmp/shifted.txt"
    shift_translation_parameters(input_file, dx, dy, dz, outfile)
    final = read_parameters(outfile)["TransformParameters"]
    assert final[0] == init[0] - dx
    assert final[1] == init[1] - dy
    assert final[2] == init[2] - dz
    os.remove(outfile)

def test_get_default_pfiles():
    """Test getting list of default parameter files."""

    from skrt.registration import get_default_pfiles
    default = get_default_pfiles()
    assert len(default)
    assert "MI_Affine.txt" in default

def test_get_default_pfiles_dir():
    """Test getting list of default parameter files via parent directory."""

    from skrt.registration import get_default_pfiles, get_default_pfiles_dir
    pfiles1 = get_default_pfiles()
    pfiles_dir = get_default_pfiles_dir()
    pfiles2 = list(get_default_pfiles_dir().iterdir())
    assert pfiles_dir.is_dir()
    assert "elastix" in pfiles_dir.name
    assert len(pfiles2)
    assert len(pfiles1) == len(pfiles2)

    for pfile in pfiles2:
        assert pfile.name in pfiles1

def test_add_default_pfile():
    """Test adding a default parameter file."""

    reg = Registration("tmp/reg")
    init_len = len(reg.steps)
    reg.add_default_pfile("MI_BSpline30", 
                          params={"MaximumNumberOfIterations": 300})
    assert len(reg.steps) == init_len + 1
    assert reg.steps[-1] == "MI_BSpline30"
    pars = reg.get_input_parameters(-1)
    assert pars["MaximumNumberOfIterations"] == 300
    assert pars["Transform"] == "BSplineTransform"

def test_translation_tfile():
    """Test creation of translation tfile from 3-element list."""

    translation = [5, 10, -5]
    reg = Registration("tmp/reg2", tfiles={"translation": translation})
    assert  len(reg.tfiles) == 1
    assert Path(reg.tfiles["translation"]).exists()
    parameters = read_parameters(reg.tfiles["translation"])
    assert parameters["TransformParameters"] == translation

def test_translation_tfile_with_image():
    """Test creation of translation tfile, with fixed image defined."""

    translation = [5, 10, -5]
    reg = Registration("tmp/reg2", fixed=im1,
            tfiles={"translation": translation})
    affine = im1.astype("nii").get_affine()
    affine[0:2, :] = -affine[0:2, :]
    voxel_size = [abs(dxyz) for dxyz in im1.get_voxel_size()]
    directions = [0 + affine[row, col] / voxel_size[col]
                for col in range(3) for row in range(3)]
    assert  len(reg.tfiles) == 1
    assert Path(reg.tfiles["translation"]).exists()
    parameters = read_parameters(reg.tfiles["translation"])
    assert parameters["TransformParameters"] == translation
    assert parameters["Size"] == im1.get_n_voxels()
    assert parameters["Spacing"] == voxel_size
    assert parameters["Direction"] == directions

def test_initial_alignment_from_string():
    """Test initial alignment defined by string."""

    # Create synthetic (blank) images.
    shapes = [(100, 100, 40), (80, 80, 50)]
    origins = [(-50, 40, -20), (0, -10, 30)]
    sims = [SyntheticImage(shapes[idx], origin=origins[idx]) for idx in [0, 1]]

    # Define expected translations.
    # For initial alignment from string, fixed and moving image
    # have their (x, y) centres aligned, and have z positions aligned
    # at image top, centre, or bottom.
    translations = {"_centre_" : [origins[1][idx] + 0.5 * shapes[1][idx]
        - origins[0][idx] - 0.5 * shapes[0][idx] for idx in range(3)]}
    dx, dy, dz = translations["_centre_"]
    translations.update({
        "_bottom_" : [dx, dy, origins[1][2] - origins[0][2]],
        "_top_" : [dx, dy,
            origins[1][2] + shapes[1][2] - origins[0][2] - shapes[0][2]],
            })

    # Check translations for initial alignment.
    for alignment, translation in translations.items():
        reg2_dir = "tmp/reg2"
        if Path(reg2_dir).exists():
            shutil.rmtree(reg2_dir)
        reg = Registration(reg2_dir, fixed=sims[0], moving=sims[1],
                initial_alignment=alignment)
        assert len(reg.tfiles) == 1
        assert(Path(reg.tfiles["initial_alignment"]).exists())
        parameters = read_parameters(reg.tfiles["initial_alignment"])
        assert parameters["TransformParameters"] == translation

def test_initial_alignment_from_dict():
    """Test initial alignment defined by dictionary."""

    # Create synthetic (blank) images.
    shapes = [(100, 100, 40), (80, 80, 50)]
    origins = [(-50, 40, -20), (0, -10, 30)]
    sims = [SyntheticImage(shapes[idx], origin=origins[idx]) for idx in [0, 1]]

    # Define expected translations, for initial alignment from dictionary:
    # 1 : align on lowest coordinate;
    # 2 : align on centre coordinate;
    # 3 : align on highest coordinate.
    translations = [
            ({"x": 1, "y": 1, "z":1},
             [origins[1][idx] - origins[0][idx] for idx in range(3)]),
            ({"x": 2, "y": 2, "z":2},
             list(sims[1].get_centre() - sims[0].get_centre())),
            ({"x": 3, "y": 3, "z":3},
             [origins[1][idx] + shapes[1][idx]
              - origins[0][idx] - shapes[0][idx] for idx in range(3)]),
            ]

    # Check translations for initial alignment.
    for alignment, translation in translations:
        reg2_dir = "tmp/reg2"
        if Path(reg2_dir).exists():
            shutil.rmtree(reg2_dir)
        reg = Registration(reg2_dir, fixed=sims[0], moving=sims[1],
                initial_alignment=alignment)
        assert len(reg.tfiles) == 1
        assert(Path(reg.tfiles["initial_alignment"]).exists())
        parameters = read_parameters(reg.tfiles["initial_alignment"])
        assert parameters["TransformParameters"] == translation

def test_initial_alignment_using_rois():
    """Test initial alignment defined by ROIs."""

    # Create synthetic images featuring a sphere.
    shapes = [(100, 100, 40), (80, 80, 50)]
    origins = [(-50, 40, -20), (0, -10, 30)]
    centres = [(30, 80, 0), (60, 40, 50)]
    radii = (10, 15)
    intensity=50
    sims = []
    for idx in range(len(shapes)):
        sims.append(SyntheticImage(shapes[idx], origin=origins[idx]))
        sims[idx].add_sphere(radius=radii[idx], name="sphere",
                centre=centres[idx], intensity=intensity)
        sims[idx].add_sphere(radius=radii[idx], name=f"sphere{idx}",
                centre=centres[idx], intensity=intensity)

    # Define expected translations for foreground alignment:
    # (1) upper coordinates; (2) centre coordinates; (3) lower coordinates.
    translations = {
            "sphere" : [centres[1][idx] - centres[0][idx] for idx in range(3)],
            ("sphere0", "sphere1") : [centres[1][idx] - centres[0][idx]
                for idx in range(3)],
            (("sphere0",), ("sphere1",)) : [centres[1][idx] - centres[0][idx]
                for idx in range(3)],
            (("sphere0", None), "sphere1") : [centres[1][idx] - centres[0][idx]
                for idx in range(3)],
            ("sphere0", ("sphere1", None)) : [centres[1][idx] - centres[0][idx]
                for idx in range(3)],
            (("sphere0", None), ("sphere1", None)) : [
                centres[1][idx] - centres[0][idx] for idx in range(3)],
            ("sphere", None) : [centres[1][idx] - centres[0][idx]
                for idx in range(3)],
            }

    # Check translations for initial alignment.
    for alignment, translation in translations.items():
        reg2_dir = "tmp/reg2"
        if Path(reg2_dir).exists():
            shutil.rmtree(reg2_dir)
        initial_alignment = alignment
        reg = Registration(reg2_dir, fixed=sims[0].get_image(),
                moving=sims[1].get_image(),
                initial_alignment=initial_alignment)
        assert len(reg.tfiles) == 1
        assert(Path(reg.tfiles["initial_alignment"]).exists())
        parameters = read_parameters(reg.tfiles["initial_alignment"])
        assert parameters["TransformParameters"] == translation
