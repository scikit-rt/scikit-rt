"""Check that a patient object can be read."""

from pathlib import Path

import numpy as np
import os
import shutil
import sys
import time

from pydicom.uid import generate_uid

import skrt
from skrt.patient import Patient, Study
from skrt.simulation import SyntheticImage

# Create synthetic patient object
pid = "test_patient"
pdir = f"tmp/{pid}"
p = Patient(pid)

# Create synthetic image
nz = 40
sim = SyntheticImage((100, 100, nz), auto_timestamp=True)
sim.add_cube(50, name='cube')
sim.add_sphere(25, name='sphere')

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

def test_write_blank():
    assert p.id == pid
    assert len(p.studies) == 0
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.write("tmp")
    assert os.path.exists(pdir)
    assert len(os.listdir(pdir)) == 0
    shutil.rmtree(pdir)

def test_add_study():
    pid = "test_patient"
    p = Patient(pid)
    p.add_study()
    assert len(p.studies) == 1
    assert isinstance(p.studies[0].timestamp, str)
    assert p.studies[0].subdir == ""
    p.add_study("something")
    assert len(p.studies) == 2
    assert p.studies[-1].subdir == "something"

def test_write_with_studies():
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.studies = []
    p.add_study()
    p.add_study("something")
    p.write("tmp")
    assert len(os.listdir(pdir)) == 2
    assert os.path.isdir(f"{pdir}/{p.studies[0].timestamp}")
    assert os.path.isdir(f"{pdir}/something")
    assert os.path.isdir(f"{pdir}/something/{p.studies[-1].timestamp}")

def test_load_studies():
    p.studies = []
    p.add_study("study1")
    p.add_study("study2")
    p.write("tmp")
    p2 = Patient(pdir)
    assert len(p2.studies) == 2
    assert p2.studies[0].subdir == "study1"
    assert p2.studies[1].subdir == "study2"

def test_study_with_images():
    p.studies = []
    im = sim.get_image()
    p.add_study(images=[im], image_type="MR")
    s = p.studies[0]
    assert len(s.mr_images) == 1
    assert s.mr_images[0].get_voxel_size() == im.get_voxel_size()
    assert len(s.mr_structure_sets) == 1
    assert len(s.mr_structure_sets[0].get_rois()) == 2
    s.add_image(im, image_type="CT")
    assert len(s.ct_images) == 1
    assert len(s.ct_structure_sets) == 1
    p.write("tmp", structure_set=None)
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert os.path.exists(f"{sdir}/MR")
    assert os.path.exists(f"{sdir}/CT")
    assert not os.path.exists(f"{sdir}/RTSTRUCT")
    assert len(os.listdir(f"{sdir}/CT")) == 1
    assert im.timestamp in os.listdir(f"{sdir}/CT")
    assert im.timestamp in os.listdir(f"{sdir}/MR")
    assert len(os.listdir(f"{sdir}/CT/{im.timestamp}")) == im.n_voxels[2]

def test_load_images():
    p2 = Patient(pdir)
    s = p2.studies[0]
    assert len(s.ct_images) == 1
    assert len(s.mr_images) == 1
    assert np.all(s.ct_images[0].get_affine() == sim.get_affine())
    images = p2.combined_objs('image_types')
    assert len(images) == 2
    assert images[0] == s.ct_images[0]
    assert images[1] == s.mr_images[0]

def test_unsorted_dicom_defaults():
    for unsorted_dicom in [True, False]:
        skrt.core.Defaults().unsorted_dicom = unsorted_dicom
        p2 = Patient(pdir)
        s = p2.studies[0]
        assert len(s.ct_images) == 1
        assert hasattr(s.ct_images[0], "dicom_paths") == unsorted_dicom
        '''
        ### Need to understand problems with MR data ###
        assert len(s.mr_images) == 1
        assert hasattr(s.mr_images[0], "dicom_paths") == unsorted_dicom
        '''

def test_write_rois_nifti():
    p.studies = []
    p.add_study(images=[sim.get_image()])
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.write("tmp", ext=".nii", structure_set="all")
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert 'CT' in os.listdir(f"{sdir}")
    assert 'CT' in os.listdir(f"{sdir}/RTSTRUCT")
    nifti_rois = os.listdir(f"{sdir}/RTSTRUCT/CT/{sim.timestamp}")
    assert "cube.nii" in nifti_rois
    assert "sphere.nii" in nifti_rois

def test_read_nifti_patient():
    pass

def test_write_rois_dicom():
    p.studies = []
    p.add_study(images=[sim.get_image()])
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.write("tmp", ext='.dcm', structure_set="all")
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert 'CT' in os.listdir(f"{sdir}")
    assert os.path.exists(f"{sdir}/CT/{sim.timestamp}")
    items = os.listdir(f"{sdir}/CT/{sim.timestamp}")
    assert len(items) == nz
    for item in items:
        assert 'dcm' in item
    assert 'CT' in os.listdir(f"{sdir}/RTSTRUCT")
    assert os.path.exists(f"{sdir}/RTSTRUCT/CT/{sim.timestamp}")
    items = os.listdir(f"{sdir}/RTSTRUCT/CT/{sim.timestamp}")
    assert len(items) == 1

def test_read_dicom_patient():
    p_test = Patient(pdir)
    assert len(p_test.studies) == 1
    study = p_test.studies[-1]
    assert len(study.ct_images) == 1
    assert len(study.ct_images[0].files) == nz
    assert len(study.ct_structure_sets) == 1
    roi_names = study.ct_structure_sets[0].get_roi_names()
    assert len(roi_names) == 2
    assert "cube" in roi_names
    assert "sphere" in roi_names
    assert isinstance(p_test._init_time, float)
    assert p_test._init_time > 0

def test_patient_references():
    # Test that all of a patient's data objects have a reference to the patient.
    p_test = Patient(pdir)
    n_objs = {"image_types": 0, "structure_set_types": 0, "studies": 0}

    for category in n_objs.keys():
        objs = getattr(p_test, category, p_test.combined_objs(category))
        for obj in objs:
            n_objs[category] += 1
            assert obj.patient == p_test
        assert 1 == n_objs[category]

def test_copy_dicom_patient():
    p_test = Patient(pdir)
    pdir_copy = f"tmp/test_patient_copy"
    p_test.copy(pdir_copy)
    p_test_copy = Patient(pdir_copy)

    assert len(p_test.studies) == len(p_test.studies)
    s1 = p_test.studies[-1]
    s2 = p_test_copy.studies[-1]
    assert len(s1.ct_images) == len(s2.ct_images)
    assert len(s1.ct_images[0].files) == len(s2.ct_images[0].files)
    assert len(s1.ct_structure_sets) == len(s2.ct_structure_sets)
    roi_names1 = s1.ct_structure_sets[0].get_roi_names()
    roi_names2 = s2.ct_structure_sets[0].get_roi_names()
    assert roi_names1 == roi_names2

def test_null_patient():
    p = Patient()
    assert(type(p).__name__ == 'Patient')
    assert(p.id == '')
    assert(p.path == '')
    assert(p.studies == [])
    assert(p.get_studies() == [])
    assert(p.get_images() == [])
    assert(p.get_structure_sets() == [])
    assert(p.get_doses() == [])
    assert(p.get_plans() == [])

def test_null_study():
    s = Study()
    assert(type(s).__name__ == 'Study')
    assert(s.date == '')
    assert(s.files == [])
    assert(s.path == '')
    assert(s.subdir == '')
    assert(s.time == '')
    assert(s.timestamp == '')
    assert(s.get_images() == [])
    assert(s.get_structure_sets() == [])
    assert(s.get_doses() == [])
    assert(s.get_plans() == [])

def test_unsorted_images():
    '''Test loading to patient object of unsorted DICOM images.'''

    # Create directory for patient data.
    pid = "test_patient2"
    pdir = Path("tmp")/pid
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)

    # Create set of (duplicate) images and write to patient directory.
    im = sim.get_image()
    study_date = time.strftime("%Y%m%d")
    study_time = time.strftime("%H%M%S")
    study_instance_uid = generate_uid()
    header_extras = {
            "StudyDate": study_date,
            "StudyInstanceUID": study_instance_uid,
            "StudyTime": study_time
            }
    modality = "CT"
    series_numbers = range(1, 6)
    for idx in series_numbers:
        header_extras["SeriesNumber"] = idx
        im.write(pdir/str(idx), modality=modality, header_extras=header_extras)

    # Load images to Patient object.
    p = Patient(pdir, unsorted_dicom=True)

    # Check that there is a single study, then check its date and time.
    assert len(p.studies) == 1
    s = p.get_studies()[0]
    assert s.study_instance_uid == study_instance_uid
    assert s.date == study_date
    assert s.time == study_time

    # Check that the number of images loaded is correct,
    # that they have the right modality, that the series numbers
    # are as expected, and that the image data can be loaded.
    assert len(p.get_images()) == len(series_numbers)
    assert len(s.get_images()) == len(series_numbers)
    series_numbers2 = []
    for key, images in s.image_types.items():
        assert key == modality.lower()
        assert len(images) == len(series_numbers)
        for image in images:
            series_number = image.get_dicom_dataset().SeriesNumber
            assert series_number in series_numbers
            series_numbers2.append(series_number)
            image.load()
            image.data.shape == im.data.shape
    assert len(series_numbers) == len(series_numbers2)

def test_pathlib_path():
    # Test passing of pathlib.Path.
    p = Patient(Path())
    s = Study(Path())
    assert p.path == skrt.core.fullpath(".")
    assert s.path == skrt.core.fullpath(".")

def test_id_mappings():
    p_test = Patient(pdir)
    mapped_id = "new_id"
    id_mappings = {p_test.id: mapped_id}
    p_mapped = Patient(pdir, id_mappings=id_mappings)
    assert p_mapped.id == mapped_id
