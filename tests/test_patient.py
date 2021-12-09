"""Check that a patient object can be read."""

import numpy as np
import os
import shutil

from skrt.patient import Patient, Study
from skrt.simulation import SyntheticImage

# Create synthetic patient object
pid = "test_patient"
pdir = f"tmp/{pid}"
p = Patient(pid)

# Create synthetic image
sim = SyntheticImage((100, 100, 40))
sim.add_cube(50, name='cube')
sim.add_sphere(25, name='sphere')

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

def test_write_rois_nifti():
    p.studies = []
    p.add_study(images=[sim])
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.write("tmp", ext=".nii", structure_set="all")
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert os.path.exists(f"{sdir}/CT")
    assert os.path.exists(f"{sdir}/RTSTRUCT/CT")
    nifti_rois = os.listdir(f"{sdir}/RTSTRUCT/CT/{sim.timestamp}/"
                               f"RTSTRUCT_{sim.structure_sets[0].timestamp}")
    assert "cube.nii" in nifti_rois
    assert "sphere.nii" in nifti_rois

def test_write_rois_dicom():
    pass

def test_read_nifti_patient():
    pass

def test_null_patient():
    p = Patient()
    assert(type(p).__name__ == 'Patient')
    assert(p.id == '')
    assert(p.path == '')
    assert(p.studies == [])

def test_null_study():
    s = Study()
    assert(type(s).__name__ == 'Study')
    assert(s.date == '')
    assert(s.files == [])
    assert(s.path == '')
    assert(s.subdir == '')
    assert(s.time == '')
    assert(s.timestamp == '')
