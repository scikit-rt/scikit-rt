"""Check that a patient object can be read."""

import numpy as np
import os
import shutil

from skrt.patient import Patient
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

def test_study_with_scans():
    p.studies = []
    p.add_study(images=[sim], scan_type="MR")
    s = p.studies[0]
    assert len(s.mr_scans) == 1
    assert s.mr_scans[0].get_voxel_size() == sim.get_voxel_size()
    assert len(s.mr_structs) == 1
    assert len(s.mr_structs[0].get_structs()) == 2
    s.add_scan(sim, scan_type="CT")
    assert len(s.ct_scans) == 1
    assert len(s.ct_structs) == 1
    p.write("tmp", structure_set=None)
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert os.path.exists(f"{sdir}/MR")
    assert os.path.exists(f"{sdir}/CT")
    assert not os.path.exists(f"{sdir}/RTSTRUCT")
    assert len(os.listdir(f"{sdir}/CT")) == 1
    assert sim.timestamp in os.listdir(f"{sdir}/CT")
    assert sim.timestamp in os.listdir(f"{sdir}/MR")
    assert len(os.listdir(f"{sdir}/CT/{sim.timestamp}")) == sim.n_voxels[2]

def test_load_images():
    p2 = Patient(pdir)
    s = p2.studies[0]
    assert len(s.ct_scans) == 1
    assert len(s.mr_scans) == 1
    assert np.all(s.ct_scans[0].get_affine() == sim.get_affine())

def test_write_structs_nifti():
    p.studies = []
    p.add_study(images=[sim])
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    p.write("tmp", ext=".nii", structure_set="all")
    sdir = f"{pdir}/{p.studies[0].timestamp}"
    assert os.path.exists(f"{sdir}/CT")
    assert os.path.exists(f"{sdir}/RTSTRUCT/CT")
    nifti_structs = os.listdir(f"{sdir}/RTSTRUCT/CT/{sim.timestamp}/"
                               f"RTSTRUCT_{sim.get_structs()[0].timestamp}")
    assert "cube.nii" in nifti_structs
    assert "sphere.nii" in nifti_structs

def test_write_structs_dicom():
    pass

def test_read_nifti_patient():
    pass
