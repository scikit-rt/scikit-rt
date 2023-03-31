"""Tests of multi-patient dataset."""

from pathlib import Path
from shutil import rmtree

from skrt.core import fullpath
from skrt.multi import PatientDataset
from skrt.patient import Patient

data_dir = Path(fullpath("tmp/multi"))
if data_dir.exists():
    rmtree(data_dir)

paths = [data_dir / f"path{idx}" for idx in range(10)]
for path in paths:
    path.mkdir(parents=True)

def test_null_patient_dataset():
    """Test creation of null PatientDataset."""
    ds = PatientDataset()
    assert(type(ds).__name__ == 'PatientDataset')
    assert isinstance(ds.paths, list)
    assert not ds.paths

def test_single_path_patient_dataset():
    """Test creation of PatientDataset for single path."""
    ds = PatientDataset(paths[0])
    assert isinstance(ds.paths, list)
    assert 1 == len(ds.paths)
    assert str(paths[0]) == ds.paths[0]

def test_add_to_patient_dataset():
    """Test additions of paths to PatientDataset."""
    ds = PatientDataset()
    for idx, path in enumerate(paths):
        ds.add_path(path)
        assert idx + 1 == len(ds.paths)
        assert str(path) == ds.paths[-1]

def test_iteration_over_patient_dataset():
    """Test iteration over PatientDataset."""
    ds = PatientDataset(paths)
    for idx, p in enumerate(ds):
        assert isinstance(p, Patient)
        assert p.path == str(paths[idx])
    assert idx == len(paths) - 1
