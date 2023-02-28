"""
Test legacy QuickViewer code.

Tests migrated from tests/test_structs.py at:
    https://github.com/hlpullen/quickviewer

Test the Struct class."""

from pytest import approx

from skrt.simulation import SyntheticImage
from skrt.viewer.core import Struct

sm = Struct("data/structs/"
                "RTSTRUCT_CT_20140715_113632_002_oral_cavity.nii.gz")
# Define cube.
cx, cy, cz = 50.5, 50.5, 15.5
gn = SyntheticImage((100, 100, 30))
gn.add_cube(side_length=10, centre=(cx, cy, cz), name="cube")
roi_cube = gn.get_roi("cube")
sm = Struct(roi_cube.get_mask()[::-1, ::-1, :], name=roi_cube.name,
               origin=roi_cube.get_origin(),
               voxel_sizes=roi_cube.get_voxel_size())

def test_struct_mask():
    assert len(sm.voxel_sizes) == 3
    assert len(sm.origin) == 3
    assert len(sm.n_voxels) == 3
    assert sm.get_volume("voxels") > 0
    assert sm.get_volume("ml") > 0
    assert sm.get_volume("mm") == approx(sm.get_volume("ml") * 1e3)
    assert len(sm.get_length("mm")) == 3
    assert sm.get_length("voxels")[0]

def test_contours():
    assert len(sm.contours["x-y"])
    assert len(sm.contours["y-z"]) <= sm.n_voxels["x"]
