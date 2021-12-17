"""Tests of VoxTox extensions to ROI and StructureSet classes."""

import os
from pathlib import Path

from skrt.simulation import SyntheticImage
from voxtox.structures import StructureSet

# Make temporary test dir.
tmp_dir = Path('tmp')
tmp_dir.mkdir(exist_ok=True)

# Create synthetic structure set.
sim = SyntheticImage((100, 100, 40))
sim.add_cube(side_length=40, name="cube", intensity=1)
sim.add_sphere(radius=20, name="sphere", intensity=10)
structure_set = StructureSet(sim.get_structure_set())
for name in structure_set.get_roi_names():
    roi_path = tmp_dir / f'{name}.txt'
    roi_path.unlink(missing_ok=True)

def test_structure_set_write_point_file():

    # Write structure set as point cloud.
    structure_set.write(outdir=str(tmp_dir), point_cloud=True)

    for name in structure_set.get_roi_names():
        # Check that the ROI point cloud was written.
        assert (tmp_dir / f'{name}.txt').exists()

        # Load the point-cloud data,
        # then check that the contours obtain are the same as the originals.
        ss1 = StructureSet(sources=str(tmp_dir), image=sim)

        # Check centroids.
        assert structure_set.get_roi(name).get_centroid().all() == \
                ss1.get_roi(name).get_centroid().all()

        # Check number of planes.
        ss0_keys = list(structure_set.get_roi(name).get_contours().keys())
        ss1_keys = list(ss1.get_roi(name).get_contours().keys())
        assert len(ss0_keys) == len(ss1_keys)

        ss0_keys.sort()
        ss1_keys.sort()

        for i in range(len(ss0_keys)):

            # Check plane z-coordinates.
            assert ss0_keys[i] == ss1_keys[i]

            # Check number of contours in plane.
            contours0 = structure_set.get_roi(name).get_contours()[ss0_keys[i]]
            contours1 = structure_set.get_roi(name).get_contours()[ss1_keys[i]]
            assert len(contours0) == len(contours1)

            # Check contour points.
            for j in range(len(contours0)):
                assert contours0[j].all() == contours1[j].all()
