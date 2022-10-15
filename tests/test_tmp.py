"""Tests for the ROI and StructureSet classes."""

import fnmatch
import math
import os
import random
import shutil
import pandas as pd
import pathlib
import pytest
import numpy as np
import matplotlib.colors

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from skrt.core import fullpath
from skrt.simulation import SyntheticImage
from skrt.structures import contour_to_polygon, polygon_to_contour, \
        StructureSet, ROI, interpolate_points_single_contour, \
        get_comparison_metrics


# Make temporary test dir
if not os.path.exists("tmp"):
    os.mkdir("tmp")

def get_synthetic_image_with_structure_set(shape=(100, 100, 100),
        cubes={"cube": 40}, spheres={"sphere": 20}):
    '''
    Create synthetic image with associated structure set.

    **Parameters:**

    shape : tuple, default=(100, 100, 100)
        Shape (x, y, z) of the image array.

    cubes : dict, default={"cube": 40}
        Dictionary defining cube ROIs to be included in structure set, where
        a key is an ROI name and the associated value is a cube side length.

    spheres : dict, default={"sphere": 20}
        Dictionary defining sphere ROIs to be included in structure set, where
        a key is an ROI name and the associated value is a sphere radius.
    '''
    # Create the synthetic image.
    sim = SyntheticImage(shape)

    # Add cube ROIs.
    for name, side_length in cubes.items():
        sim.add_cube(side_length=side_length, name=name,
                centre=sim.get_centre(), intensity=1)

    # Add sphere ROIs.
    for name, radius in spheres.items():
        sim.add_sphere(radius=radius, name=name,
                centre=sim.get_centre(), intensity=10)

    return sim

sim = get_synthetic_image_with_structure_set()
structure_set = sim.get_structure_set()
cube = structure_set.get_roi("cube")

def test_slice_thickness():

    # Set z-coordinate of first contour, and z-distance between contours.
    z0 = 2.0
    dz = 2.0

    # Create null ROI, and check that slice thickness is unset.
    roi1 = ROI()
    assert roi1.slice_thickness_contours == None

    # Check that slice thickness can be set.
    roi1.set_slice_thickness_contours(dz)
    assert roi1.slice_thickness_contours == dz

    # Create single-slice ROI, and check that slice thickness is unset.
    contours = {z0: [np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])]}
    roi2 = ROI(source=contours)
    assert roi2.slice_thickness_contours == None

    # Create double-slice ROI, and check that slice thickness is set.
    contours = {z0: [np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])],
            z0 + dz: [np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])]}
    roi3 = ROI(source=contours)
    assert roi3.slice_thickness_contours == dz

    # Create StructureSet from single-slice ROI and double-slice ROI,
    # then check that the slice thickness of the former is set
    # to match the slice thickness of the latter.
    ss = StructureSet([roi2, roi3])
    for roi in ss.get_rois():
        assert roi.get_slice_thickness_contours() == dz
