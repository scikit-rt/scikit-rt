'''Tests for the ROI and StructureSet classes.'''

import os
import shutil
import pandas as pd

from skrt.simulation import SyntheticImage
from skrt.structures import StructureSet, ROI


# Make temporary test dir
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Create synthetic structure set
sim = SyntheticImage((100, 100, 40))
sim.add_cube(side_length=40, name='cube', intensity=1)
sim.add_sphere(radius=20, name='sphere', intensity=10)
structure_set = sim.get_structure_set()


def test_structure_set_from_sythetic_image():
    assert isinstance(structure_set, StructureSet)

def test_write_nifti():
    nii_dir = 'tmp/nii_structs'
    if os.path.exists(nii_dir):
        shutil.rmtree(nii_dir)
    structure_set.write(outdir='tmp/nii_structs')
    assert len(os.listdir(nii_dir)) == 2

def test_get_rois():
    assert len(structure_set.get_rois()) == 2

def test_get_roi_names():
    names = structure_set.get_roi_names()
    assert len(names) == 2
    assert 'cube' in names
    assert 'sphere' in names

def test_get_dict():
    sdict = structure_set.get_roi_dict()
    assert len(sdict) == 2
    assert set(sdict.keys()) == set(['cube', 'sphere'])

def test_get_roi():
    roi = structure_set.get_roi('cube')
    assert isinstance(roi, ROI)
    assert roi.name == 'cube'

def test_rename():
    new_names = {
        'cube2': ['cube', 'test'],
        'sphere2': 'spher*'
    }
    structure_set.rename_rois(new_names)
    assert structure_set.get_rois()[0].name != structure_set.get_rois()[0].original_name
    assert set(structure_set.get_roi_names()) == set(new_names.keys())
    old_names = {
        'cube': 'cube2',
        'sphere': 'sphere2'
    }
    structure_set.rename_rois(old_names)
    assert set(structure_set.get_roi_names()) == set(old_names.keys())

def test_copy_rename():
    new_names = {'cube3': 'cube'}
    structure_set2 = structure_set.copy(new_names, name='copy', keep_renamed_only=True)
    assert len(structure_set2.get_rois()) == 1
    assert structure_set2.get_roi_names() == ['cube3']
    assert structure_set2.name == 'copy'
    structure_set.rename_rois({'cube': 'cube3'})

def test_copy_remove():
    structure_set2 = structure_set.copy(to_remove='cube')
    assert structure_set2.get_roi_names() == ['sphere']

def test_copy_keep():
    structure_set2 = structure_set.copy(to_keep='sphere')
    assert structure_set2.get_roi_names() == ['sphere']

def test_read_nii():
    nii_dir = 'tmp/nii_structs'
    structs_from_nii = StructureSet(nii_dir)
    assert len(structs_from_nii.get_rois()) == 2
    assert set(structure_set.get_roi_names()) \
            == set(structs_from_nii.get_roi_names())

def test_get_geometry():
    geom = structure_set.get_geometry()
    assert isinstance(geom, pd.DataFrame)
    assert geom.shape[0] == 2

def test_get_comparison_pairs():
    pairs = structure_set.get_comparison_pairs()
    assert len(pairs) == 2
    assert len(pairs[0]) == 2

def test_get_comparison_pairs_with_other():
    structure_set2 = StructureSet('tmp/nii_structs')
    pairs = structure_set.get_comparison_pairs(structure_set2)
    assert len(pairs) == 2
    assert pairs[0][0].name == pairs[0][1].name
    assert pairs[1][0].name == pairs[1][1].name

def test_get_comparison():
    comp = structure_set.get_comparison()
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape[0] == 2

def test_plot_comparisons():
    plot_dir = 'tmp/struct_plots'
    if os.path.exists(plot_dir):
        shutil.rmdir(plot_dir)
    structure_set.plot_comparisons(outdir=plot_dir, show=False)
    assert len(os.listdir(plot_dir)) == 2
    
def test_write_dicom():
    pass

def test_roi_from_image_threshold():
    roi = ROI(sim, mask_level=5)  
    assert roi.get_area() == sim.get_roi("sphere").get_area()
