'''Tests for the ROI and RtStruct classes.'''

import os
import shutil
import pandas as pd

from skrt.simulation import SyntheticImage
from skrt.structures import RtStruct, ROI


# Make temporary test dir
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Create synthetic structure set
sim = SyntheticImage((100, 100, 40))
sim.add_cube(side_length=40, name='cube')
sim.add_sphere(radius=20, name='sphere')
structs = sim.get_rtstruct()


def test_rtstruct_from_sythetic_image():
    assert isinstance(structs, RtStruct)

def test_write_nifti():
    nii_dir = 'tmp/nii_structs'
    if os.path.exists(nii_dir):
        shutil.rmtree(nii_dir)
    structs.write(outdir='tmp/nii_structs')
    assert len(os.listdir(nii_dir)) == 2

def test_get_structs():
    assert len(structs.get_structs()) == 2

def test_get_struct_names():
    names = structs.get_struct_names()
    assert len(names) == 2
    assert 'cube' in names
    assert 'sphere' in names

def test_get_dict():
    sdict = structs.get_struct_dict()
    assert len(sdict) == 2
    assert set(sdict.keys()) == set(['cube', 'sphere'])

def test_get_struct():
    roi = structs.get_struct('cube')
    assert isinstance(roi, ROI)
    assert roi.name == 'cube'

def test_rename():
    new_names = {
        'cube2': ['cube', 'test'],
        'sphere2': 'spher*'
    }
    structs.rename_structs(new_names)
    assert set(structs.get_struct_names()) == set(new_names.keys())
    old_names = {
        'cube': 'cube2',
        'sphere': 'sphere2'
    }
    structs.rename_structs(old_names)
    assert set(structs.get_struct_names()) == set(old_names.keys())

def test_copy_rename():
    new_names = {'cube3': 'cube'}
    structs2 = structs.copy(new_names, name='copy', keep_renamed_only=True)
    assert len(structs2.get_structs()) == 1
    assert structs2.get_struct_names() == ['cube3']
    assert structs2.name == 'copy'
    structs.rename_structs({'cube': 'cube3'})

def test_copy_remove():
    structs2 = structs.copy(to_remove='cube')
    assert structs2.get_struct_names() == ['sphere']

def test_copy_keep():
    structs2 = structs.copy(to_keep='sphere')
    assert structs2.get_struct_names() == ['sphere']

def test_read_nii():
    nii_dir = 'tmp/nii_structs'
    structs_from_nii = RtStruct(nii_dir)
    assert len(structs_from_nii.get_structs()) == 2
    assert set(structs.get_struct_names()) \
            == set(structs_from_nii.get_struct_names())

def test_get_geometry():
    geom = structs.get_geometry()
    assert isinstance(geom, pd.DataFrame)
    assert geom.shape[0] == 2

def test_get_comparison_pairs():
    pairs = structs.get_comparison_pairs()
    assert len(pairs) == 2
    assert len(pairs[0]) == 2

def test_get_comparison_pairs_with_other():
    structs2 = RtStruct('tmp/nii_structs')
    pairs = structs.get_comparison_pairs(structs2)
    assert len(pairs) == 2
    assert pairs[0][0].name == pairs[0][1].name
    assert pairs[1][0].name == pairs[1][1].name

def test_get_comparison():
    comp = structs.get_comparison()
    assert isinstance(comp, pd.DataFrame)
    assert comp.shape[0] == 2

def test_plot_comparisons():
    plot_dir = 'tmp/struct_plots'
    if os.path.exists(plot_dir):
        shutil.rmdir(plot_dir)
    structs.plot_comparisons(outdir=plot_dir, show=False)
    assert len(os.listdir(plot_dir)) == 2
    
def test_write_dicom():
    pass
