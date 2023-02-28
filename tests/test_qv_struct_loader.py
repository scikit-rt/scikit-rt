"""
Test legacy QuickViewer code.

Tests migrated from tests/test_struct_loader.py at:
    https://github.com/hlpullen/quickviewer

Test the StructureSet class.
"""

import os
import shutil

from skrt.simulation import SyntheticImage
from skrt.viewer.core import StructureSet
from matplotlib.colors import to_rgba

# Make temporary test directory, and quickviewer subdirectory.
if not os.path.exists('tmp'):
    os.mkdir('tmp')
if not os.path.exists('tmp/qv'):
    os.mkdir('tmp/qv')
ss_path = 'tmp/qv/loader'
if os.path.exists(ss_path):
    shutil.rmtree(ss_path)
os.mkdir(ss_path)

def make_sim():
    """Make synthetic image containing ROIs."""

    sim = SyntheticImage((100, 100, 10))
    sim.add_sphere(20, name="sphere1")
    sim.add_sphere(25, name="sphere2")
    sim.add_cube(10, name="cube1")
    sim.add_cube(15, name="cube2")
    return sim

sim = make_sim()
sim_path = 'tmp/qv/sim.nii'
sim.write(sim_path)
ss = sim.get_structure_set()
ss.write(outdir=ss_path, ext='.nii')

# Load single structure from single file
def test_single():
    s = StructureSet(f"{ss_path}/cube1.nii")
    assert len(s.get_structs()) == 1
    assert not len(s.get_comparisons())
    #  assert not len(s.get_structs(ignore_unpaired=True))
    assert s.get_structs()[0].name == "Cube1"

#  # Default colour
#  def test_default_color():
    #  s = StructureSet("data/structs/my_structs/cube.nii",
                     #  names={"*cube*": "right parotid"})
    #  struct = s.get_structs()[0]
    #  assert struct.color == to_rgba("red")

# Custom name and colour
def test_custom_name():
    names = {"custom": "*cube*.nii"}
    colors = {"custom": "purple"}
    s = StructureSet(f"{ss_path}/cube1.nii", names=names,
                     colors=colors)
    struct = s.get_structs()[0]
    assert struct.name == "custom"
    assert struct.color == to_rgba("purple")
    assert struct.loaded
    assert struct.label == ""

# Load multiple structures from directory
def test_dir():
    colors = {"cube1": "red", "sphere1": "green"}
    s = StructureSet(structs=ss_path, colors=colors)
    structs = s.get_structs()
    assert(len(structs) == 4)
    names_colors = {s.name: s.color for s in structs}
    assert names_colors["Cube1"] == to_rgba(colors["cube1"])
    assert names_colors["Sphere1"] == to_rgba(colors["sphere1"])

# Load structures using wildcard filename
def test_wildcard():
    names = {"with_cube": "*cube*", "sphere_only": "*"}
    s = StructureSet(f"{ss_path}/*1*", names=names)
    snames = [s.name for s in s.get_structs()]
    assert len(snames) == 2
    assert sorted(snames) == sorted(list(names.keys()))

"""
# Load multiple structure masks from one file
def test_multi_structs():
    s = StructureSet(
        multi_structs="data/structs/my_structs/sphere_and_cube.nii")
    structs = s.get_structs()
    assert(len(structs) == 2)
    assert structs[0].name == "Structure 1"

# Test list of structure names inside file
def test_many_names():
    names = ["cube", "sphere"]
    colors = {"cube": "green"}
    s = StructureSet(
        multi_structs="data/structs/my_structs/sphere_and_cube.nii",
        names=names, colors=colors)
    snames = [s.name for s in s.get_structs()]
    assert sorted(snames) == sorted(names)
    assert [s for s in s.get_structs() if s.name == "cube"][0].color == \
            to_rgba("green")

# Load structures from list of files
def test_list():
    s = StructureSet(["data/structs/my_structs", 
                      "data/structs/my_structs/subdir"])
    assert len(s.get_structs()) == 5

# Load structures with labels
def test_labels():

    # Load structures from multiple sources
    multi = {"set1": "data/structs/my_structs/sphere_and_cube.nii"}
    structs = {"set2": "data/structs/my_structs/subdir"}
    names = {"set1": ["Sphere", "Cube"]}
    colors = {"set1": {"*": "green"}, 
              "set2": {"cube": "yellow", "sphere": "black"}}
    s = StructureSet(structs, multi, names=names, colors=colors)

    # Test structure properties
    structs = s.get_structs()
    assert len(structs) == 4
    assert len([s for s in structs if s.label == "set1"]) == 2
    assert len([s for s in structs if s.label == "set2"]) == 2
    assert len([s for s in structs if s.name == "Sphere"]) == 2
    assert len([s for s in structs if s.name == "Cube"]) == 2
    assert [s.color for s in structs if s.label == "set1"][0] == \
            to_rgba("green")
    assert [s.color for s in structs if s.label == "set1"][1] == \
            to_rgba("green")
    assert [s.color for s in structs if s.label == "set2"][1] == \
            to_rgba("yellow")
    assert [s.color for s in structs if s.label == "set2"][0] == \
            to_rgba("black")

    # Test comparisons
    comps = s.get_comparisons()
    assert len(comps) == 2
    assert sorted([c.name for c in comps]) == sorted(["Sphere", "Cube"])
    assert len(s.get_structs(False)) == len(s.get_structs(True))
    assert not len(s.get_standalone_structs())

# Test extraction of standalone structs
def test_standalone():

    # Load structures from multiple sources
    multi = {"set1": "data/structs/my_structs/sphere_and_cube.nii"}
    structs = {"set2": "data/structs/my_structs/subdir"}
    names = {"set1": ["Sphere"]}
    s = StructureSet(structs=structs, multi_structs=multi, names=names)
    assert len(s.get_structs()) == 4
    assert len(s.get_structs(True)) == 2
    assert len(s.get_comparisons()) == 1
    assert len(s.get_standalone_structs()) == 2

# Load pairs of structures for comparison
def test_pairs():
    pairs = [
        ["data/structs/my_structs/sphere.nii", 
         "data/structs/my_structs/cube.nii"],
        ["data/structs/my_structs/sphere_and_cube.nii",
         "data/structs/my_structs/subdir/sphere.nii"]]
    s = StructureSet(pairs)
    assert len(s.get_structs()) == 4
    assert len(s.get_comparisons()) == 2
    assert len(s.get_structs(True)) == 4
    assert len(s.get_standalone_structs()) == 0
"""

# Comparison of two structs
def test_two_comparison():
    s = StructureSet(f"{ss_path}/sphere*")
    assert not len(s.get_standalone_structs())
    assert len(s.get_structs(True)) == 2
    assert len(s.get_comparisons()) == 1
