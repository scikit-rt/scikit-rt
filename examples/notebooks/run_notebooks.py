"""
Functions for running scikit-rt example notebooks non-interactively.

This module is intended mainly to allow easy checking that the notebooks
run without errors following any updates to the scikit-rt code.

The following functions are defined:
* **example_notebooks()** - Return sorted list of names of example notebooks;
* **run_notebooks()** - Run example notebooks non-interactively.

The following global variables are defined:

* **PROJECT_PATH** - Path to scikit-rt top-level directory;
* **NOTEBOOKS_PATH** - Path to directory containing example notebooks;
* **EXAMPLES_MD_PATH** - Path to examples documentation file (examples.md).
"""

from pathlib import Path
from subprocess import run
from sys import argv

import skrt

# Path to scikit-rt top-level directory.
PROJECT_PATH = Path(skrt.__file__).parents[2]
# Path to directory containing example notebooks;
NOTEBOOKS_PATH = PROJECT_PATH / "examples" / "notebooks"
# Path to examples documentation file.
EXAMPLES_MD_PATH = PROJECT_PATH /"docs"/"markdown"/"examples.md"

def example_notebooks():
    """
    Return sorted list of names of example notebooks.

    Notebook names are determined by parsing the examples documentation file.
    """
    with open(EXAMPLES_MD_PATH) as examples_md:
        notebooks = [line.split(".ipynb]")[0].split("[")[-1]
                     for line in examples_md if ".ipynb]" in line]

    return sorted(notebooks)

def run_notebooks(notebooks=None):
    """
    Run example notebooks non-interactively.

    **Parameter:**

    notebooks : list, default=None
        List of names (without suffixes) of example notebooks to be run.
        If None, all notebooks in the list returned by example_notebooks()
        are run.
    """
    failures = []
    notebooks = notebooks or example_notebooks()
    for jnotebook in notebooks:
        notebook_path = NOTEBOOKS_PATH / f"{jnotebook}.ipynb"
        cmd = ("jupyter nbconvert --to notebook "
               f"--execute {notebook_path} --output=tmp.ipynb")
        print(f"\n{cmd}")
        result = run(cmd.split())
        if 0 != result.returncode:
            failures.append(jnotebook)

if "__main__" == __name__:
    """
    Call function for running example notebooks, or a selection of them.
    """
    if len(argv) > 1 and argv[1] in ["-h", "--h", "-help", "--help"]:
        print("\nUsage: python run_notebooks.py <notebooks>")
        print("\n       <notebooks> : space-separated names (no suffixes) "
              "of notebooks to run;")
        print("       if omitted, all example notebooks are run.")
    else:
        notebooks = argv[1:]
        failures = run_notebooks(notebooks)
        print(f"\nNotebooks run:")
        for notebook in notebooks:
            print(f"    {notebook}")
        if failures:
            print(f"\nNotebooks giving non-zero return code:")
            for notebook in failures:
                print(f"    {notebook}")
        else:
            print("\nNo notebooks giving non-zero return code")
