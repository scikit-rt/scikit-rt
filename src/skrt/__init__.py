"""
Toolkit for analysis of radiotherapy data.

Scikit-rt, imported as skrt, is a toolkit for loading, plotting, and analysing
radiotherapy data in DICOM and NIfTI formats.  It includes image registration
via elastix, NiftyReg, and MATLAB, single- and multi-atlas segmentation,
and region-of-interest (ROI) comparisons.

Documentation: https://scikit-rt.github.io/scikit-rt/
Code repository: https://github.com/scikit-rt/scikit-rt/
PyPI project page: https://pypi.org/project/scikit-rt/
"""
from importlib.metadata import version
import warnings

from skrt.better_viewer import BetterViewer
from skrt.better_viewer.options import set_viewer_options
from skrt.core import Defaults
from skrt.dose import Dose
from skrt.image import Image
from skrt.patient import Patient, Study
from skrt.registration import Registration
from skrt.simulation import SyntheticImage
from skrt.structures import ROI, StructureSet

# Suppress warning from pydicom.
warnings.filterwarnings(
    "ignore", message="Invalid value for VR UI", module="pydicom"
)
warnings.filterwarnings("ignore", message="The value length", module="pydicom")

# Assign version number.
__version__ = version("scikit-rt")
