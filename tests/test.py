import numpy as np
import os
import shutil
import pytest

from skrt.simulation import SyntheticImage
from skrt.reg2 import Registration


# Directory to store test registration data
reg_dir = "tmp/reg"
if os.path.exists(reg_dir):
    shutil.rmtree(reg_dir)
    
# Test images
sim1 = SyntheticImage((10, 12, 8))
sim1.add_cube(2, centre=(4, 4, 4))
im1 = sim1.get_image()
sim2 = SyntheticImage((11, 11, 11))
sim2.add_cube(2, centre=(6, 6, 6))
im2 = sim2.get_image()


reg = Registration("tmp/reg", sim1, sim2, pfiles=["pfiles/MI_Translation.txt"])
reg.register()

sim2.add_cube(4, name="cube")
roi = sim2.get_roi("cube")
roi2 = reg.transform_roi(roi)
                                                  
