"""Tools for performing image registration."""
import os
import re
import shutil
import subprocess

from skrt.image import Image, ImageComparison
from skrt.core import Data

_elastix_dir = None
_elastix = "elastix"
_transformix = "transformix"


class Registration(Data):

    def __init__(
        self,
        path,
        fixed=None,
        moving=None,
        pfiles=None,
        auto_reg=True,
        overwrite=False
    ):
        """Load data for an image registration and run the registration if
        auto_seg=True.

        Parameters
        ----------
        path : str
            Path to directory where the data for this image registration is
            stored. Can either be a non-existing directory, in which case 
            registration will be performed from scratch, or a directory already
            containing a fixed and moving image and optionally some registration
            steps.

        fixed : Image/str, default=None
            Image object representing the fixed image, or a source from which
            an Image object can be initialised. Must be set if the directory 
            at <path> does not already contain a fixed image file called
            "fixed.nii.gz".

        moving : Image/str, default=None
            Image object representing the fixed image, or a source from which
            an Image object can be initialised. Must be set if the directory 
            at <path> does not already contain a moving image file called
            "moving.nii.gz".
            
        pfiles : str/list/dict, default=None
            Path(s) to elastix parameter file(s) to be used in each step of the
            registration. If a list of more than one path, the parameter files 
            will be used to apply registrations in series, which the output of 
            each step being used as an initial transformation for the following 
            step. If None, the Registration will be initialised with no 
            registration steps and no registration will be performed. If a dict,
            the keys will be taken as names for the registration steps and 
            the values should be the path to the parameter file for that step;
            otherwise, the name of the step will be taken from the parameter
            filename.

        auto_reg : bool, default=True
            If True, the registration will be performed immediately.

        overwrite : bool, default=False
            If True and <path> already contains files, these will be deleted,
            meaning that no prior registration results will be loaded.
        """

        # Set up directory
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        elif overwrite:
            shutil.rmtree(path)
            os.mkdir(path)

        # Set up fixed and moving images
        self.fixed_path = os.path.join(self.path, "fixed.nii.gz")
        self.moving_path = os.path.join(self.path, "moving.nii.gz")
        if fixed is not None:
            self.set_fixed_image(fixed)
        if moving is not None:
            self.set_moving_image(moving)
        if fixed is None and moving is None:
            self._load_existing_input_images()

        # Set up registration steps
        self.steps = []
        self.steps_file = os.path.join(self.path, "registration_steps.txt")
        self.outdirs = {}
        self.pfiles = {}
        if isinstance(pfiles, str):
            pfiles = [pfiles]
        if pfiles is not None:
            self.add_pfiles(pfiles)
        else:
            self.load_pfiles()

    def set_image(self, im, category, force=True):
        """Assign a fixed or moving image.

        Parameters
        ----------
        im : Image/str
            The Image object to set, or a path that can be used to intialise
            an Image object.

        category : str
            Category of image ("fixed" or "moving").

        force : bool, default=True
            If True, the image file within self.path will be overwritten by
            this image even if it already exists.
        """

        if category not in ["fixed", "moving"]:
            raise RuntimeError(f"Unrecognised image category {category}; "
                               "should be either fixed or moving")

        if not isinstance(im, Image):
            im = Image(im)
        path = getattr(self, f"{category}_path")
        if not os.path.exists(path) or force:
            Image.write(im, path)
        setattr(self, f"{category}_image", Image(path))

    def set_fixed_image(self, im):
        """Assign a fixed image."""
        self.set_image(im, "fixed")

    def set_moving_image(self, im):
        """Assign a moving image."""
        self.set_image(im, "moving")

    def _load_existing_input_images(self):
        """Attempt to load images from fixed.nii.gz and moving.nii.gz from
        inside self.path. Print a warning if not found."""

        for category in ["fixed", "moving"]:
            path = getattr(self, f"{category}_path")
            if not os.path.exists(path):
                print(f"Warning: no {category} image found at {path}! "
                      f"Make sure you run Registration.set_{category}_image"
                      " before running a registration.")
                return
            self.set_image(path, category, force=False)

    def add_pfile(self, pfile, name=None, write_file=True):
        """Add a single parameter file to the list of registration steps.
        If write_file=True, the current registration steps file will be 
        overwritten."""

        # Infer name from parameter file name if name is None
        if name is None:
            name = os.path.basename(pfile).replace(".txt", "")

        # Check whether name already exists and add counter if so
        i = 1
        orig_name = name
        while name in self.steps:
            name = f"{orig_name}_{i}"
            i += 1

        # Add to list of registration steps
        self.steps.append(name)    
        
        # Make output directory, overwriting if it already exists
        outdir = os.path.join(self.path, name)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        self.outdirs[name] = outdir

        # Copy parameter file into outdir
        path = f"{outdir}/InputParameters.txt"
        shutil.copy(pfile, path)
        self.pfiles[name] = path

        # Rewrite steps directory
        if write_file:
            self.write_steps()

    def add_pfiles(self, pfiles):
        """Add multiple parameter files to the list of registration steps, 
        then write list of registration steps to a file."""

        for p in pfiles:
            if isinstance(pfiles, dict):
                name = p
                p = pfiles[p]
            else:
                name = None
            self.add_pfile(p)

        self.write_steps()

    def write_steps(self):
        """Write list of registration steps to a file at 
        self.path/registration_steps.txt."""

        with open(self.steps_file, "w") as file:
            for step in self.steps:
                file.write(step + "\n")

    def load_pfiles(self):
        """Attempt to load registration steps from a registration step 
        file."""

        # No steps file: don't load anything
        if not os.path.exists(self.steps_file):
            return

        # Load list of step names
        with open(self.steps_file) as file:
            steps = [line.strip() for line in file.readlines()]

        # Check each step name has an associated output directory and
        # parameter file
        self.steps = []
        self.outdirs = {}
        self.pfiles = {}
        for step in steps:
            outdir = os.path.join(self.path, step)
            if not os.path.exists(outdir):
                print(f"Warning: no output directory ({self.path}/{step}) "
                      f"found for registration step {step} listed in "
                      f"{self.steps_file}. This step will be ignored.")
                continue

            pfile = os.path.join(outdir, "InputParameters.txt")
            if not os.path.exists(pfile):
                print(f"Warning: no parameter file ({outdir}/InputParameters.txt) "
                      f"found for registration step {step} listed in "
                      f"{self.steps_file}. This step will be ignored.")
                continue
            
            self.steps.append(step)
            self.outdirs[step] = outdir
            self.pfiles[step] = pfile

    def clear(self):
        """Remove all registration steps and their output directories."""

        # Remove all subdirectories
        for d in os.listdir(self.path):
            full = os.path.join(self.path, d)
            if os.path.isdir(full):
                shutil.rmtree(full)

        # Remove registration steps file
        if os.path.exists(self.steps_file):
            os.remove(self.steps_file)

        # Reset attributes
        self.steps = []
        self.pfiles = {}
        self.outdirs = {}

    def view_init(self, **kwargs):
        """Interactively view initial fixed and moving images."""

        from skrt.better_viewer import BetterViewer

        kwargs.setdefault("hu", self.fixed_image.default_window)
        #  kwargs.setdefault("match_axes", "y")
        kwargs.setdefault("title", ["Fixed", "Moving"])
        kwargs.setdefault("comparison", True)
        BetterViewer([self.fixed_image, self.moving_image], **kwargs)


def set_elastix_dir(path):

    # Set directory
    global _elastix_dir
    _elastix_dir = path

    # Find elastix exectuable
    global _elastix
    global _transformix
    if os.path.exists(os.path.join(_elastix_dir, "bin/elastix")):
        _elastix = os.path.join(_elastix_dir, "bin/elastix")
        _transformix = os.path.join(_elastix_dir, "bin/transformix")
    elif os.path.exists(os.path.join(_elastix_dir, "elastix.exe")):
        _elastix = os.path.join(_elastix_dir, "elastix.exe")
        _transformix = os.path.join(_elastix_dir, "transformix.exe")
    else:
        raise RuntimeError(f"No elastix executable found in {_elastix_dir}!")
