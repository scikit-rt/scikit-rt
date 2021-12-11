"""Tools for performing image registration."""
import os
import re
import shutil
import subprocess

from skrt.image import Image
from skrt.structures import ROI, StructureSet
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
        auto=False,
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

        auto : bool, default=True
            If True, the registration will be performed immediately for all 
            steps.

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
            self.load_existing_input_images()

        # Set up registration steps
        self.steps = []
        self.steps_file = os.path.join(self.path, "registration_steps.txt")
        self.outdirs = {}
        self.pfiles = {}
        self.tfiles = {}
        self.transformed_images = {}
        if isinstance(pfiles, str):
            pfiles = [pfiles]
        if pfiles is not None:
            self.add_pfiles(pfiles)
        else:
            self.load_pfiles()

        # Perform registration
        if auto:
            self.register()

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

    def load_existing_input_images(self):
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
        self.tfiles = {}
        self.transformed_images = {}
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

            # Check for transform parameter files
            tfile = os.path.join(outdir, "TransformParameters.0.txt")
            if os.path.exists(tfile):
                self.tfiles[step] = tfile

            # Check for transformed moving images
            im_path = os.path.join(outdir, "result.0.nii")
            if os.path.exists(im_path):
                self.transformed_images[step] = Image(
                    im_path, title="Transformed moving")

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
        self.tfiles = {}
        self.transformed_images = {}

    def get_elastix_cmd(self, step, use_previous_tfile=True):
        """Get elastix registration command for a given step."""

        # Get step number
        if isinstance(step, int):
            i = step
            step = self.steps[step]
        else:
            i = self.steps.index(step)

        # Check for input transform file
        tfile = None
        if use_previous_tfile and i > 0:
            prev_step = self.steps[i - 1]
            if prev_step not in self.tfiles:
                print(f"Warning: previous step {prev_step} has not yet "
                      f"been performed! Input transform file for step {step}"
                      " will not be used.")
            else:
                tfile = self.tfiles[prev_step]

        # Construct command
        cmd = (
            f"{_elastix} -f {self.fixed_path} -m {self.moving_path} "
            f"-p {self.pfiles[step]} -out {self.outdirs[step]}"
        )
        if tfile is not None:
            cmd += f" -t0 {tfile}"

        # Replace any backslashes
        cmd = cmd.replace("\\", "/")

        return cmd

    def register(self, step=None, force=False, use_previous_tfile=True):
        """Run a registration. By default the registration will be run for
        all steps in self.steps, but can optionally be run for just one step
        by setting <step> to a step name or number. Note that if 
        use_previous_tfile=True, any prior steps that have not yet been
        run will be run.

        Parameters
        ----------
        step : int/str/list, default=None
            Name, number, or list of the step(s) for which the registration 
            should be performed. By default, registration will be performed 
            for all steps in series, using the previous steps's output 
            transform as input for the next. Available steps are listed in 
            self.steps.

        force : bool, default=None
            If False and a registration has already been performed for the 
            given step(s), the registration will not be re-run.

        use_previous_tfile : bool, default=True
            If True, each step will use the transform file from the previous 
            step as an initial transform.
        """

        # Make list of steps to run
        if step is None:
            steps_input = self.steps
        elif not isinstance(step, list):
            steps_input = [step]
        else:
            steps_input = step

        # Check all steps exist and convert numbers to names
        steps = []
        for step in steps_input:
            if isinstance(step, int):
                if step >= len(self.steps):
                    raise RuntimeError(f"Step {step} not found!")
                step = self.steps[step]
            if step not in self.steps:
                raise RuntimeError(f"Step {step} not found!")
            steps.append(step)

        # Run registration for each step
        for step in steps:
            self.register_step(step, force=True, use_previous_tfile=True)

    def register_step(self, step, force=False, use_previous_tfile=True):
        """Run a single registration step. Note that if use_previous_tfile=True, 
        any prior steps that have not yet been run will be run.

        Parameters
        ----------
        step : int/str/list, default=None
            Name or number of the step for which the registration should be 
            performed. Available steps are listed in self.steps.

        force : bool, default=None
            If False and a registration has already been performed for the 
            given step, the registration will not be re-run. Note that setting
            force=True will only force rerunning of the chosen step, not of
            any preceding steps needed for the input transform file. To enforce
            rerunning of multiple steps, call self.register(step, force=True) 
            where <steps> is a list of steps to run.

        use_previous_tfile : bool, default=True
            If True, this step will use the transform file from the previous 
            step as an initial transform, unless it is the first step. Note
            that this will cause any prior steps that have not yet been run
            to be run.
        """

        # Check if the registration has already been performed
        if self.already_performed(step) and not force:
            return

        # Check that previous step has been run if needed
        if isinstance(step, int):
            i = step
            step = self.steps[i]
        else:
            i = self.steps.index(step)
        if use_previous_tfile and i > 0:
            if not self.already_performed(i - 1):
                self.register(i, use_previous_tfile=True)

        # Run
        cmd = self.get_elastix_cmd(step, use_previous_tfile)
        print("Running command:\n", cmd)
        code = subprocess.run(cmd.split()).returncode

        # Check whether registration succeeded
        if code:
            logfile = os.path.join(self.outdirs[step], "elastix.log")
            print(f"Warning: registration step {step} failed! See "
                  f"{logfile} for more info.")
        else:
            self.tfiles[step] = os.path.join(self.outdirs[step], 
                                             "TransformParameters.0.txt")
            self.transformed_images[step] = Image(
                os.path.join(self.outdirs[step], "result.0.nii"),
                title="Transformed moving"
            )

    def already_performed(self, step):
        """Check whether a registration step has already performed (i.e. has
        a valid output transform file)."""

        if isinstance(step, int):
            step = self.steps[step]
        return step in self.tfiles and os.path.exists(self.tfiles[step])

    def transform_image(self, im, step=-1, outfile=None, params=None):
        """Transform an image using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps
        will be run. Either return the transformed image or write it to 
        a file.

        Parameters
        ----------
        im : Image/str
            Image to be transformed. Can be either an Image object or a 
            path that can be used to initialise an Image object.

        step : int/str, default=-1
            Name or number of step for which to apply the transform; by 
            default, the final step will be used.

        outfile : str, default=None
            If None, the transformed Image object will be returned; otherwise,
            it will be written to the path specified in <outfile>.

        params : dict, default=None
            Optional list of parameters to temporarily overwrite in the 
            transform file before applying the transform. Should be a dict,
            where keys are parameter names and desired values are values.
            Note that strings in the parameter file, need to include quotes, 
            so you will need to use double quotes.
        """

        # Save image temporarily as nifti if needed
        im = Image(im)
        self.make_tmp_dir()
        if im.source_type == "nifti":
            im_path = im.path
        else:
            im_path = os.path.join(self._tmp_dir, "image.nii.gz")
            im.write(im_path, verbose=False)

        # Transform the nifti file
        result_path = self.transform_nifti(im_path, step, params)
        if result_path is None:
            return

        # Copy to output dir if outfile is set
        if outfile is not None:
            shutil.copy(result_path, outfile)
            return

        # Otherwise, return Image object
        im = Image(result_path)
        self.rm_tmp_dir()
        return im

    def transform_nifti(self, path, step=-1, params=None):
        """Transform a nifti file at a given path for a given step, ensuring
        that the step has been run. Return the path to the transformed file
        inside self._tmp_dir."""

        if isinstance(step, int):
            i = step
            step = self.steps[i]
        else:
            i = self.steps.index(step)

        # Check registration has been performed, and run it if not
        if not self.already_performed(step):
            self.register(self.steps[:i + 1])

        # Make temporary modified parameter file if needed
        self.make_tmp_dir()
        if params is None:
            tfile = self.tfiles[step]
        else:
            tfile = os.path.join(self._tmp_dir, "TransformParameters.txt")
            adjust_parameters(self.tfiles[step], tfile, params)

        # Run transformix
        cmd = (
            f"{_transformix} -in {path} -out {self._tmp_dir} "
            f"-tp {tfile}"
        )
        cmd = cmd.replace("\\", "/")
        print("running command:", cmd)
        code = subprocess.run(cmd.split()).returncode

        # If command failed, move log out from temporary dir
        if code:
            logfile = os.path.join(self.path, "transformix.log")
            if os.path.exists(logfile):
                os.remove(logfile)
            shutil.move(os.path.join(self._tmp_dir, "transformix.log"),
                        self.path)
            print(f"Warning: image transformation failed! See "
                  f"{logfile} for more info.")
            return

        # Return path to result
        return os.path.join(self._tmp_dir, "result.nii")

    def transform_roi(self, roi, step=-1, outfile=None, params=None):
        """Transform a single ROI using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps 
        will be run. Either return the transformed ROI or write it to a file.

        Parameters
        ----------
        roi : ROI/str
            ROI to be transformed. Can be either an ROI object or a 
            path that can be used to initialise an ROI object.

        step : int/str, default=-1
            Name or number of step for which to apply the transform; by 
            default, the final step will be used.

        outfile : str, default=None
            If None, the transformed ROI object will be returned; otherwise,
            it will be written to the path specified in <outfile>.

        params : dict, default=None
            Optional list of parameters to temporarily overwrite in the 
            transform file before applying the transform. Should be a dict,
            where keys are parameter names and desired values are values.
            Note that strings in the parameter file, need to include quotes, 
            so you will need to use double quotes.
            By default, "ResampleInterpolator" will be set to 
            "FinalNearestNeighborInterpolator".
        """

        # Save ROI temporarily as nifti if needed
        roi = ROI(roi)
        roi.load()
        self.make_tmp_dir()
        if roi.source_type == "mask" and roi.mask.source_type == "nifti":
            roi_path = roi.mask.path
        else:
            roi_path = os.path.join(self._tmp_dir, f"{roi.name}.nii.gz")
            roi.write(roi_path, verbose=False)

        # Set default parameters
        default_params = {
            "ResampleInterpolator": '"FinalNearestNeighborInterpolator"'
        }
        if params is not None:
            default_params.update(params)

        # Transform the nifti file
        result_path = self.transform_nifti(roi_path, step, default_params)
        if result_path is None:
            return result_path

        # Copy to output dir if outfile is set
        if outfile is not None:
            shutil.copy(result_path, outfile)
            return

        # Otherwise, return ROI object
        roi = ROI(result_path, name=roi.name, color=roi.color)
        self.rm_tmp_dir()
        roi.set_image(self.get_transformed_image(step))
        return roi

    def transform_structure_set(self, structure_set, step=-1, outfile=None,
                                params=None):
        """Transform a structure set using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps 
        will be run. Either return the transformed ROI or write it to a file.

        Parameters
        ----------
        structure_set : StructureSet/str
            StructureSet to be transformed. Can be either a StructureSet 
            object or a path that can be used to initialise a StructureSet.

        step : int/str, default=-1
            Name or number of step for which to apply the transform; by 
            default, the final step will be used.

        outfile : str, default=None
            If None, the transformed StructureSet object will be returned; 
            otherwise, it will be written to the path specified in <outfile>.

        params : dict, default=None
            Optional list of parameters to temporarily overwrite in the 
            transform file before applying the transform. Should be a dict,
            where keys are parameter names and desired values are values.
            Note that strings in the parameter file, need to include quotes, 
            so you will need to use double quotes.
            By default, "ResampleInterpolator" will be set to 
            "FinalNearestNeighborInterpolator".
        """

        final = StructureSet()
        for roi in structure_set:
            transformed_roi = self.transform_roi(roi, step, params=params)
            if transformed_roi is not None:
                final.add_roi(transformed_roi)

        # Write structure set if outname is given
        if outfile is not None:
            final.write(outname)
            return

        # Otherwise, return structure set
        final.set_image(self.get_transformed_image(step))
        return final

    def get_transformed_image(self, step=-1, force=False):
        """Get the transformed moving image for a given step, by default the
        final step. If force=True, the registration will be re-run for that
        step."""

        if isinstance(step, int):
            step = self.steps[step]
        if not self.already_performed(step):
            self.register(step, force=force)
        if step in self.transformed_images:
            return self.transformed_images[step]

    def make_tmp_dir(self):
        """Create temporary directory."""

        self._tmp_dir = os.path.join(self.path, ".tmp")
        if not os.path.exists(self._tmp_dir):
            os.mkdir(self._tmp_dir)

    def rm_tmp_dir(self):
        """Delete temporary directory and its contents."""

        if not hasattr(self, "_tmp_dir"):
            return
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    def view_init(self, **kwargs):
        """Interactively view initial fixed and moving images."""

        from skrt.better_viewer import BetterViewer

        kwargs.setdefault("hu", self.fixed_image.default_window)
        #  kwargs.setdefault("match_axes", "y")
        kwargs.setdefault("title", ["Fixed", "Moving"])
        kwargs.setdefault("comparison", True)
        BetterViewer([self.fixed_image, self.moving_image], **kwargs)

    def view_result(self, step=-1, compare_with_fixed=True, **kwargs):
        """Interactively view transformed image, optionally side-by-side
        with fixed image.

        Parameters
        ----------
        step : int/str, default=-1
            Name or number of step for which to view the result. By default,
            the result of the final step will be shown.

        compare_with_fixed : bool, default=True
            If True, the result will be displayed in comparison with the
            fixed image.

        **kwargs :
            Optional keyword arguments to pass to BetterViewer.
        """

        from skrt.better_viewer import BetterViewer

        if isinstance(step, int):
            step = self.steps[step]
        if step not in self.transformed_images:
            self.register(step)
        if compare_with_fixed:
            ims = [self.fixed_image, self.transformed_images[step]]
            kwargs.setdefault("comparison", True)
            kwargs.setdefault("hu", self.fixed_image.default_window)
            kwargs.setdefault("title", ["Fixed", "Transformed moving"])
        else:
            ims = self.transformed_images[step]
            kwargs.setdefault("title", "Transformed moving")
        BetterViewer(ims, **kwargs)


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


def adjust_parameters(infile, outfile, params):
    """Open an elastix parameter file (works for both input parameter and 
    output transform files), adjust its parameters, and save it to a new 
    file.

    Parameters
    ----------
    file : str
        Path to an elastix parameter file.

    outfile : str
        Path to output file.

    params : dict
        Dictionary of parameter names and new values.
    """

    # Read input
    with open(infile) as file:
        txt = file.read()

    # Modify
    for name, value in params.items():
        txt = re.sub(fr"\({name}.*\)", fr"({name} {value})", txt)

    # Write to output
    with open(outfile, "w") as file:
        file.write(txt)
