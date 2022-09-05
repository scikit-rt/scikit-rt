"""Tools for performing image registration."""
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import numpy as np
import os
from pathlib import Path
from pkg_resources import resource_filename
import shutil
import subprocess

import skrt.image
from skrt.structures import ROI, StructureSet
from skrt.core import (fullpath, get_logger, Data, is_list, to_list,
        Defaults, PathData)
from skrt.dose import ImageOverlay, Dose
from skrt.simulation import make_grid

_ELASTIX = "elastix"
_transformix = "transformix"


class Registration(Data):

    def __init__(
        self, path, fixed=None, moving=None, fixed_mask=None,
        moving_mask=None, pfiles=None, auto=False, overwrite=False,
        tfiles=None, initial_alignment=None, initial_transform_name=None,
        capture_output=False, log_level=None):
        """Load data for an image registration and run the registration if
        auto_seg=True.

        **Parameters:**

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

        fixed_mask : Image/str, default=None
            Image object representing a mask to be applied to the fixed image,
            or a source from which an Image object can be initialised. If
            None and an image file 'fixed_mask.nii.gz' exists at <path>
            then this will be used.  Setting a mask is optional.

        moving_mask : Image/str, default=None
            Image object representing a mask to be applied to the moving image,
            or a source from which an Image object can be initialised. If
            None and an image file 'moving_mask.nii.gz' exists at <path>
            then this will be used.  Setting a mask is optional.

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

        tfiles : dict, default=None
            Dictionary of pre-defined transforms, where a key is a
            registration step and the associated value is the path to
            a pre-defined registration transform.  Transformations
            are performed, in the order given in the dictionary,
            before any registration step is performed.

        initial_alignment : tuple/dict/str, default=None
            Alignment to be performed before any registration steps
            are run.  This can be any of the following:
            - a tuple indicating the amounts (dx, dy, dz) by which
              a point in the fixed image must be translated to align
              with the corresponding point in the moving image;
            - a dictionary specifying parameters and values to be passed
              to skrt.image.get_translation_to_align(), which defines
              a translation for aligning fixed and moving image;
            - one of the strings "_top_", "_centre_", "_bottom_", in which case
              skrt.image.get_translation_to align() is called to define
              a translationg such that fixed and moving image have
              their (x, y) centres aligned, and have z positions aligned at
              image top, centre or bottom;
            - a string specifying the name of an ROI associated with both
              fixed and moving image, with ROI centroids to be aligned;
            - a tuple of strings specifying names of an ROI associated with
              the fixed image and an ROI associated with the moving image,
              with ROI centroids to be aligned;
            - a tuple specifying ROI name and relative position along z-axis
              on which to align (0 most-inferior slice of ROI, 1 most-superior
              slice), with the same ROI name and relative position used for
              fixed and moving image;
            - a tuple of tuples specifying ROI name and relative position
              along z-axis on which to align, with one tuple for the fixed
              image and one tuple for the moving image.
            The result of the initial alignment is stored as the first
            entry of <self.tfiles", with key <initial_alignment_name>.  If
            <initial_alignment> is set to None, no initial alignment
            is performed.

        initial_transform_name : str, default=None
            Name to be used in registration steps for transform corresponding
            to <initial_alignment>.  If None, the name 'initial_alignment'
            is used.

        capture_output : bool, default=False
            If True, capture to stdout messages from performing
            registration and transformation.  This suppresses
            informational messages to screen, but doesn't suppress error
            messages.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        """

        # Set up event logging and output capture
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)
        self.capture_output = capture_output

        # Set up directory
        self.path = fullpath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        elif overwrite:
            shutil.rmtree(path)
            os.mkdir(path)

        # Set up fixed and moving images and optional masks
        self.fixed_path = os.path.join(self.path, "fixed.nii.gz")
        self.moving_path = os.path.join(self.path, "moving.nii.gz")
        self.fixed_mask_path = os.path.join(self.path, "fixed_mask.nii.gz")
        self.moving_mask_path = os.path.join(self.path, "moving_mask.nii.gz")
        self.fixed_source = fixed
        self.moving_source = moving
        im_fixed = None
        im_moving = None
        if fixed is not None:
            im_fixed = self.set_fixed_image(fixed)
            if fixed_mask is not None:
                self.set_fixed_mask(fixed_mask)
        if moving is not None:
            im_moving = self.set_moving_image(moving)
            if moving_mask is not None:
                self.set_moving_mask(moving_mask)
        if fixed is None and moving is None:
            self.load_existing_input_images()

        # Set initial transform corresponding to initial alignment.
        if im_fixed and im_moving and initial_alignment:

            initial_translation = im_fixed.get_alignment_translation(
                    im_moving, initial_alignment)

            if initial_translation:
                initial_transform_name = (initial_transform_name
                        or "initial_alignment")
                tfiles = tfiles or {}
                tfiles = {initial_transform_name: initial_translation, **tfiles}

        # Set up registration steps
        self.steps = []
        self.steps_file = os.path.join(self.path, "registration_steps.txt")
        self.outdirs = {}
        self.pfiles = {}
        self.tfiles = {}
        self.transformed_images = {}
        self.jacobians = {}
        self.deformation_fields = {}
        self.file_types = {'t': tfiles, 'p': pfiles}
        for ftype, files in self.file_types.items():
            if isinstance(files, (str, Path)):
                files = [str(files)]
            if files is not None:
                self.add_files(files, ftype)

        if not self.outdirs:
            # No parameter files or transform files defined:
            # try to load registration steps from a step file.
            self.load_files()
        else:
            # Link any pre-existing transform files for registration steps.
            for step, outdir in self.outdirs.items():
                path = f"{outdir}/TransformParameters.0.txt"
                if step in self.pfiles and os.path.exists(path):
                    self.tfiles[step] = path

        self.moving_grid_path = os.path.join(self.path, "moving_grid.nii.gz")
        self.transformed_grids = {}

        # Perform registration
        if auto:
            self.register()

    def set_image(self, im, category, force=True):
        """Assign a fixed or moving image.

        **Parameters:**

        im : Image/str
            The Image object to set, or a path that can be used to intialise
            an Image object.

        category : str
            Category of image: "fixed", "moving", "fixed_mask", "moving_mask",
            "moving_grid".

        force : bool, default=True
            If True, the image file within self.path will be overwritten by
            this image even if it already exists.
        """

        categories = ["fixed", "moving", "fixed_mask", "moving_mask",
                "moving_grid"]
        if category not in categories:
            raise RuntimeError(
                f"Unrecognised image category {category}; "
                f"should be one of{categories}"
            )

        if not isinstance(im, skrt.image.Image):
            im = skrt.image.Image(im)
        path = getattr(self, f"{category}_path")
        if not os.path.exists(path) or force:
            skrt.image.Image.write(im, path, verbose=(self.logger.level < 30))
        if 'grid' in category or 'mask' in category:
            setattr(self, f"{category}", skrt.image.Image(path))
        else:
            setattr(self, f"{category}_image", skrt.image.Image(path))

        return im

    def set_fixed_image(self, im):
        """Assign a fixed image."""
        return self.set_image(im, "fixed")
        

    def set_fixed_mask(self, im):
        """Assign a fixed-image mask."""
        return self.set_image(im, "fixed_mask")

    def set_moving_image(self, im):
        """Assign a moving image."""
        return self.set_image(im, "moving")

    def set_moving_mask(self, im):
        """Assign a moving-image mask."""
        return self.set_image(im, "moving_mask")

    def set_moving_grid(self, im):
        """Assign a grid in the reference system of the moving image."""
        return self.set_image(im, "moving_grid")

    def load_existing_input_images(self):
        """Attempt to load images from fixed.nii.gz and moving.nii.gz from
        inside self.path. Print a warning if not found."""

        for category in ["fixed", "moving"]:
            path = getattr(self, f"{category}_path")
            if not os.path.exists(path):
                self.logger.warning(
                    f"No {category} image found at {path}! "
                    f"Make sure you run Registration.set_{category}_image"
                    " before running a registration."
                )
                return
            self.set_image(path, category, force=False)

    def define_translation(self, dxdydz):
        """
        Define Elastix translation parameters from 3-element tuple or list.

        **Parameter:**

        dxdydz : tuple/list
            Three-element tuple or list, giving translation in order (x, y, z).
        """
        # Define translation parameters that are enough to allow
        # application before a registration step.
        translation = {
                "Transform": "TranslationTransform",
                "NumberOfParameters": 3,
                "TransformParameters": dxdydz,
                "InitialTransformParametersFileName": "NoInitialTransform",
                "UseBinaryFormatForTransformationParameters": False,
                "HowToCombineTransforms": "Compose"
                }

        if hasattr(self, "fixed_image"):
            # Add parameters needed to allow warping of the moving image.
            translation.update(get_image_transform_parameters(self.fixed_image))

        return translation

    def add_file(self, file, name=None, params=None, ftype="p"):
        """Add a single file of type <ftype> to the list of registration steps.
        This file can optionally be modified by providing a dict
        of parameter names and new values in <params>.

        **Parameters:**

        file : str/dict/tuple/list
            Path to the elastix parameter file to copy into the registration
            directory. Can also be a dict containing parameters and values,
            which will be used to create a parameter file from scratch. In this
            case, the <name> argument must also be provided.  For a transform
            file (<ftype> of 't'), can also be a three element tuple or list,
            specifying the x, y, z components of a translation.

        name : str, default=None
            Name for this step of the registration. If None, a name will
            be taken from the parameter file name.

        params : dict, default=None
            Dictionary of parameter names and replacement values with which
            the input parameter file will be modified.

        ftype : str, default='p'
            String indicating whether the file to be added is a parameter
            file ('p') or a transform file ('t').
        """
        # Check that file type is recognised.
        if not ftype.lower() in self.file_types:
            self.logger.warning(f"File ignored: '{file}'")
            self.logger.warning(f"File type not recognised: '{ftype}'")
            self.logger.warning(f"\nValid file types are: '{self.file_types}'")
            return
        ftype = ftype.lower()

        # Only allow file to be passed as a list or tuple
        # if it has exactly three elements and is for a transform file.
        if isinstance(file, (tuple, list)):
            if len(file) != 3:
                self.logger.warning("Input for file: '{file}'")
                self.logger.warning("Translation passed as list or tuple "
                        "must contain exactly 3 elements")
                return
            if ftype != 't':
                self.logger.warning(f"Translation may only be passed as list "
                        "or tuple for transform")
                return

        # If name is null, infer from file name.
        if not name:
            if isinstance(file, (dict, tuple, list)):
                self.logger.warning(
                        "If passing parameters from dict, tuple or list, "
                        "<name> must be set.")
                return
            name = Path(file).stem

        # Create dictionary for translation from list or tuple
        if isinstance(file, (tuple, list)):
            file = self.define_translation(file)

        # Check whether name already exists and add counter if so
        i = 1
        orig_name = name
        while name in self.steps:
            name = f"{orig_name}_{i}"
            i += 1

        # Add to list of registration steps
        self.steps.append(name)

        # Ensure that output directory exists.
        outdir = os.path.join(self.path, name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outdirs[name] = outdir

        # Define and store file path in step output directory.
        if "p" == ftype.lower():
            path = f"{outdir}/InputParameters.txt"
            self.pfiles[name] = path
        elif "t" == ftype.lower():
            path = f"{outdir}/TransformParameters.0.txt"
            self.tfiles[name] = path

        # Create new file, or copy existing file.
        if isinstance(file, dict):
            Path(path).touch()
            self.adjust_file(name, file, ftype)
        else:
            shutil.copy(file, path)

        # Modify the file if custom parameters are given
        if params is not None:
            self.adjust_file(name, params, ftype)

        # Rewrite text file containing list of steps
        self.write_steps()

    def add_files(self, files, ftype='p'):
        """Add multiple files of type <ftype> to the list of registration steps,
        then write list of registration steps to a file.

        The valid values for <ftype? are:
        - 'p': parameter file;
        - 't': transform file.
        """

        for f in files:
            if isinstance(files, dict):
                name = f
                f = files[f]
            else:
                name = None
            self.add_file(f, name=name, ftype=ftype)

        self.write_steps()

    def list_default_pfiles(self):
        """
        List the available default parameter files. Note that this
        does not affect the current Registration object and is only for
        informative purposes.
        """

        for file in get_default_pfiles():
            print(file.replace(".txt", ""))

    def get_default_params(self, filename):
        """
        Get the contents of a default parameter file as a dict. Note that this
        does not affect the current Registration object and is only for
        informative purposes.
        """

        files = get_default_pfiles()
        if not filename.endswith(".txt"):
            filename += ".txt"
        if filename not in files:
            self.logger.warning(
                    f"Default file {name} not found. Available files:")
            self.list_default_pfiles(self)
            return
        full_files = get_default_pfiles(False)
        pfile = full_files[files.index(filename)]
        return read_parameters(pfile)

    def add_default_pfile(self, filename, params=None):
        """
        Add a default parameter file. For options, run
        Registration.list_default_pfiles(). You can also inspect the contents
        of a default parameter file by running get_default_params(name).

        **Parameters:**

        filename : str
            Name of the default parameter file to add (with or without
            the .txt extension).

        params : dict, default=None
            Dictionary of parameter names and replacement values with which
            the input parameter file will be modified.
        """

        files = get_default_pfiles()
        if not filename.endswith(".txt"):
            filename += ".txt"
        if filename not in files:
            self.logger.warning(
                    f"Default file {name} not found. Available files:")
            self.list_default_pfiles(self)
            return

        full_files = get_default_pfiles(False)
        pfile = str(full_files[files.index(filename)])
        self.add_file(pfile, params=params, ftype="p")

    def write_steps(self):
        """Write list of registration steps to a file at
        self.path/registration_steps.txt."""

        with open(self.steps_file, "w") as file:
            for step in self.steps:
                file.write(step + "\n")

    def load_files(self):
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
                self.logger.warning(
                    f"No output directory ({self.path}/{step}) "
                    f"found for registration step {step} listed in "
                    f"{self.steps_file}. This step will be ignored."
                )
                continue

            pfile = os.path.join(outdir, "InputParameters.txt")
            tfile = os.path.join(outdir, "TransformParameters.0.txt")
            if not os.path.exists(pfile) and not os.path.exists(tfile):
                self.logger.warning(
                    f"No parameter file ({pfile}) "
                    f"and no transform file ({tfile}) "
                    f"found for registration step {step} listed in "
                    f"{self.steps_file}. This step will be ignored."
                )
                continue

            # Add step to the list.
            self.steps.append(step)
            self.outdirs[step] = outdir
            if os.path.exists(pfile):
                self.pfiles[step] = pfile
            if os.path.exists(tfile):
                self.tfiles[step] = tfile

            # Check for transformed moving images
            im_path = os.path.join(outdir, "result.0.nii")
            if os.path.exists(im_path):
                self.transformed_images[step] = skrt.image.Image(
                    im_path, title="Transformed moving"
                )

            # Check for Jacobian determinant
            jac_path = os.path.join(outdir, "spatialJacobian.nii")
            if os.path.exists(jac_path):
                self.jacobians[step] = Jacobian(
                    jac_path, title="Jacobian determinant",
                    image=self.transformed_images[step]
                )

            # Check for deformation field
            df_path = os.path.join(outdir, "deformationField.nii")
            if os.path.exists(df_path):
                self.deformation_fields[step] = DeformationField(
                    df_path, title="Deformation field",
                    image=self.transformed_images[step]
                )

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

    def get_ELASTIX_cmd(self, step, use_previous_tfile=True):
        """Get elastix registration command for a given step."""

        # Get step number
        i = self.get_step_number(step)
        step = self.get_step_name(step)

        # Check for input transform file
        tfile = None
        if use_previous_tfile and i > 0:
            prev_step = self.steps[i - 1]
            if prev_step not in self.tfiles:
                self.logger.warning(
                    f"Previous step {prev_step} has not yet "
                    f"been performed! Input transform file for step {step}"
                    " will not be used."
                )
            else:
                tfile = self.tfiles[prev_step]

        # Construct command
        cmd = [
            _ELASTIX,
            '-f', self.fixed_path.replace("\\", "/"),
            '-m', self.moving_path.replace("\\", "/"),
            ]
        if os.path.exists(self.fixed_mask_path):
            cmd.extend(['-fMask', self.fixed_mask_path.replace("\\", "/"),])
        if os.path.exists(self.moving_mask_path):
            cmd.extend(['-mMask', self.moving_mask_path.replace("\\", "/"),])
        cmd.extend([
            "-p", self.pfiles[step].replace("\\", "/"),
            '-out', self.outdirs[step].replace("\\", "/")
            ])
        if tfile is not None:
            cmd.extend(['-t0', tfile.replace("\\", "/")])

        return cmd

    def register(self, step=None, force=False, use_previous_tfile=True,
            ensure_transformed_moving=True):
        """Run a registration. By default the registration will be run for
        all steps in self.steps, but can optionally be run for just one step
        by setting <step> to a step name or number. Note that if
        use_previous_tfile=True, any prior steps that have not yet been
        run will be run.

        **Parameters:**

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

        ensure_transformed_moving : bool, default=True
            Ensure that a transformed moving image is created for each
            registration step.  This may mean forcing the image creation
            for a step with a pre-defined transform file.
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
            steps.append(self.get_step_name(step))

        # Run registration for each step
        for step in steps:
            self.register_step(step, force=force,
                    use_previous_tfile=use_previous_tfile,
                    ensure_transformed_moving=ensure_transformed_moving)

    def register_step(self, step, force=False, use_previous_tfile=True,
            ensure_transformed_moving=True):
        """Run a single registration step, if it has a parameter file
        assigned. Note that if use_previous_tfile=True,
        any prior steps that have not yet been run will be run.

        **Parameters:**

        step : int/str/list, default=None
            Name or number of the step for which the registration should be
            performed. Available steps are listed in self.steps.

        force : bool, default=None
            If False and a registration has already been performed for the
            given step, the registration will not be re-run. Ignored
            if the step has no parameter file assigned (in which case
            the step must have a pre-defined transform file).  Note that setting
            force=True will only force rerunning of the chosen step, not of
            any preceding steps needed for the input transform file. To enforce
            rerunning of multiple steps, call self.register(step, force=True)
            where <steps> is a list of steps to run.

        use_previous_tfile : bool, default=True
            If True, this step will use the transform file from the previous
            step as an initial transform, unless it is the first step. Note
            that this will cause any prior steps that have not yet been run
            to be run.

        ensure_transformed_moving : bool, default=True
            Ensure that a transformed moving image is created for the
            registration step.  This may mean forcing the image creation
            for a step with a pre-defined transform file.
        """
        global _ELASTIX

        # Check if the registration has already been performed
        if self.is_registered(step) and not force:
            if ensure_transformed_moving:
                self.transform_moving_image(step)
            return
        
        # Obtain step index and name.
        i = self.get_step_number(step)
        step = self.get_step_name(step)

        # Check that step has pfile assigned (i.e. it can be run/rerun).
        if not self.pfiles[step]:
            return

        # Check that previous step has been run if needed
        if use_previous_tfile and i > 0:
            if not self.is_registered(i - 1):
                self.register(i - 1, use_previous_tfile=True)

        # Run
        cmd = self.get_ELASTIX_cmd(step, use_previous_tfile)
        self.logger.info(f"Running command:\n {' '.join(cmd)}")
        code = subprocess.run(
                cmd, capture_output=self.capture_output).returncode

        # Check whether registration succeeded
        if code:
            logfile = os.path.join(self.outdirs[step], "elastix.log")
            self.logger.warning(
                f"Registration step {step} failed! See "
                f"{logfile} or run Registration.print_log({step}) for "
                " more info."
            )
        else:
            self.tfiles[step] = os.path.join(
                self.outdirs[step], "TransformParameters.0.txt"
            )
            self.transformed_images[step] = skrt.image.Image(
                os.path.join(self.outdirs[step], "result.0.nii"),
                title="Transformed moving",
            )

    def is_registered(self, step):
        """Check whether a registration step has already been performed (i.e. 
        has a valid output transform file)."""

        step = self.get_step_name(step)
        return step in self.tfiles and os.path.exists(self.tfiles[step])

    def print_log(self, step=-1):
        """Print elastix output log for a given step (by default, the 
        last step)."""

        step = self.get_step_name(step)
        logfile = os.path.join(self.outdirs[step], "elastix.log")
        if not os.path.exists(logfile):
            print(f"No log found - try running registration step {step}.")
            return

        with open(logfile) as file:
            for line in file.readlines():
                print(line)

    def transform(self, to_transform, **kwargs):
        """
        Call one of the transform functions, depending on the type of
        `to_transform`. Functions called depending on type are:

        Image:
            transform_image(to_transform, `**`kwargs)
        str:
            transform_data(to_transform, `**`kwargs)
        StructureSet:
            transform_structure_set(to_transform, `**`kwargs)
        ROI:
            transform_roi(to_transform, `**`kwargs)
        """

        if issubclass(type(to_transform), skrt.image.Image):
            return self.transform_image(to_transform, **kwargs)
        elif isinstance(to_transform, str):
            return self.transform_data(to_transform, **kwargs)
        elif isinstance(to_transform, ROI):
            return self.transform_roi(to_transform, **kwargs)
        elif isinstance(to_transform, StructureSet):
            return self.transform_structure_set(to_transform, **kwargs)
        else:
            self.logger.warning(
                    f"Unrecognised transform input type {type(to_transform)}")

    def transform_image(self, im, step=-1, outfile=None, params=None, rois=None,
            ):
        """
        Transform an image using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps
        will be run. Either return the transformed image or write it to
        a file.

        **Parameters:**

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

        # If image is a Dose object, set pixel type to float
        if params is None:
            params = {}
        is_dose = isinstance(im, Dose)
        if is_dose:
            params["ResultImagePixelType"] = "float"

        # Save image temporarily as nifti if needed
        im = skrt.image.Image(im)
        self.make_tmp_dir()
        if im.source_type == "nifti":
            im_path = im.path
        else:
            im_path = os.path.join(self._tmp_dir, "image.nii.gz")
            im.write(im_path, verbose=(self.logger.level < 30))

        # Transform the nifti file
        result_path = self.transform_data(im_path, step, params)
        if result_path is None:
            return

        # Copy to output dir if outfile is set
        if outfile is not None:
            shutil.copy(result_path, outfile)
            return

        # Otherwise, return Image object
        if is_dose:
            final_im = Dose(result_path)
        else:
            final_im = skrt.image.Image(result_path)

            # Transform structure sets
            if rois == "all":
                rois_to_transform = im.structure_sets
            elif isinstance(rois, int):
                rois_to_transform = [im.structure_sets[rois]]
            else:
                rois_to_transform = []
            for ss in rois_to_transform:
                ss2 = self.transform_structure_set(ss, step=step)
                final_im.add_structure_set(ss2)

        self.rm_tmp_dir()
        return final_im

    def transform_data(self, path, step=-1, params=None):
        """Transform a nifti file or point cloud at a given path
        for a given step, ensuring that the step has been run. Return
        the path to the transformed file inside self._tmp_dir."""

        # Check registration has been performed, and run it if not
        i = self.get_step_number(step)
        step = self.get_step_name(step)
        self.ensure_registered(step)

        # Make temporary modified parameter file if needed
        self.make_tmp_dir()
        if params is None:
            tfile = self.tfiles[step]
        else:
            tfile = os.path.join(self._tmp_dir, "TransformParameters.txt")
            adjust_parameters(self.tfiles[step], tfile, params)

        # Define parameters specific to data type.
        if '.txt' == os.path.splitext(path)[1]:
            option = '-def'
            outfile = 'outputpoints.txt' 
        else:
            option = '-in'
            outfile = 'result.nii'

        # Run transformix
        cmd = [
            _transformix,
            option,
            path.replace("\\", "/"),
            "-out",
            self._tmp_dir,
            "-tp",
            tfile,
        ]
        self.logger.info(f'Running command:\n {" ".join(cmd)}')
        code = subprocess.run(
                cmd, capture_output=self.capture_output).returncode

        # If command failed, move log out from temporary dir
        if code:
            logfile = os.path.join(self.path, "transformix.log")
            if os.path.exists(logfile):
                os.remove(logfile)
            shutil.move(os.path.join(self._tmp_dir, "transformix.log"), self.path)
            self.logger.warning(
                f"Image transformation failed! See "
                f"{logfile} for more info."
            )
            return

        # Return path to result
        return os.path.join(self._tmp_dir, outfile)

    def transform_roi(self, roi, step=-1, outfile=None, params=None,
            transform_points=False):
        """Transform a single ROI using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps
        will be run. Either return the transformed ROI or write it to a file.

        **Parameters:**

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

        transform_points : bool, default=False
           If False, the transform is applied to pull the ROI mask
           from the reference frame of the moving image to the reference
           frame of the fixed image.  If True, the transform is applied
           to push ROI contour points from the reference frame of the
           fixed image to the reference frame of the moving image.
        """

        # Save ROI temporarily as nifti if needed
        roi = ROI(roi)
        roi.load()
        self.make_tmp_dir()
        if (roi.source_type == "mask" and roi.mask.source_type == "nifti"
                and not transform_points):
            roi_path = roi.mask.path
        else:
            ext = 'txt' if transform_points else 'nii.gz'
            roi_path = os.path.join(self._tmp_dir, f"{roi.name}.{ext}")
            roi.write(roi_path, verbose=(self.logger.level < 30))

        # Set default parameters
        default_params = {"ResampleInterpolator": '"FinalNearestNeighborInterpolator"'}
        if params is not None:
            default_params.update(params)

        # Transform the nifti file or point cloud
        result_path = self.transform_data(roi_path, step, default_params)
        if result_path is None or not os.path.exists(str(result_path)):
            return

        # Identify image to be associated with the transformed ROI.
        if transform_points:
            if issubclass(skrt.image.Image, type(self.moving_source)):
                image = self.moving_source
            elif isinstance(self.moving_source, str):
                image = skrt.image.Image(self.moving_source)
            else:
                image = getattr(self, 'moving_image', None)
        else:
            image = self.get_transformed_image(step)

        # Create ROI object, and check that it has contours defined.
        roi = ROI(result_path, name=roi.name, color=roi.color, image=image)
        if not roi.get_contours():
            return

        # Copy to output dir if outfile is set
        if outfile is not None:
            shutil.copy(result_path, outfile)
            return

        # Delete the temporary directory.
        self.rm_tmp_dir()

        # Return ROI object
        return roi

    def transform_structure_set(
        self, structure_set, step=-1, outfile=None, params=None,
        transform_points=False):
        """Transform a structure set using the output transform from a given
        registration step (by default, the final step). If the registration
        step has not yet been performed, the step and all preceding steps
        will be run. Either return the transformed ROI or write it to a file.

        **Parameters:**

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

        transform_points : bool, default=False
           If False, the transform is applied to pull ROI masks
           from the reference frame of the moving image to the reference
           frame of the fixed image.  If True, the transform is applied
           to push ROI contour points from the reference frame of the
           fixed image to the reference frame of the moving image.
        """

        final = StructureSet()
        for roi in structure_set:
            transformed_roi = self.transform_roi(roi, step, params=params,
                    transform_points=transform_points)
            if transformed_roi is not None:
                final.add_roi(transformed_roi)

        # Write structure set if outname is given
        if outfile is not None:
            final.write(outfile, verbose=(self.logger.level < 30))
            return

        # Otherwise, return structure set
        final.name = "Transformed"
        if transform_points:
            if issubclass(skrt.image.Image, type(self.moving_source)):
                image = self.moving_source
            elif isinstance(self.moving_source, str):
                image = skrt.image.Image(self.moving_source)
            else:
                image = getattr(self, 'moving_image', None)
        else:
            image = self.get_transformed_image(step)
        if image is not None:
            final.set_image(image)
        return final

    def transform_moving_image(self, step=-1):
        """Transform the moving image using the output of a registration step
        and set it to self.transformed_images[step]."""

        step = self.get_step_name(step)
        outfile = os.path.join(self.outdirs[step], "result.0.nii")
        self.transform(self.moving_image, step=step, outfile=outfile)
        self.transformed_images[step] = skrt.image.Image(
            outfile, title="Transformed moving")

    def get_transformed_image(self, step=-1, force=False):
        """Get the transformed moving image for a given step, by default the
        final step. If force=True, the transform will be applied with
        transformix even if there is already a resultant image in the
        output directory for that step."""

        # Run registration if needed
        step = self.get_step_name(step)
        was_registered = self.ensure_registered(step)

        # If forcing and registration had already been done, re-transform the 
        # moving image (otherwise, moving image will have just been recreated
        # anyway by running registration)
        if force and was_registered:
            self.transform_moving_image(step)

        # Return clone of the transformed image object
        if step in self.transformed_images:
            return self.transformed_images[step].clone()
        else:
            return None

    def ensure_registered(self, step):
        """If a step has not already been registered, perform registration
        for this step and any preceding steps. Return True if registration
        had already been performed."""

        if not self.is_registered(step):
            self.register_step(step)
            return False
        return True

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

    def adjust_file(self, step, params, ftype='p', reset=True):
        """Adjust parameters in a parameter or transform file for a given step. 

        **Parameters:**

        step : str/int
            Name or number of step for which the parameter file should be 
            adjusted.

        params : dict
            Dict of parameter names and desired parameter values. Any parameters
            matching those in the existing file will be overwritten with the
            new values. Any parameters not in the existing file will be added.
            Parameters in the existing file but not in <params> will remain
            unchanged.

        ftype : str, default='p'
            String indicating whether the file to be added is a parameter
            file ('p') or a transform file ('t').

        reset : bool, default=True
            If True, and the step can be rerun (that is, it has a
            parameter file assigne), this will remove existing
            registration transforms.
        """
        # Check that file type is recognised.
        if not ftype.lower() in self.file_types:
            self.logger.warning(f"File ignored: '{file}'")
            self.logger.warning(f"File type not recognised: '{ftype}'")
            self.logger.warning(f"\nValid file types are: '{self.file_types}'")
            return
        ftype = ftype.lower()

        step = self.get_step_name(step)

        if "p" == ftype:
            adjust_parameters(self.pfiles[step], self.pfiles[step], params)
        elif "t" == ftype:
            adjust_parameters(self.tfiles[step], self.tfiles[step], params)

        if (reset and step in self.tfiles
                and os.path.exists(self.pfiles.get(step, ""))):
            del self.tfiles[step]

    def view_init(self, **kwargs):
        """Interactively view initial fixed and moving images."""

        from skrt.better_viewer import BetterViewer

        kwargs.setdefault(
            "intensity", 
            [self.fixed_image._default_vmin, self.fixed_image._default_vmax]
        )
        kwargs.setdefault("match_axes", "y")
        kwargs.setdefault("title", ["Fixed", "Moving"])
        kwargs.setdefault("comparison", True)
        BetterViewer([self.fixed_image, self.moving_image], **kwargs)

    def view_result(self, step=-1, compare_with_fixed=True, **kwargs):
        """
        Interactively view transformed image, optionally side-by-side
        with fixed image.

        **Parameters:**

        step : int/str, default=-1
            Name or number of step for which to view the result. By default,
            the result of the final step will be shown.

        compare_with_fixed : bool, default=True
            If True, the result will be displayed in comparison with the
            fixed image.

        `**`kwargs :
            Optional keyword arguments to pass to BetterViewer.
        """

        from skrt.better_viewer import BetterViewer

        step = self.get_step_name(step)
        if step not in self.transformed_images:
            self.register_step(step)
        if compare_with_fixed:
            ims = [self.fixed_image, self.transformed_images[step]]
            kwargs.setdefault("comparison", True)
            kwargs.setdefault(
                "intensity", 
                [self.fixed_image._default_vmin, self.fixed_image._default_vmax]
            )
            kwargs.setdefault("title", ["Fixed", "Transformed moving"])
        else:
            ims = self.transformed_images[step]
            kwargs.setdefault("title", "Transformed moving")
        BetterViewer(ims, **kwargs)

    def manually_adjust_translation(self, step=None, reapply_transformation=True):
        """
        Open BetterViewer and manually adjust the translation between the
        fixed image and the result of a registration. If the "write translation"
        button is clicked, the manual translation will be added to the
        translation in the output transform file.

        **Parameters:**

        step : int/str, default=None
            Name, number, or list of the step which should have its translation
            modified. This must be provided if the registration has more than
            one step. If the tfile for the chosen step does not contain a
            translation, the function will immediately return None.

        reapply_transformation : bool, default=True
            If True, upon saving a translation, transformix will be run to
            reproduce the transformed moving image according to the new
            translation parameters.
        """

        # Get step name
        if step is None:
            if len(self.steps) == 1:
                step = self.steps[0]
            else:
                self.logger.warning(
                    "This registration has more than one step. The step to "
                    "be manually adjusted must be specified when running "
                    "Registration.manually_adjust_transform()."
                )
                return
        step = self.get_step_name(step)

        # Check registration has been run
        if not self.is_registered(step):
            self.logger.warning(
                    f"Registration for {step} has not yet been performed.")
            return

        # Check the tfile contains a 3-parameter translation
        pars = read_parameters(self.tfiles[step])
        if pars["Transform"] != "TranslationTransform":
            self.logger.warning(
                f"Can only manually adjust a translation step. Incorrect "
                f"transform type for step {step}: {pars['Transform']}"
            )
            return

        # Create BetterViewer and modify its write_translation function
        from skrt.better_viewer import BetterViewer

        bv = BetterViewer(
            [self.fixed_image, self.get_transformed_image(step=step)],
            comparison=True,
            translation=True,
            translation_write_style="shift",
            show=False,
        )
        bv.translation_output.value = self.tfiles[step]
        if reapply_transformation:
            bv._registration = self
            bv._registration_step = step
        bv.show()
        return bv

    def get_input_parameters(self, step):
        """
        Get dict of input parameters for a given step.
        """

        step = self.get_step_name(step)
        return read_parameters(self.pfiles[step])

    def get_transform_parameters(self, step):
        """
        Get dict of output transform parameters for a given step.
        """

        step = self.get_step_name(step)
        if not self.is_registered(step):
            self.logger.warning(
                    f"Registration step {step} has not yet been performed.")
            return
        return read_parameters(self.tfiles[step])

    def get_step_name(self, step):
        """Convert <step> to a string containing a step name. If <step> is
        already a string, use this; otherwise if <step> is an integer,
        obtain the corresponding step name.  Check that step has
        a parameter file and/or a transform file assigned."""

        if isinstance(step, str):
            if not step in self.steps:
                raise RuntimeError(f"Step {step} not a valid registration step")
        else:
            step = self.steps[step]

        if not self.pfiles.get(step, None) and not self.tfiles.get(step, None):
            raise RuntimeError(f"Step {step} not a valid registration step"
                    f"\nStep{step} has neither pfile nor tfile assigned")
        
        return step

    def get_step_number(self, step):
        """Convert <step> to an int containing the number of a given step.
        If <step> is an int, check it corresponds to a step number, ensure it 
        is positive, and return it. Otherwise if it's a string, return the 
        index of that step name in self.steps."""

        if isinstance(step, int):
            if step >= len(self.steps) or step < -len(self.steps):
                raise IndexError(f"Step {step} not a valid index number for "
                                 f"step list of length {len(self.steps)}")
            if step < 0:
                return len(self.steps) + step
            return step

        try:
            return self.steps.index(step)
        except ValueError:
            raise ValueError(f"Step {step} not found")

    def _get_jac_or_def(self, step, is_jac, force):

        # Settings for jacobian or deformation field
        if is_jac:
            storage = self.jacobians
        else:
            storage = self.deformation_fields

        # If object already exists, return it unless forcing
        step = self.get_step_name(step)
        if step in storage and not force:
            return storage[step]

        # Ensure registration has been performed for this step
        self.ensure_registered(step)

        # Create new object
        storage[step] = self.run_transformix_on_all(
            is_jac, outdir=self.outdirs[step], tfile=self.tfiles[step], 
            image=self.transformed_images[step]
        )
        return storage[step]

    def get_transformed_grid(self, step=-1, force=False,
            spacing=(30, 30, 30), thickness=(2, 2, 2),
            voxel_units=False, color='green'):
        '''
        Obtaing transformed grid.

        A three-dimensional grid is defined in the space of
        the moving image, and is transformed by applying the
        result of a registration step.

        **Parameters:**

        step : int, default=-1
            Registration step for which transformed grid is
            to be obtained.

        force : bool, default=False
            If False, return any previously calculated transformed
            grid.  If True, disregard any previous calculations.

        spacing : tuple, default=(30, 30, 30)
            Spacing along (x, y, z) directions of grid lines.  If
            voxel_units is True, values are taken to be in numbers
            of voxels.  Otherwise, values are taken to be in the
            same units as the voxel dimensions of the moving image.

        thickness : tuple, default=(2, 2, 2)
            Thickness along (x, y, z) directions of grid lines.  If
            voxel_units is True, values are taken to be in numbers
            of voxels.  Otherwise, values are taken to be in the
            same units as the voxel dimensions of the moving image.

        voxel_units : bool, default=False
            If True, values for spacing and thickness are taken to be
            in numbers of voxels.  If False, values for spacing and
            thickness are taken to be in the same units as the
            voxel dimensions of the moving image.

        color : tuple/str, default='green'
            Colour to use for grid lines.  The colour may be specified
            in any of the forms recognised by matplotlib:
            https://matplotlib.org/stable/tutorials/colors/colors.html
        '''

        # If object already exists, return it unless forcing
        storage = self.transformed_grids
        step = self.get_step_name(step)
        if step in storage and not force:
            return storage[step]

        # Ensure registration has been performed for this step
        self.ensure_registered(step)

        # Fixed intensities for foreground and background
        background = 0
        foreground = 1

        # Ensure that untransformed grid exists
        if not hasattr(self, 'moving_grid') or force:
            self.set_moving_grid(make_grid(self.moving_image, spacing,
                thickness, background, foreground, voxel_units))
            self.moving_grid = Grid(self.moving_grid.path,
                    color=color, image=self.moving_image, title='Grid')

        grid_path = Path(
                self.transform_data(path=self.moving_grid_path, step=step))
        grid_path = grid_path.rename(Path(self.outdirs[step]) / 'grid.nii')

        #self.transformed_grids[step] = skrt.image.Image(str(grid_path))
        self.transformed_grids[step] = Grid(str(grid_path), color=color,
                image=self.transformed_images[step], title='Transformed grid')

        return self.transformed_grids[step]

    def get_jacobian(self, step=-1, force=False, moving_to_fixed=False):
        """
        Generate Jacobian determinant using transformix for a given 
        registration step (or return existing Jacobian object, unless 
        force=True).

        Positive values in the Jacobian determinant represent:
        - moving_to_fixed=False: scaling from fixed image to moving image;
        - moving_to_fixed=True: scaling from moving image to fixed image.
        """

        if moving_to_fixed:
            jac1 = self._get_jac_or_def(step, True, force)
            jac1.load()
            jac2 = Jacobian(jac1)
            jac2.load()
            jac2.image = jac1.image
            jac2.data[jac2.data > 0] = 1 / jac2.data[jac2.data > 0]
            jac2._data_canonical[jac2._data_canonical > 0] = (
                    1 / jac2._data_canonical[jac2._data_canonical > 0])
            return jac2
        else:
            return self._get_jac_or_def(step, True, force)

    def get_deformation_field(self, step=-1, force=False):
        """Generate deformation field using transformix for a given 
        registration step."""
        
        return self._get_jac_or_def(step, False, force)

    def run_transformix_on_all(self, is_jac, outdir, tfile, image=None):
        """Run transformix with either `-jac all` or `-def all` to create a
        Jacobian determinant or deformation field file, and return either
        a Jacobian or DeformationField object initialised from the output file.
        """

        # Settings
        if is_jac:
            opt = "-jac"
            dtype = Jacobian
            expected_outname = "spatialJacobian.nii"
            title = "Jacobian determinant"
        else:
            opt = "-def"
            dtype = DeformationField
            expected_outname = "deformationField.nii"
            title = "Deformation field"

        # Run transformix
        cmd = [
            _transformix,
            opt,
            "all",
            "-out",
            outdir,
            "-tp",
            tfile
        ]
        self.logger.info(f'Running command:\n {" ".join(cmd)}')
        code = subprocess.run(cmd, capture_output=self.capture_output).returncode
        if code:
            logfile = os.path.join(outdir, 'transformix.log')
            raise RuntimeError(f"Creation of {title }failed. See {logfile} for"
                           " more info.")

        # Create output object
        output_file = os.path.join(outdir, expected_outname)
        assert os.path.exists(output_file)
        return dtype(output_file, image=image, title=title)


class Grid(ImageOverlay):

    def __init__(self, *args, color='green', **kwargs):

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to Grid.
        cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), color])
        self._default_cmap = cmap
        self._default_colorbar_label = "Intensity"
        self._default_vmin = 0
        self._default_vmax = 1

    def view(self, **kwargs):
        return ImageOverlay.view(self, kwarg_name="grid", **kwargs)


class Jacobian(ImageOverlay):

    def __init__(self, *args, **kwargs):

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to Jacobian determinant
        self._default_cmap = get_jacobian_colormap()
        self._default_colorbar_label = "Jacobian determinant"
        self._default_vmin = -1
        self._default_vmax = 2
        self._default_opacity = 0.8
        self.load()
        self.data = -self.data

    def view(self, **kwargs):
        return ImageOverlay.view(self, kwarg_name="jacobian", **kwargs)


class DeformationField(PathData):
    """Class representing a vector field."""

    def __init__(self, path, image=None, **kwargs):
        """Load vector field."""

        # Perform base-class initialisation.
        super().__init__(path)

        # Initialise own image object
        self._image = skrt.image.Image(path, **kwargs)

        # Assign an associated Image
        self.image = image

        # Initialise store for displacement images
        self.displacement_images = {"x": None, "y": None, "z": None, "3d": None}

        # Set some default plotting values.
        self._default_cmap = "seismic"
        self._default_colorbar_label = "displacement (mm)"
        self._3d_colorbar_label = "3D displacement magnitude (mm)"
        self._default_opacity = 0.8
        self._default_vmin = -15
        self._default_vmax = 15
        self._quiver_colorbar_label = "2D displacement magnitude (mm)"

    def load(self, force=False):

        self._image.load()
        assert self._image.get_data().ndim == 4

    def get_slice(self, view, sl=None, idx=None, pos=None, scale_in_mm=True):
        """Get voxel positions and displacement vectors on a 2D slice."""

        data = self._image.get_slice(view, sl=sl, idx=idx, pos=pos)
        x_ax, y_ax = skrt.image._plot_axes[view]

        # Get x/y displacement vectors
        df_x = np.squeeze(data[:, :, x_ax])
        df_y = np.squeeze(data[:, :, y_ax])
        if not scale_in_mm:
            df_x /= self._image.voxel_size[x_ax]
            df_y /= self._image.voxel_size[y_ax]

        # Get x/y coordinates of each point on the slice
        xs = np.arange(0, data.shape[1])
        ys = np.arange(0, data.shape[0])
        if scale_in_mm:
            xs = self._image.origin[x_ax] + xs * self._image.voxel_size[x_ax]
            ys = self._image.origin[y_ax] + ys * self._image.voxel_size[y_ax]
        y, x = np.meshgrid(ys, xs)
        x = x.T
        y = y.T
        return x, y, df_x, df_y

    def get_displacement_image(self, displacement="3d"):
        """
        Get skrt.dose.ImageOverlay object representing point displacements.

        **Parameter:**

        displacement : str, default="3d"
            Point displacement to be represented:
            - 'x': displacement along x-axis;
            - 'y': displacement along y-axis;
            - 'z': displacement along z-axis;
            - '3d': magnitude of 3d displacement.
        """

        # Check that type of information requested is unknown.
        if displacement not in self.displacement_images:
            print(f"Displacement type not known: '{displacement}'")
            print(f"Known displacement types: {list(self.displacement_images)}")
            return

        # Check if OverlayImage of displacement requested is already stored.
        # If not, then create and store it.
        if self.displacement_images[displacement] is None:
            im_data = self._image.clone().get_data()
            idx = "xyz".find(displacement)
            if -1 == idx:
                # Magnitudes of 3d displacements.
                im_data = np.sqrt(np.sum(np.square(im_data), axis=idx))
            else:
                # Signed displacements along axis.
                im_data = np.squeeze(im_data[:, :, :, idx])
            im = ImageOverlay(im_data, affine=self._image.get_affine())
            im._default_cmap = self._default_cmap
            if "3d" == displacement:
                im._default_colorbar_label = self._3d_colorbar_label
            else:
                im._default_colorbar_label = (
                        f"{displacement}-{self._default_colorbar_label}")
            im._default_opacity = self._default_opacity
            im._default_vmin = self._default_vmin
            im._default_vmax = self._default_vmax
            self.displacement_images[displacement] = im

        return self.displacement_images[displacement]

    def plot(
        self,
        view="x-y",
        sl=None,
        idx=None,
        pos=None,
        df_plot_type="quiver",
        include_image=False,
        df_opacity=None,
        df_spacing=30,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        show=True,
        save_as=None,
        scale_in_mm=True,
        title=None,
        colorbar=False,
        no_xlabel=False,
        no_ylabel=False,
        no_xticks=False,
        no_yticks=False,
        no_xtick_labels=False,
        no_ytick_labels=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        no_axis_labels=False,
        mask=None,
        mask_threshold=0.5,
        masked=True,
        invert_mask=False,
        mask_color="black",
        **mpl_kwargs
    ):

        # Set up axes
        self._image.set_ax(view, ax, gs, figsize, zoom)
        self.ax = self._image.ax
        self.fig = self._image.fig
        xlim = None
        ylim = None

        # Plot the underlying image
        if include_image and self.image is not None:
            self.image.plot(view, sl=sl, idx=idx, pos=pos, ax=self.ax, gs=gs,
                    show=False, title="", colorbar=max((colorbar - 1), 0),
                    no_xlabel=no_xlabel, no_ylabel=no_ylabel,
                    no_xticks=no_xticks, no_yticks=no_yticks,
                    no_xtick_labels=no_xtick_labels,
                    no_ytick_labels=no_ytick_labels,
                    mask=mask, mask_threshold=mask_threshold,
                    masked=masked, invert_mask=invert_mask,
                    mask_color=mask_color)
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

        # Get spacing in each direction in number of voxels
        df_spacing = self.convert_spacing(df_spacing, scale_in_mm)

        # Get vectors and positions on this slice
        data_slice = self.get_slice(view, sl=sl, idx=idx, pos=pos, 
                                    scale_in_mm=scale_in_mm)

        # Define plot's pre-zoom aspect ratio and axis limits.
        im_kwargs = self._image.get_mpl_kwargs(view, None, scale_in_mm)
        xlim = xlim or im_kwargs["extent"][0: 2]
        ylim = ylim or im_kwargs["extent"][2: 4]
        aspect = im_kwargs["aspect"]

        # Define plot opacity.
        mpl_kwargs["alpha"] = df_opacity or mpl_kwargs.get(
                "alpha", self._default_opacity)

        # Extract kwargs for colour bar and label.
        clb_kwargs = mpl_kwargs.pop("clb_kwargs", {})
        clb_label_kwargs = mpl_kwargs.pop("clb_label_kwargs", {})

        # Create plot
        if df_plot_type == "quiver":
            self._plot_quiver(view, data_slice, df_spacing, colorbar,
                    clb_kwargs, clb_label_kwargs, mpl_kwargs)
        elif df_plot_type == "grid":
            self._plot_grid(view, data_slice, df_spacing, mpl_kwargs)
        elif df_plot_type in ["x-displacement", "y-displacement",
                "z-displacement", "3d-displacement"]:
            self._plot_displacement(df_plot_type=df_plot_type,
                    view=view, sl=sl, idx=idx, pos=pos,
                    include_image=False, ax=self.ax, gs=gs, show=False,
                    title="", colorbar=colorbar, no_xlabel=no_xlabel,
                    no_ylabel=no_ylabel, no_xticks=no_xticks,
                    no_yticks=no_yticks, no_xtick_labels=no_xtick_labels,
                    no_ytick_labels=no_ytick_labels,
                    annotate_slice=annotate_slice,
                    major_ticks=major_ticks, minor_ticks=minor_ticks,
                    ticks_all_sides=ticks_all_sides,
                    no_axis_labels=no_axis_labels, mask=mask,
                    mask_threshold=mask_threshold, masked=masked,
                    invert_mask=invert_mask, mask_color=mask_color,
                    mpl_kwargs = mpl_kwargs)
        else:
            print(f"Unrecognised plot type '{df_plot_type}'")

        # Set plot's pre-zoom aspect ratio and axis limits.
        self.ax.set_aspect(aspect)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Label and zoom axes
        idx = self._image.get_idx(view, sl=sl, idx=idx, pos=pos)
        self._image.label_ax(view, idx=idx, scale_in_mm=scale_in_mm,
                title=title, no_xlabel=no_xlabel, no_ylabel=no_ylabel,
                no_xticks=no_xticks, no_yticks=no_yticks,
                no_xtick_labels=no_xtick_labels,
                no_ytick_labels=no_ytick_labels, annotate_slice=annotate_slice,
                major_ticks=major_ticks, ticks_all_sides=ticks_all_sides,
                no_axis_labels=no_axis_labels, **mpl_kwargs)
        self._image.zoom_ax(view, zoom, zoom_centre)

        # Display image
        plt.tight_layout()
        if show:
            plt.show()

        # Save to file
        if save_as:
            self.fig.savefig(save_as)
            plt.close()

    def _plot_quiver(
        self, 
        view, 
        data_slice,
        spacing,
        colorbar=0,
        clb_kwargs=None,
        clb_label_kwargs=None,
        mpl_kwargs=None, 
    ):
        """Draw a quiver plot."""

        # Get arrow positions and lengths
        x_ax, y_ax = skrt.image._plot_axes[view]
        x, y, df_x, df_y = data_slice
        arrows_x = df_x[:: spacing[y_ax], :: spacing[x_ax]]
        arrows_y = -df_y[:: spacing[y_ax], :: spacing[x_ax]]
        plot_x = x[:: spacing[y_ax], :: spacing[x_ax]]
        plot_y = y[:: spacing[y_ax], :: spacing[x_ax]]

        # Make plotting kwargs
        mpl_kwargs = mpl_kwargs or {}
        vmin = mpl_kwargs.pop("vmin", 0)
        vmax = mpl_kwargs.pop("vmax",
                self.get_displacement_image("3d").get_data().max() * 1.1)
        default_kwargs = {"cmap": "jet"}
        default_kwargs.update(mpl_kwargs)
        default_kwargs["clim"] = default_kwargs.get("clim", (vmin, vmax))
        default_dot_colour = matplotlib.cm.get_cmap(default_kwargs["cmap"])(0)
        clb_kwargs = clb_kwargs or None
        clb_label_kwargs = clb_label_kwargs or None

        # Plot arrows
        if arrows_x.any() or arrows_y.any():
            if "color" in default_kwargs:
                self.ax.quiver(
                        plot_x,
                        plot_y,
                        arrows_x,
                        arrows_y,
                        **default_kwargs
                        )
            else:
                M = np.hypot(arrows_x, arrows_y)
                quiver = self.ax.quiver(
                        plot_x,
                        plot_y,
                        arrows_x,
                        arrows_y,
                        M,
                        **default_kwargs
                        )
                # Add colorbar
                if colorbar > 0 and mpl_kwargs.get(
                        "alpha", self._default_opacity) > 0:
                    clb = self.fig.colorbar(quiver, ax=self.ax,
                            **clb_kwargs)
                    clb.set_label(self._quiver_colorbar_label,
                            **clb_label_kwargs)
                    clb.solids.set_edgecolor("face")
        else:
            # If arrow lengths are zero, plot dots
            dot_colour = default_kwargs.get("color", default_dot_colour)
            ax.scatter(plot_x, plot_y, c=dot_colour, marker=".")

    def _plot_grid(
        self, 
        view, 
        data_slice,
        spacing,
        mpl_kwargs=None, 
    ):
        """Draw a grid plot."""

        # Get gridline positions
        #  self.ax.autoscale(False)
        x_ax, y_ax = skrt.image._plot_axes[view]
        x, y, df_x, df_y = data_slice
        grid_x = x + df_x
        grid_y = y + df_y

        # Make plotting kwargs
        default_kwargs = {"color": "green", "linewidth": 2}
        if mpl_kwargs is not None:
            default_kwargs.update(mpl_kwargs)

        # Disregard kwargs that aren't valid here,
        # but are valid for other representations of deformation field.
        for key in ["cmap", "vmin", "vmax"]:
            default_kwargs.pop(key, None)

        # Plot gridlines
        for i in np.arange(0, x.shape[0], spacing[y_ax]):
            self.ax.plot(grid_x[i, :], grid_y[i, :], **default_kwargs)
        for j in np.arange(0, x.shape[1], spacing[x_ax]):
            self.ax.plot(grid_x[:, j], grid_y[:, j], **default_kwargs)

    def _plot_displacement(self, df_plot_type="3d-displacement", **kwargs):
        """
        Plot displacements along an axis, or magnitudes of 3d displacements.

        **Parameter:**

        df_plot_type : str, default="3d-displacement"
            Type of plot to produce for deformation field.
            This function handles the plot types:
            'x-displacement', 'y-displacement', 'z-displacement',
            '3d-displacement'
        """

        if not df_plot_type.endswith("-displacement"):
            print(f"df_plot_type not recognised: {df_plot_type}")
            return

        displacement = df_plot_type.split("-")[0]
        im = self.get_displacement_image(displacement)
        if im is not None:

            # Set default values for vmin, vmax, cmap, alpha.
            kwargs["mpl_kwargs"] = kwargs.get("mpl_kwargs", {})
            vmax = max(abs(im._default_vmax), abs(im.get_data().max()),
                    abs(im._default_vmin), abs(im.get_data().min()), 0)

            mpl_kwargs = {"vmin": -vmax, "vmax": vmax,
                    "cmap": im._default_cmap, "alpha": im._default_opacity}
            for key, default_value in mpl_kwargs.items():
                kwargs["mpl_kwargs"][key] = (
                        kwargs["mpl_kwargs"].get(key, default_value))

            # Disregard kwargs that aren't valid here,
            # but are valid for other representations of deformation field.
            for key in ["color"]:
                kwargs["mpl_kwargs"].pop(key, None)

            # Plot image.
            im.plot(**kwargs)

    def view(self, include_image=False, **kwargs):
        """
        View the deformation field.

        **Parameters:**
        
        include_image : bool, default=True
            If True, the image associated with the deformation field will
            be displayed as underlay.

        Any ``**kwargs`` will be passed to BetterViewer initialisation.
        """

        from skrt.better_viewer import BetterViewer
        self.load()

        # Ensure that df keyword isn't passed also via kwargs.
        kwargs.pop("df", None)

        # Define image for display: the image associated with
        # the deformation field, or a dummy image of the same size.
        if include_image and self.image is not None:
            im = self.image
        else:
            im = skrt.image.Image(
                    np.ones(self._image.get_data().shape[0: 3]) * 1e4,
                    affine=self._image.get_affine())
            im._default_cmap = self._default_cmap
            im._default_colorbar_label = self._default_colorbar_label
            im._default_vmin = self._default_vmin
            im._default_vmax = self._default_vmax
        
        # Create viewer
        return BetterViewer(im, df=self, **kwargs)
    
    def convert_spacing(self, spacing, scale_in_mm):
        """Convert grid spacing in mm or voxels to list containing grid spacing 
        in voxels in each dimension."""

        self.load()

        spacing = to_list(spacing)
        output = []
        if scale_in_mm:
            output = [abs(round(spacing[i] / self._image.voxel_size[i])) 
                      for i in range(3)]
        else:
            output = spacing

        # Ensure spacing is at least 2 voxels
        for i, sp in enumerate(output):
            if sp < 2:
                output[i] = 2

        return output


def set_elastix_dir(path):

    # Set directory
    _ELASTIX_DIR = fullpath(path)

    # Find elastix exectuable
    global _ELASTIX
    global _transformix
    if os.path.exists(os.path.join(_ELASTIX_DIR, "bin/elastix")):
        _ELASTIX = os.path.join(_ELASTIX_DIR, "bin/elastix")
        _transformix = os.path.join(_ELASTIX_DIR, "bin/transformix")
        lib_dir = os.path.join(_ELASTIX_DIR, 'lib')
    elif os.path.exists(os.path.join(_ELASTIX_DIR, "elastix.exe")):
        _ELASTIX = os.path.join(_ELASTIX_DIR, "elastix.exe")
        _transformix = os.path.join(_ELASTIX_DIR, "transformix.exe")
        lib_dir = os.path.join(os.path.dirname(_ELASTIX_DIR), 'lib')
    else:
        print(f"WARNING: No elastix executable found in {_ELASTIX_DIR}!")
        cmd = f"which {_ELASTIX}".split()
        stdout = subprocess.run(cmd, capture_output=True).stdout
        if stdout:
            exe_dir = str(Path(stdout.decode()).parent)
            print(f"INFO: Using elastix executable found in {exe_dir}")
        _ELASTIX_DIR = None

    # Cover Linux and MacOS
    if _ELASTIX_DIR:
        prepend_path('DYLD_FALLBACK_LIBRARY_PATH', lib_dir)
        prepend_path('LD_LIBRARY_PATH', lib_dir)

def prepend_path(variable, path, path_must_exist=True):
    '''
    Prepend path to environment variable.

    **Parameters:**

    variable : str
        Environment variable to which path is to be prepended.

    path : str
        Path to be prepended.

    path_must_exist : bool, default=True
        If True, only append path if it exists.
    '''
    path_ok = True
    if path_must_exist:
        if not os.path.exists(path):
            path_ok = False

    if path_ok:
        if variable in os.environ:
            if os.environ[variable]:
                os.environ[variable] = f'{path}:{os.environ[variable]}'
        else:
            os.environ[variable] = path

def adjust_parameters(infile, outfile, params):
    """Open an elastix parameter file (works for both input parameter and
    output transform files), adjust its parameters, and save it to a new
    file.

    **Parameters:**

    file : str
        Path to an elastix parameter file.

    outfile : str
        Path to output file.

    params : dict
        Dictionary of parameter names and new values.
    """

    # Check that file to be adjusted exists.
    if not os.path.exists(infile):
        self.logger.warning(f"File not found: '{infile}'")
        self.logger.warning("\nNo parameter-adjustment performed")
        return

    # Read input
    original_params = read_parameters(infile)
    original_params.update(params)
    write_parameters(outfile, original_params)

def read_parameters(infile):
    """Get dictionary of parameters from an elastix parameter file."""

    lines = [line for line in open(infile).readlines() if line.startswith("(")]
    lines = [line[line.find("(") + 1 : line.rfind(")")].split()
            for line in lines]
    params = {line[0]: " ".join(line[1:]) for line in lines}
    for name, param in params.items():
        if '"' in param:
            params[name] = param.strip('"')
            if params[name] == "false":
                params[name] = False
            elif params[name] == "true":
                params[name] = True
        elif len(param.split()) > 1:
            params[name] = [p for p in param.rstrip("(").split()]
            if "." in params[name][0]:
                params[name] = [float(p) for p in params[name]]
            else:
                params[name] = [int(p) for p in params[name]]
        else:
            if "." in param:
                params[name] = float(param)
            else:
                params[name] = int(param)
    return params


def write_parameters(outfile, params):
    """Write dictionary of parameters to an elastix parameter file."""

    file = open(outfile, "w")
    for name, param in params.items():
        if "//" == name:
            line = f"\n{name} {param}"
        else:
            line = f"({name}"
            if isinstance(param, str):
                line += f' "{param}")'
            elif isinstance(param, (list, tuple)):
                for item in param:
                    line += " " + str(item)
                line += ")"
            elif isinstance(param, bool):
                line += f' "{str(param).lower()}")'
            else:
                line += " " + str(param) + ")"
        file.write(line + "\n")
    file.close()


def shift_translation_parameters(infile, dx=0, dy=0, dz=0, outfile=None):
    """Add to the translation parameters in a file."""

    if outfile is None:
        outfile = infile
    pars = read_parameters(infile)
    init = pars["TransformParameters"]
    if pars["Transform"] != "TranslationTransform":
        self.logger.warning(
            f"Can only manually adjust a translation step. Incorrect "
            f"transform type: {pars['Transform']}"
        )
        return

    pars["TransformParameters"] = [init[0] - dx, init[1] - dy, init[2] - dz]
    write_parameters(outfile, pars)

def get_data_dir():
    """Return path to data directory within the Scikit-rt package."""
    return Path(resource_filename("skrt", "data"))

def get_default_pfiles_dir():
    """Return path to directory containing default parameter files."""
    return get_data_dir() / "elastix_parameter_files"

def get_default_pfiles(basename_only=True):
    """
    Get list of default parameter files.

    **Parameter:**

    basename_only : bool, default=True
        If True, return list of filenames only.  If True, return list
        of paths, as pathlib.Path objects.
    """

    files = [file for file in get_default_pfiles_dir().iterdir()
        if str(file).endswith(".txt")]
    if basename_only:
        return [file.name for file in files]
    return files

def get_jacobian_colormap(col_per_band=100, sat_values={0: 1, 1: 0.5, 2: 1}):
    '''
    Return custom colour map, for highlighting features of Jacobian determinant.

    Following image registration by Elastix, the Jacobian determinant of
    the registration transformation reflects the characteristics of the
    deformation field.  A positive value, x, in the Jacobian determinant
    indicates a volume scaling by x in going from the fixed image to the
    moving image.  A negative value indicates folding.

    The colour map defined here is designed to cover the range -1 to 2,
    highlighting the following:

    - regions of no volume change (value equal to 1);
    - regions of small and large expansion (values greater than 1
      by a small or large amount);
    - regions of small and large expansion (non-negative values less than 1
      by a small or large amount);
    - regions of folding (negative values).

    The colour map is as follows:
    - x < 0 (band 0): yellow, increasing linearly in opacity,
      from 0 at x ==0 to 1 at saturation value;
    - 0 <= x < 1 (band 1): blue, increasing in opacity as 1/x,
      from 0 at x == 1 to 1 at saturation value;
    - x == 1: transparent;
    - x > 1 (band 2): red, increasing linearly in opacity,
      from 0 at x == 1 to 1 at saturation value;

    **Parameters:**

    col_per_band : int, default=1000
        Number of colours (anchor points) per band in the colour map.

    sat_values : dict, default={0: 1, 1: 0.5, 2: 1}
        Dictionary of saturation values for the colour bands.  The
        saturation value is the (unsigned) distance along the x scale
        from the band threshold.
    '''
    # Define row indices within colour map for each band,
    # and determine total number of rows.
    n_band = 3
    ranges = {}
    all_ranges = set()
    for i in range(n_band):
        ranges[i] = list(range(i * col_per_band, 1 + (i + 1) * col_per_band))
        all_ranges = all_ranges.union(set(ranges[i]))
    n_col = len(all_ranges)

    # Initialise n_col x 3 array for red, green, blue, alpha.
    values = np.zeros(shape=(n_col, 3))
    anchors = np.linspace(0, 1, n_col)
    values[:, 0] = anchors
    red = values.copy()
    green = values.copy()
    blue = values.copy()
    alpha = values.copy()

    # Local function for mapping between colour intensities
    # and values in Jacobian determinant (scale from -1 to 2).
    def get_x(u):
        return (-1 + 3 * u)

    # Define rgba values for band 0:
    # yellow increasing linearly in opacity with negative x,
    # starting with opacity 0.5.
    band = 0
    x_sat = -sat_values[band]
    for i in ranges[band]:
        x = get_x(anchors[i])
        v = x / x_sat if (x_sat and x > x_sat) else 1
        v = min(v + 0.5, 1)
        red[i, 1:3] = 1
        green[i, 1:3] = 1
        alpha[i, 1:3] = v
    
    # Define rgba values for band 1:
    # blue increasing in opacity as 1/x from 1 to 0.
    band = 1
    x_1 = get_x(anchors[ranges[band][1]])
    x_max = get_x(anchors[ranges[band][-1]])
    x_sat = max(1 - sat_values[band], x_1)
    v_min = 1 / x_max
    v_max = 1 / x_sat - v_min
    for i in ranges[band]:
        x = get_x(anchors[i])
        v = ((1 / x) - v_min) / v_max if (x and x > x_sat) else 1
        if i == ranges[band][0]:
            red[i, 2] = 0
            green[i, 2] = 0
            blue[i, 2] = 1
            alpha[i, 2] = v
        else:
            blue[i, 1:3] = 1
            alpha[i, 1:3] = v
        
    # Define rgba values for band 2:
    # red increasing linearly in opacity above x=1.
    band = 2
    x_sat = 1 + sat_values[band]
    for i in ranges[band]:
        x = get_x(anchors[i])
        v = (x - 1)/ (x_sat - 1) if (x_sat - 1 and x < x_sat) else 1
        if i == ranges[band][0]:
            red[i, 2] = 1
            green[i, 2] = 0
            blue[i, 2] = 0
            alpha[i, 2] = v
        else:
            red[i, 1:3] = 1
            alpha[i, 1:3] = v
 
    # Create colour map.
    cdict = {'red' : red, 'green' : green, 'blue' : blue, 'alpha': alpha}
    cmap = matplotlib.colors.LinearSegmentedColormap('jacobian',
            segmentdata=cdict)

    return cmap

def get_image_transform_parameters(im):
    """
    Define Elastix parameters for warping an image to the space of image <im>.

    **Parameter:**
    
    im : skrt.image.Image
        Image object representing a fixed image in the context of registration.
    """
    # Use affine matrix for defining origin and direction cosines.
    # Seem to need to reverse sign of first two rows for agreement with
    # convention used in Elastix - may not work for arbitrary image orientation.
    affine = im.get_affine()
    affine[0:2, :] = -affine[0:2, :]
    
    # Spacings need to be positive.
    # Signs taken into account via direction cosines. 
    voxel_size = [abs(dxyz) for dxyz in im.get_voxel_size()]

    image_transform_parameters = {
            "//": "Image specific",
            "FixedImageDimension": 3,
            "MovingImageDimension": 3,
            "FixedInternalImagePixelType": "float",
            "MovingInternalImagePixelType": "float",
            "Size": im.get_n_voxels(),
            "Index": (0, 0, 0),
            "Spacing": voxel_size,
            "Origin": [0 + affine[row, 3] for row in range(3)],
            "Direction": [0 + affine[row, col] / voxel_size[col]
                for col in range(3) for row in range(3)],
            "UseDirectionCosines": True,
            "//": "ResampleInterpolator specific",
            "ResampleInterpolator": "FinalBSplineInterpolator",
            "FinalBSplineInterpolationOrder": 3,
            "//": "Resampler specific",
            "Resampler": "DefaultResampler",
            "DefaultPixelValue": 0,
            "ResultImageFormat": "nii",
            "ResultImagePixelType": "short",
            "CompressResultImage": False,
            }

    return image_transform_parameters
