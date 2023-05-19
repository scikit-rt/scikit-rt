"""
Classes and functions relating to image registration.

The following classes are defined:

- Registration : Class for handling image registration.
- DeformationField : Class representing a deformation field.
- Grid : Class representing a grid.
- Jacobian : Class representing a Jacobian determinant.
- RegistrationEngine : Base class for interfacing to a registration engine.
- Elastix: Class interfacing to elastix registration engine.
- NiftyReg: Class interfacing to NiftyReg registration engine.

The following functions are defined:

- add_engine() : Decorator for adding RegistrationEngine subclasses
  to skrt.registartion.engines dictionary.
- adjust_parameters() : Modify contents of a registration parameter file.
- get_data_dir() : Return path to data directory within the scikit-rt package.
- get_default_pfiles() : Get list of default parameter files.
- get_default_pfiles_dir() :  Return path to directory containing
  default parameter files for specified engine.
- get_engine_cls() : Get registration-engine class, given engine name
  or software directory.
- get_image_transform_parameters() : Define Elastix registration-independent
  parameters for transforms to the space of a specified image.
- get_jacobian_colormap() : Return custom colour map, for highlighting
  features of Jacobian determinant.
- prepend_path() : Prepend path to environment variable.
- read_parameters() : Get dictionary of parameters from a registration
  parameter file.
- set_elastix_dir() :  Perform environment setup for using elastix software.
- set_engine_dir() : Perform environment setup for using registration software.
- shift_translation_parameters() : Add offsets to the translation parameters
  in a registration parameter file.
- write_parameters() : Write dictionary of parameters to a registration
  parameter file.
"""
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import numbers
import numpy as np
import os
from pathlib import Path
from pkg_resources import resource_filename
import shutil
import subprocess
import warnings

import skrt.image
from skrt.structures import ROI, StructureSet
from skrt.core import (fullpath, get_logger, Data, is_list, to_list,
        Defaults, PathData)
from skrt.dose import ImageOverlay, Dose
from skrt.simulation import make_grid

# Set default registration engine.
Defaults({"registration_engine": "elastix"})

engines = {}
def add_engine(cls):
    """
    Decorator for adding skrt.registration.RegistrationEngine subclasses
    to the skrt.registartion.engines dictionary.
    """
    engines[cls.__name__.lower()] = cls
    return cls


class Registration(Data):
    """
    Class for handling image registration.
    """

    def __init__(
        self, path, fixed=None, moving=None, fixed_mask=None,
        moving_mask=None, pfiles=None, auto=False, overwrite=False,
        tfiles=None, initial_alignment=None, initial_transform_name=None,
        capture_output=False, log_level=None, keep_tmp_dir=False,
        engine=None, engine_dir=None):
        """
        Load data for an image registration, and run the registration if
        auto_seg=True.

        **Parameters:**

        path : str
            Path to directory where the data for this image registration is
            stored. Can either be a non-existing directory, in which case
            registration will be performed from scratch, or a directory already
            containing a fixed and moving image and optionally some registration
            steps.  The path shouldn't contain spaces.  It the path does
            contain spaces, these will be replaced with underscores.

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
            Path(s) to parameter file(s) to be used in each step of the
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

        keep_tmp_dir: bool, default=False
            If True, don't delete directory self._tmp_dir used when
            performing transformations.  Otherwise, delete this directory
            after use.

        engine: str, default=None
            Name of the registration engine to use for
            image registration.  This should be a key of
            the dictionary skrt.registration.engines.  If None,
            the first key found to be a substring of <engine_dir>
            is used, or if there's no match then the value of
            Defaults().registration_engine is used.

        engine_dir: str/pathlib.Path, default=None
            Path to directory containing software for registration engine.
        """

        # Set up event logging, output capture, and handling of self._tmp_dir
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)
        self.capture_output = capture_output
        self.keep_tmp_dir = keep_tmp_dir

        # Set registration engine.
        engine_cls = get_engine_cls(engine, engine_dir)

        if not (isinstance(engine_cls, type)
                and issubclass(engine_cls, RegistrationEngine)):
            raise RuntimeError(
                f"Unable to determine RegistrationClass for engine: '{engine}',"
                f"engine_dir: '{engine_dir}';"
                "\nknown engines are: {sorted(engines)}")

        self.engine = engine_cls(path=engine_dir)

        # Set up directory
        #path = fullpath(path).replace(" ", "_")
        self.path = path
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
                path = self.get_tfile(outdir)
                if step in self.pfiles and path:
                    self.tfiles[step] = path

        self.moving_grid_path = os.path.join(self.path, "moving_grid.nii.gz")
        self.transformed_grids = {}

        # Perform registration
        if auto:
            self.register()

    def get_tfile(self, dir_path):
        """
        Return path to transform file in directory <dir_path>.

        The transform file may be named either 'TransformParameters.0.txt'
        (e.g. Elastix or NiftyReg affine) or 'TransformParameters.0.nii'
        (e.g. NiftyReg deformable).  If both exist in the directory,
        the latter should be used.
        """
        tfiles = sorted(list(Path(dir_path).glob("TransformParameters*")))
        return str(tfiles[0]) if tfiles else None

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
            skrt.image.Image.write(im, path, verbose=(self.logger.level < 20))
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

    def add_pfile(self, file, name=None, params=None):
        """Alias for add_file() with ftype='p'."""
        self.add_file(file, name=name, params=params, ftype="p")

    def add_file(self, file, name=None, params=None, ftype="p"):
        """Add a single file of type <ftype> to the list of registration steps.
        This file can optionally be modified by providing a dict
        of parameter names and new values in <params>.

        **Parameters:**

        file : str/dict/tuple/list
            Path to the parameter file to copy into the registration
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
                self.logger.warning(f"Input for file: '{file}'")
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
            file = self.engine.define_translation(
                    file, getattr(self, "fixed_image", None))

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
            path = (self.get_tfile(outdir)
                    or f"{outdir}/TransformParameters.0.txt")
            self.tfiles[name] = path

        # Create new file, or copy existing file.
        if isinstance(file, dict):
            Path(path).touch()
            self.adjust_file(name, file, ftype)
        elif isinstance(file, list):
            with open(path, "w") as out_file:
                out_file.write("\n".join(file))
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

        for file in self.engine.get_default_pfiles():
            print(file.replace(".txt", ""))

    def get_default_params(self, filename):
        """
        Get the contents of a default parameter file as a dict. Note that this
        does not affect the current Registration object and is only for
        informative purposes.
        """

        files = self.engine.get_default_pfiles()
        if not filename.endswith(".txt"):
            filename += ".txt"
        if filename not in files:
            self.logger.warning(
                    f"Default file {name} not found. Available files:")
            self.list_default_pfiles(self)
            return
        full_files = self.engine.get_default_pfiles(False)
        pfile = full_files[files.index(filename)]
        return self.engine.read_parameters(pfile)

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

        files = self.engine.get_default_pfiles()
        if not filename.endswith(".txt"):
            filename += ".txt"
        if filename not in files:
            self.logger.warning(
                    f"Default file {name} not found. Available files:")
            self.list_default_pfiles(self)
            return

        full_files = self.engine.get_default_pfiles(False)
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
            tfile = self.get_tfile(outdir)
            if not os.path.exists(pfile) and not tfile:
                self.logger.warning(
                    f"No parameter file ({pfile}) "
                    f"and no transform file ({outdir}/TransformParameters*) "
                    f"found for registration step {step} listed in "
                    f"{self.steps_file}. This step will be ignored."
                )
                continue

            # Add step to the list.
            self.steps.append(step)
            self.outdirs[step] = outdir
            if pfile and os.path.exists(pfile):
                self.pfiles[step] = pfile
            if tfile and os.path.exists(tfile):
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
                    path=df_path, signs=self.engine.def_signs,
                    title="Deformation field",
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

    def get_registration_cmd(self, step, use_previous_tfile=True):
        """Get registration command for a given step."""

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
        if isinstance(tfile, str):
            tfile = tfile.replace("\\", "/")

        # Construct command
        return self.engine.get_registration_cmd(
                fixed_path=self.fixed_path.replace("\\", "/"),
                moving_path=self.moving_path.replace("\\", "/"),
                fixed_mask_path=self.fixed_mask_path.replace("\\", "/"),
                moving_mask_path=self.moving_mask_path.replace("\\", "/"),
                pfile=self.pfiles[step].replace("\\", "/"),
                outdir=self.outdirs[step].replace("\\", "/"),
                tfile=tfile)

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
        cmd = self.get_registration_cmd(step, use_previous_tfile)
        self.logger.debug(f"Running command:\n {' '.join(cmd)}")
        code = subprocess.run(
                cmd, capture_output=self.capture_output).returncode

        # Check whether registration succeeded
        if code:
            logfile = os.path.join(self.outdirs[step],
                                   f"{type(self.engine).__name__.lower()}.log")
            """
            self.logger.error(
                f"Registration step {step} failed! See "
                f"{logfile} or run Registration.print_log({step}) for "
                " more info."
            )
            """
            raise RuntimeError(
                f"Registration step {step} failed! See "
                f"{logfile} or run Registration.print_log({step}) for "
                " more info."
                    )
        else:
            self.tfiles[step] = self.get_tfile(self.outdirs[step])
            result_path = os.path.join(self.outdirs[step], "result.0.nii")
            im = skrt.image.Image(result_path, title="Transformed moving")
            if np.isnan(np.sum(im.get_data())):
                im.data = np.nan_to_num(im.get_data())
                im.write(result_path, verbose=(self.logger.level < 20))
            self.transformed_images[step] = im

    def is_registered(self, step):
        """Check whether a registration step has already been performed (i.e. 
        has a valid output transform file)."""

        step = self.get_step_name(step)
        return step in self.tfiles and os.path.exists(self.tfiles[step])

    def print_log(self, step=-1):
        """Print registration output log for a given step (by default, the 
        last step)."""

        step = self.get_step_name(step)
        logfile = os.path.join(self.outdirs[step],
                               f"{type(self.engine).__name__.lower()}.log")
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

    def transform_image(self, im, step=-1, outfile=None, params=None,
                        rois=None):
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
        is_dose = isinstance(im, Dose)
        if is_dose:
            params = params or {}
            params["ResultImagePixelType"] = "float"

        # Save image temporarily as nifti if needed
        im = skrt.image.Image(im)
        self.make_tmp_dir()
        if im.source_type == "nifti":
            im_path = im.path
        else:
            im_path = os.path.join(self._tmp_dir, "image.nii.gz")
            im.write(im_path, verbose=(self.logger.level < 20))

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
        tfile = self.tfiles[step]

        # Define parameters specific to data type.
        if '.txt' == os.path.splitext(path)[1]:
            outfile = 'outputpoints.txt' 
        else:
            outfile = 'result.nii'

        # Perform transformation
        cmd = self.engine.get_transform_cmd(
                fixed_path=self.fixed_path.replace("\\", "/"),
                moving_path=path, outdir=self._tmp_dir, tfile=tfile,
                params=params)

        self.logger.debug(f'Running command:\n {" ".join(cmd)}')
        code = subprocess.run(
                cmd, capture_output=self.capture_output).returncode

        # If command failed, move log out from temporary dir
        if code:
            logfile = os.path.join(
                    self.path, self.engine.get_transform_log())
            if os.path.exists(logfile):
                os.remove(logfile)
            shutil.move(os.path.join(
                self._tmp_dir, self.engine.get_transform_log()), self.path)
            self.logger.warning(
                f"Image transformation failed! See "
                f"{logfile} for more info."
            )
            return

        # Return path to result
        return os.path.join(self._tmp_dir, outfile)

    def transform_roi(self, roi, step=-1, outfile=None, params=None,
            transform_points=False, require_contours=False):
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

        require_contours : bool, default=False
           If the transformed ROI doesn't have contours defined: return
           the ROI if <require_contours> is False; return None if
           <require_contours> is True.  A transformed ROI won't have
           contours defined if, for example, it's transformed as a mask,
           and the non-zero regions of the transformed mask are outside
           the fixed image.
        """
        if transform_points and not self.engine.transform_points_implemented:
            raise RuntimeError("Transform of points not implemented "
                               f"for class {type(self.engine)}")

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
            roi.write(roi_path, verbose=(self.logger.level < 20))

        # Set default parameters
        default_params = self.engine.get_roi_params()
        if params is not None:
            default_params.update(params)

        # Identify image to be associated with the transformed ROI.
        # This needs to be here, to avoid the possibility of
        # the ROI transform result being overwritten by an image transform.
        if transform_points:
            if issubclass(skrt.image.Image, type(self.moving_source)):
                image = self.moving_source
            elif isinstance(self.moving_source, str):
                image = skrt.image.Image(self.moving_source)
            else:
                image = getattr(self, 'moving_image', None)
        else:
            image = self.get_transformed_image(step)

        # Transform the nifti file or point cloud
        result_path = self.transform_data(roi_path, step, default_params)
        if result_path is None or not os.path.exists(str(result_path)):
            return

        # Create ROI object, and check that it has contours defined.
        roi = ROI(result_path, name=roi.name, color=roi.color, image=image)
        if require_contours and not roi.get_contours():
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
            final.write(outfile, verbose=(self.logger.level < 20))
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
        final step. If force=True, the transform will be applied
        even if there is already a resultant image in the
        output directory for that step."""

        # Run registration if needed
        step = self.get_step_name(step)
        was_registered = self.ensure_registered(step)

        # If forcing and registration had already been done, re-transform the 
        # moving image (otherwise, moving image will have just been recreated
        # anyway by running registration)
        if (force or step not in self.transformed_images) and was_registered:
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

        #self._tmp_dir = os.path.join(self.path, ".tmp").replace(" ", "_")
        self._tmp_dir = os.path.join(self.path, ".tmp")
        if not os.path.exists(self._tmp_dir):
            os.mkdir(self._tmp_dir)

    def rm_tmp_dir(self):
        """Delete temporary directory and its contents."""

        if not hasattr(self, "_tmp_dir"):
            return
        if os.path.exists(self._tmp_dir):
            if not self.keep_tmp_dir:
                shutil.rmtree(self._tmp_dir)

    def adjust_pfile(self, step, params, reset=True):
        """Alias for adjust_file() with ftype='p'."""
        self.adjust_file(step, params, ftype='p', reset=reset)

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
            self.engine.adjust_parameters(
                    self.pfiles[step], self.pfiles[step], params)
        elif "t" == ftype:
            self.engine.adjust_parameters(
                    self.tfiles[step], self.tfiles[step], params)

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

    def manually_adjust_translation(self, step=None,
                                    reapply_transformation=True):
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
            If True, upon saving a translation, the transformed moving image
            will be produced according to the new translation parameters.
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
        if not self.engine.shift_translation_parameters(self.tfiles[step]):
            return

        # Create BetterViewer and modify its write_translation function
        from skrt.better_viewer import BetterViewer

        bv = BetterViewer(
            [self.fixed_image, self.get_transformed_image(step=step)],
            comparison=True,
            translation=True,
            translation_write_style=self.engine,
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
        return self.engine.read_parameters(self.pfiles[step])

    def get_transform_parameters(self, step):
        """
        Get dict of output transform parameters for a given step.
        """

        step = self.get_step_name(step)
        if not self.is_registered(step):
            self.logger.warning(
                    f"Registration step {step} has not yet been performed.")
            return
        return self.engine.read_parameters(self.tfiles[step])

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
        Generate Jacobian determinant for a given registration step
        (or return existing Jacobian object, unless force=True).

        If the registration engine used is unable to generate the
        Jacobian determinant, None will be returned.

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
        """Generate deformation field for a given registration step."""
        
        return self._get_jac_or_def(step, False, force)

    def run_transformix_on_all(self, is_jac, outdir, tfile, image=None):
        """
        Create a Jacobian determinant or deformation field file,
        and return either a Jacobian or DeformationField object initialised
        from the output file.

        This was first implemented by running transformix for an elastix
        transform file.  Creation of Jacobian or DeformationField object
        now depends on the registration engine used.
        """

        # Settings
        if is_jac:
            dtype = Jacobian
            expected_outname = "spatialJacobian.nii"
            title = "Jacobian determinant"
            cmd = self.engine.get_jac_cmd(
                    fixed_path=self.fixed_path.replace("\\", "/"),
                    outdir=outdir, tfile=tfile)
        else:
            dtype = DeformationField
            expected_outname = "deformationField.nii"
            title = "Deformation field"
            cmd = self.engine.get_def_cmd(
                    fixed_path=self.fixed_path.replace("\\", "/"),
                    outdir=outdir, tfile=tfile)

        # Obtain Jacobian determinant or deformation field.
        if cmd is not None:
            self.logger.debug(f'Running command:\n {" ".join(cmd)}')
            code = subprocess.run(
                    cmd, capture_output=self.capture_output).returncode
            if code:
                logfile = os.path.join(outdir, 'transform.log')
                raise RuntimeError(
                        f"Creation of {title }failed. See {logfile} for"
                        " more info.")

            # Create output object
            output_file = os.path.join(outdir, expected_outname)
            assert os.path.exists(output_file)
            kwargs = {} if is_jac else {"signs": self.engine.def_signs}
            return dtype(path=output_file, image=image, title=title, **kwargs)

    def get_comparison(self, step=-1, force=False, **kwargs):
        """
        Return a pandas DataFrame comparing fixed image
        and transformed moving image after step.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which
            mutual information is to be calculated.  Available
            steps are listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.

        **kwargs
            Keyword arguments passed to
            skrt.image.Image.get_comparison()
            See this method's documentation for options.
        """
        return self.fixed_image.get_comparison(
                self.get_transformed_image(step, force), **kwargs)

    def get_foreground_comparison(self, step=-1, force=False, **kwargs):
        """
        Return a pandas DataFrame comparing the foregrounds of
        fixed image and transformed moving image after step.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which
            mutual information is to be calculated.  Available
            steps are listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.

        **kwargs
            Keyword arguments passed to
            skrt.image.Image.get_foreground_comparison()
            See this method's documentation for options.
        """
        return self.fixed_image.get_comparison(
                self.get_foreground_image(step, force), **kwargs)

    def get_mutual_information(self, step=-1, force=False, **kwargs):
        """
        For fixed image and transformed moving image after step,
        calculate mutual information or a variant.
        after step.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which
            mutual information is to be calculated.  Available
            steps are listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.

        **kwargs
            Keyword arguments passed to
            skrt.image.Image.get_mutual_information()
            See this method's documentation for options.
        """
        return self.fixed_image.get_mutual_information(
                self.get_transformed_image(step, force), **kwargs)

    def get_quality(self, step=-1, force=False, metrics=None):
        """
        Evaluate quality relative to fixed image of transformed moving image
        after step.

        For information on quality metrics, see documentation of
        skrt.image.Image.get_quality().

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which relative
            structure content is to be calculated. Available steps are
            listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.

        metrics: list, default=None
            List of strings specifying quality metrics to be evaluated.
            If None, all defined quality metrics are evaluated.
        """
        return self.get_transformed_image(
                step, force).get_quality(self.fixed_image, metrics)

    def get_relative_structural_content(self, step=-1, force=False):
        """
        Quantify structural content relative to fixed image of
        transformed moving image after step.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which relative
            structure content is to be calculated. Available steps are
            listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.
        """
        return self.get_transformed_image(
                step, force).get_relative_structural_content(self.fixed_image)

    def get_fidelity(self, step=-1, force=False):
        """
        Calculate fidelity with which transformed moving image after step
        matches fixed image.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which fidelity
            is to be calculated. Available steps are listed in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.
        """
        return self.get_transformed_image(
                step, force).get_fidelity(self.fixed_image)

    def get_correlation_quality(self, step=-1, force=False):
        """
        Calculate quality of correlation between transformed moving image
        after step and fixed image.

        **Parameters:**
        step : int/str/list, default=None
            Name or number of the registration step after which correlation
            quality is to be calculated. Available steps are listed
            in self.steps.

        force : bool, default=False
            If True, transformation of the moving image will be
            forced, even if the image was transformed previously.
        """
        return self.get_transformed_image(
                step, force).get_correlation_quality(self.fixed_image)

class Grid(ImageOverlay):
    """
    Class representating a grid.

    This is the same as the ImageOverlay class, but sets different
    attribute values at instantiation.
    """

    def __init__(self, *args, color='green', **kwargs):
        """
        Perform ImageOverlay initialisation, then overwrite
        selected attribute values.
        """

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to Grid.
        cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), color])
        self._default_cmap = cmap
        self._default_colorbar_label = "Intensity"
        self._default_vmin = 0
        self._default_vmax = 1

    def view(self, **kwargs):
        """View the Grid."""
        return ImageOverlay.view(self, kwarg_name="grid", **kwargs)


class Jacobian(ImageOverlay):
    """
    Class for representing a Jacobian determinant.

    This is the same as the ImageOverlay class, but sets different
    attribute values at instantiation.
    """

    def __init__(self, *args, **kwargs):
        """
        Perform ImageOverlay initialisation, then overwrite
        selected attribute values.
        """

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to Jacobian determinant
        self._default_cmap = get_jacobian_colormap()
        self._default_colorbar_label = "Jacobian determinant"
        self._default_vmin = -1
        self._default_vmax = 2
        self._default_opacity = 0.8
        self.load()
        #self.data = -self.data

    def view(self, **kwargs):
        """View the Jacobian determinant."""
        return ImageOverlay.view(self, kwarg_name="jacobian", **kwargs)


class DeformationField(PathData):
    """Class representing a deformation field."""

    def __init__(self, path="", load=True, signs=None, image=None, **kwargs):
        """
        Initialise from a deformation-field source.

        path : str/array/Nifti1Image, default= ""
            Data source.  Possibilities are the same as for skrt.image.Image,
            but should correspond to arrays of dimension 4.
            Otherwise, it can be loaded later with the load() method.

        load : bool, default=True
            If True, the deformation-field data will be immediately loaded.

        signs : tuple, default=None
            Three element tuple of ints, indicating the igns to be applied to
            the (x, y, z) components of the deformation field, allowing
            for different conventions.  If None, components are taken to
            be as read from source.

        image : skrt.image.Image, default=None
            Image object to be associated with the deformation field.

        kwargs : dict
           Dictionary of keyword-value pairs, passed to the
           skrt.image.Image constructor when creating a representation
           of the deformation field.
        """
        # Perform base-class initialisation.
        super().__init__(path)

        # Initialise own image object
        self._image = skrt.image.Image(path=path, load=False, **kwargs)
        self.signs = signs
        if load:
            self.load()

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
        """
        Load deformation-field data from source. If already loaded and <force> 
        is False, nothing will happen.

        **Parameters:**
        
        force : bool, default=True
            If True, the deformation-field data will be reloaded from source
            even if it has previously been loaded.
        """
        if self._image.data is not None and not force:
            return

        # Load data, then perform axis transpositions and reversals.
        # Warning: this may not give the intended result for all data sources...
        self._image.load(force)
        assert self._image.get_data().ndim == 4
        self._image.data = np.transpose(
                self._image.data, (1, 0, 2, 3))[::-1, ::-1, :, :]

        # Apply convention-dependent signs to components of deformation field.
        if self.signs:
            for idx, sign in enumerate(self.signs):
                if sign < 0:
                    self._image.data[:, :, :, idx] *= sign
                    self._image._data_canonical[:, :, :, idx] *= sign

    def get_slice(self, view, sl=None, idx=None, pos=None, scale_in_mm=True):
        """Get voxel positions and displacement vectors on a 2D slice."""

        idx = self._image.get_idx(view, sl=sl, idx=idx, pos=pos)

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
            if self._image.voxel_size[x_ax] < 0:
                xs = xs[::-1]
            ys = self._image.origin[y_ax] + ys * self._image.voxel_size[y_ax]
            if self._image.voxel_size[y_ax] < 0:
                ys = ys[::-1]
        y, x = np.meshgrid(ys, xs)
        x = x.T
        y = y.T
        return x, y, df_x, -df_y

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
        """
        Plot deformation field.

        For explanation of parameters, see documentation for
        skrt.better_viewer.BetterViewer.
        """
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
        arrows_y = df_y[:: spacing[y_ax], :: spacing[x_ax]]
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
            self.ax.scatter(plot_x, plot_y, color=dot_colour, marker=".")

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


class RegistrationEngine:
    """
    Base class for interfacing to a registration engine.

    Some methods are effectively virtual: they will raise
    a NotImplementedError exception if called, and should be overridden
    in subclasses.

    This class defines the following variables and methods:

    **Class variables:**
    transform_points_implemented (bool): Indicate whether registration engine
        implements mapping of points from fixed image to moving images.
    def_signs (tuple/None): Indicate signs to be applied to
        (x, y, z) components of deformation field.

    **Methods:**
    __init__():  Create RegistrationEngine instance.
    adjust_parameters():
        Modify contents of a registration parameter file.
    define_translation(): Define translation parameters to be passed
        to registration engine.
    get_default_pfiles(): Get list of default parameter files.
    get_default_pfiles_dir(): Return path to directory containing
        default parameter files.
    get_def_cmd(): Get registration-engine command for computing
        deformation field.
    get_jac_cmd(): Get registration-engine command for computing
        Jacobian determinant.
    get_registration_cmd(): Get registration-engine command for
        performing image registration.
    get_roi_params(): Get default parameters to be used when
        transforming ROI mask.
    get_transform_cmd(): Get registration-engine command for
        applying registration transform.
    read_parameters(): Get dictionary of parameters from a
        registration parameter file.
    set_exe_env(): Set environment variables for running
        registration-engine software.
    set_exe_paths(): Set path(s)s to registration-engine executable(s).
    shift_translation_parameters(): Add offsets to the translation parameters
        in a registration transform file.
    write_parameters(): Write dictionary of parameters to a
        registration parameter file.

    **Static method:**
    get_transform_strategies(): Get list of available strategies
        for applying registration transform.
    """

    # Indicate whether registration engine implements mapping of points
    # from fixed image to moving images.
    transform_points_implemented = False

    # Indicate signs to be applied to (x, y, z) components of deformation field.
    # If None, components are taken to be as read from file.
    def_signs = None

    # Define engine executables.
    exes = []

    def __init__(self, path=None, force=False, log_level=None):
        """
        Create RegistrationEngine instance.

        **Parameters:**
        path : str/pathlib.Path, default=None
            Path to directory containing registration-engine software.

        force : bool, default=False
        If False, modify environment based on <path> only if registration
        software can't be located in the existing environment.  If True, modify
        environment based on <path> in all cases.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        """
        # Set up event logging.
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)

        if not hasattr(self, "name"):
            self.name = type(self).__name__.lower()
        self.set_exe_env(path, force)

    def adjust_parameters(self, infile, outfile, params):
        """
        Modify contents of a registration parameter file.

        **Parameters:**

        infile : str/patlib.Path
            Path to a registration parameter file.

        outfile : str/pathlib.Path
            Path to output file.

        params : dict
            Dictionary of parameter names and new values.
        """
        raise NotImplementedError("Method 'adjust_parameters()' "
                                  f"not implemented for class {type(self)}")

    def define_translation(self, dxdydz, fixed_image=None):
        """
        Define translation parameters to be passed to registration engine.

        **Parameters:**

        dxdydz : tuple/list
            Three-element tuple or list, giving translations in order
            (dx, dy, dz).  Translations correspond to the amounts added
            in mapping a point from fixed image to moving image.

        fixed_image : skrt.image.Image, default=None
            Image towards which moving image is to be warped.
        """
        raise NotImplementedError("Method 'define_translation()' "
                                  f"not implemented for class {type(self)}")

    def get_default_pfiles(self, basename_only=True):
        """
        Get list of default parameter files.

        **Parameter:**

        basename_only : bool, default=True
            If True, return list of filenames only.  If False, return list
            of paths, as pathlib.Path objects.
        """
        default_pfiles_dir = self.get_default_pfiles_dir()
        if default_pfiles_dir.is_dir():
            files = [file for file in default_pfiles_dir.iterdir()
                     if str(file).endswith(".txt")]
        else:
            files = []

        if basename_only:
            return [file.name for file in files]
        return files

    def get_default_pfiles_dir(self):
        """
        Return path to directory containing default parameter files.
        """
        return get_data_dir() / f"{self.name}_parameter_files"

    def get_def_cmd(self, fixed_path, outdir, tfile):
        """
        Get registration-engine command for computing deformation field.

        **Parameters:**

        fixed_path : str
            Path to fixed image.

        outdir : str
            Path to directory for output deformation field.

        tfile : str
            Path to registration transform file for which deformation
            field is to be computed.
        """
        raise NotImplementedError("Method 'get_def_cmd()' "
                                  f"not implemented for class {type(self)}")

    def get_jac_cmd(self, fixed_path, outdir, tfile):
        """
        Get registration-engine command for computing Jacobian determinant.

        **Parameters:**

        fixed_path : str
            Path to fixed image.

        outdir : str
            Path to directory for output deformation field.

        tfile : str
            Path to registration transform file for which deformation
            field is to be computed.
        """
        raise NotImplementedError("Method 'get_jac_cmd()' "
                                  f"not implemented for class {type(self)}")

    def get_registration_cmd(
            self, fixed_path, moving_path, fixed_mask_path, moving_mask_path,
            pfile, outdir, tfile=None):
        """
        Get registration-engine command for performing image registration.

        **Parameters:**

        fixed_path : str
            Path to fixed image.

        moving_path : str
            Path to moving image.

        fixed_mask_path : str
            Path to mask for fixed image.

        moving_mask_path : str
            Path to mask for moving image.

        pfile : str
            Path to registration paramter file.

        outdir : str
            Path to directory for registration output.

        tfile : str, default=None
            Path to registration transform file from previous registration
            step.  If None, no previous registration step is it be considered.
        """
        raise NotImplementedError("Method 'get_registration_cmd()' "
                                  f"not implemented for class {type(self)}")

    def get_roi_params(self):
        """
        Get default parameters to be used when transforming ROI mask.
        """
        return {}

    def get_transform_cmd(
            self, fixed_path, moving_path, outdir, tfile, params=None):
        """
        Get registration-engine command for applying registration transform.

        **Parameters:**

        fixed_path : str
            Path to fixed image.

        moving_path : str
            Path to moving image.

        outdir : str
            Path to directory for transform output.

        tfile : str
            Path to registration transform file to be applied.

        params: dict, default=None
            Dictionary of parameter-value pairs defining modifications
            to the registration transform file prior to running the
            transform command.  The original transform file is left
            unaltered.
        """
        raise NotImplementedError("Method 'get_transform_cmd()' "
                                  f"not implemented for class {type(self)}")

    def get_transform_log(self):
        """
        Get name of transform log file.
        """
        if not hasattr(self, "transform_log"):
            raise NotImplementedError("Class attribute 'transform_log' "
                                  f"not implemented for class {type(self)}")
        return self.transform_log

    def read_parameters(self, infile):
        """
        Get dictionary of parameters from a registration parameter file.

        **Parameter:**
        infile: str/pathlib.Path
            Path to registration parameter file.
        """
        raise NotImplementedError("Method 'read_parameters()' "
                                  f"not implemented for class {type(self)}")

    def set_exe_env(self, path=None, force=False):
        """
        Set environment variables for running registration-engine software.

        **Parameters:**

        path : str/pathlib.Path, default=None
            Path to directory containing registration-engine software.

        force : bool, default=False
        If False, modify environment based on <path> only if registration
        software can't be located in the existing environment.  If True, modify
        environment based on <path> in all cases.
        """
        exe_dir = None

        # Check if environment already set for running the registration engine.
        if not (path and force):
            exe_path = shutil.which(self.name)
            if exe_path is not None:
                exe_dir = Path(exe_path).parent

        # Return if environment already set and not forcing new setup,
        # of if no value specified for path (no new setup possible).
        if (exe_dir and not force) or path is None:
            return
        
        # Try to define path(s) to registration-engine executables,
        # if not already defined, or forcing new setup.
        if not exe_dir or force:
            exe_dir = self.set_exe_paths(path)

        if exe_dir:
            # Set environment variables.
            lib_dir = exe_dir.parent / "lib"
            # Cover Linux, MacOS, Windows.
            for env_var, env_val in [
                    ("DYLD_FALLBACK_LIBRARY_PATH", lib_dir),
                    ("LD_LIBRARY_PATH", lib_dir),
                    ("PATH", exe_dir)
                    ]:
                if not (os.environ.get(env_var, "")).startswith(str(env_val)):
                    prepend_path(env_var, env_val)

            self.logger.info(
                    f"Found {self.name} executable(s) in {exe_dir}") 
        else:
            # Registration-engine executables not found - raise exception.
            raise RuntimeError(
                    f"path={path}; {self.name} executable(s) not found")

    def set_exe_paths(self, path=None):
        """
        Set path(s)s to registration-engine executable(s).

        **Parameter:**

        path : str/pathlib.Path, default=None
            Path to directory containing registration-engine executable(s).
        """
        raise NotImplementedError("Method 'set_exe_paths()' "
                                  f"not implemented for class {type(self)}")

    def shift_translation_parameters(
            self, infile, dx=0, dy=0, dz=0, outfile=None):
        """
        Add offsets to the translation parameters in a registration
        transform file.
        
        **Parameters:**

        infile: str
            Path to input transform file.

        dx, dy, dz: int, default=0
            Amounts by which to increase translation parameters, along
            x, y, z directions.

        outfile: str, default=None
            Path to output parameter file.  If None, overwrite input
            parameter file.
        """
        raise NotImplementedError("Method 'shift_translation_parameters()' "
                                  f"not implemented for class {type(self)}")

    def write_parameters(self, outfile, params):
        """
        Write dictionary of parameters to a registration parameter file.

        **Parameters:**

        outfile: str/pathlib.Path
            Path to output file.

        params: dict
            Dictionary of parameters to be written to file.
        """
        raise NotImplementedError("Method 'write_parameters()' "
                                  f"not implemented for class {type(self)}")

    @staticmethod
    def get_transform_strategies():
        """
        Get list of available strategies for applying registration transform.

        Possible strategies are:

        - "pull": transform applied to pull image or mask from the reference
          frame of the moving image to the reference frame of the fixed image;

        - "push": transform applied to push points from the reference frame
          of the fixed image to the reference frame of the moving image.
        """
        return []


@add_engine
class Elastix(RegistrationEngine):
    """
    Class interfacing to elastix registration engine.
    """
    # Indicate whether registration engine implements mapping of points
    # from fixed image to moving images.
    transform_points_implemented = True

    # Define name of transform log file.
    transform_log = "transformix.log"

    # Define engine executables.
    exes = ["elastix", "transformix"]

    def __init__(self, **kwargs):
        # Initialise paths to executables.
        for exe in Elastix.exes:
            setattr(self, exe, exe)

        # Perform rest of initialisation via base class.
        super().__init__(**kwargs)

    def adjust_parameters(self, infile, outfile, params):
        # Modify elastix parameter file.
        # For information about format, see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf

        # Check that file to be adjusted exists.
        if not os.path.exists(infile):
            self.logger.warning(f"File not found: '{infile}'")
            self.logger.warning("No parameter-adjustment performed")
            return

        # Read input
        original_params = self.read_parameters(infile)
        original_params.update(params)
        self.write_parameters(outfile, original_params)

    def define_translation(self, dxdydz, fixed_image=None):
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

        if fixed_image is not None:
            # Add parameters needed to allow warping of the moving image.
            translation.update(get_image_transform_parameters(fixed_image))

        return translation

    def get_def_cmd(self, fixed_path, outdir, tfile):
        # Return command for computing deformation field.
        return [self.transformix, "-def", "all", "-out", outdir, "-tp", tfile]

    def get_jac_cmd(self, fixed_path, outdir, tfile):
        # Return command for computing Jacobian determinant.
        return [self.transformix, "-jac", "all", "-out", outdir, "-tp", tfile]

    def get_registration_cmd(
            self, fixed_path, moving_path, fixed_mask_path, moving_mask_path,
            pfile, outdir, tfile=None):

        # Start command with execuable and paths to fixed and moving images.
        cmd = [
            self.elastix,
            '-f', fixed_path,
            '-m', moving_path,
            ]

        # Add paths for any masks to be applied.
        if os.path.exists(fixed_mask_path):
            cmd.extend(['-fMask', fixed_mask_path])
        if os.path.exists(moving_mask_path):
            cmd.extend(['-mMask', moving_mask_path])

        # Add paths to registration parameter file and output directory.
        cmd.extend([
            "-p", pfile,
            '-out', outdir
            ])

        # Add transform parameter file from previous registration step.
        if tfile is not None:
            cmd.extend(['-t0', tfile])

        # Return command for performing image registration.
        return cmd

    def get_roi_params(self):
        # Return parameters to include when transforming ROI masks.
        return {"ResampleInterpolator": '"FinalNearestNeighborInterpolator"'}

    def get_transform_cmd(
            self, fixed_path, moving_path, outdir, tfile, params=None):
        # Perform any modifications to the transform parameter file.
        if params:
            out_tfile = str(Path(outdir) / Path(tfile).name)
            self.adjust_parameters(tfile, out_tfile, params)
            tfile = out_tfile.replace("\\", "/")

        # Return command for applying registration transform.
        return [
            self.transformix,
            "-def" if ".txt" == Path(moving_path).suffix else "-in",
            moving_path,
            "-out",
            outdir,
            "-tp",
            tfile,
        ]

    def read_parameters(self, infile):
        # Read elastix parameter file.
        # For information about format, see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
        lines = [line for line in open(infile).readlines()
                 if line.startswith("(")]
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

    def set_exe_paths(self, path):
        # Set paths to elastix executables.
        exe_dir = None
        sw_dir = Path(fullpath(path))
        if sw_dir.is_dir():
            if (sw_dir / "bin/elastix").exists():
                exe_dir = sw_dir / "bin"
                for exe in Elastix.exes:
                    setattr(self, exe, exe)
            elif (sw_dir / "elastix.exe").exists():
                exe_dir = sw_dir
                for exe in Elastix.exes:
                    setattr(self, exe, f"{exe}.exe")

        # Return the path to the directory containing the executables.
        return exe_dir

    def shift_translation_parameters(
            self, infile, dx=0, dy=0, dz=0, outfile=None):
        # Add offsets to the translation parameters
        # of an elaxtix transform file.
        if outfile is None:
            outfile = infile
        pars = self.read_parameters(infile)
        init = pars["TransformParameters"]
        if pars["Transform"] != "TranslationTransform":
            self.logger.warning(
                f"Can only manually adjust a translation step. Incorrect "
                f"transform type: {pars['Transform']}"
            )
            return False

        pars["TransformParameters"] = [init[0] - dx, init[1] - dy, init[2] - dz]
        self.write_parameters(outfile, pars)
        return True

    def write_parameters(self, outfile, params):
        # Write elastix parameter file.
        # For information about format, see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
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

    @staticmethod
    def get_transform_strategies():
        return ["pull", "push"]


@add_engine
class NiftyReg(RegistrationEngine):
    """
    Class interfacing to NiftyReg registration engine.
    """
    # Indicate signs to be applied to (x, y, z) components of deformation field.
    # If None, components are taken to be as read from file.
    def_signs = (-1, -1, 1)

    # Define engine executables.
    exes = ["reg_aladin", "reg_f3d", "reg_jacobian", "reg_resample",
            "reg_transform"]

    def __init__(self, **kwargs):
        # Initialise paths to executables.
        for exe in NiftyReg.exes:
            setattr(self, exe, exe)

        # Perform rest of initialisation via base class.
        super().__init__(**kwargs)

    def adjust_parameters(self, infile, outfile, params):
        # Modify NiftyReg parameter file.
        # NiftyReg doesn't define its own file format.  Here adopt
        # format used by elastix.  For information about this,
        # see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
        return Elastix.adjust_parameters(self, infile, outfile, params)

    def define_translation(self, dxdydz, fixed_image=None):
        # Define translation parameters that are enough to allow
        # application before a registration step.
        # Signs account for different conventions between NiftyReg and Elastix.
        translation = [
                f"1 0 0 {-dxdydz[0]}",
                f"0 1 0 {-dxdydz[1]}",
                f"0 0 1 {dxdydz[2]}",
                "0 0 0 1",
                ]

        return translation

    def get_def_cmd(self, fixed_path, outdir, tfile):
        # Return command for computing deformation field.
        return [self.reg_transform, "-ref", fixed_path, "-disp",
                tfile, f"{outdir}/deformationField.nii"]

    def get_jac_cmd(self, fixed_path, outdir, tfile):
        # Return command for computing Jacobian determinant.
        if ".nii" == Path(tfile).suffix:
            return [self.reg_jacobian, "-trans", tfile, "-ref", fixed_path,
                    "-jac", f"{outdir}/spatialJacobian.nii"]

    def get_registration_cmd(
            self, fixed_path, moving_path, fixed_mask_path, moving_mask_path,
            pfile, outdir, tfile=None):

        # Read registration parameters from file.
        params = self.read_parameters(pfile)

        # Define path to output directory.
        outdir = Path(outdir)

        # Start command with execuable, paths to fixed and moving images,
        # and path to output file (name following elastix convention).
        cmd = [
            getattr(self, params["exe"]),
            '-ref', fixed_path,
            '-flo', moving_path,
            '-res', str(outdir / "result.0.nii"),
            ]

        # Add paths for any masks to be applied.
        if os.path.exists(fixed_mask_path):
            cmd.extend(['-rmask', fixed_mask_path])
        if os.path.exists(moving_mask_path):
            cmd.extend(['-fmask', moving_mask_path])

        # Add path to transform from any previous (affine) step,
        # and path to output transform.
        if "reg_aladin" in cmd[0]:
            cmd.extend(['-aff', str(outdir / "TransformParameters.0.txt")])
            if tfile:
                cmd.extend(['-inaff', tfile])
        elif ("reg_f3d" in cmd[0]):
            if tfile:
                cmd.extend(['-aff', tfile])
            cmd.extend(['-cpp', str(outdir / "TransformParameters.0.nii")])

        # Add any other parameter-value pairs.
        for param, val in params.items():
            if "exe" != param and param not in cmd:
                cmd.append(param)
                if val != "":
                    cmd.append(str(val))

        return cmd

    def get_roi_params(self):
        # Return parameters to include when transforming ROI masks.
        return {"-inter": 0}

    def get_transform_cmd(
            self, fixed_path, moving_path, outdir, tfile, params=None):
        # Construct part of command relating to any input parameters.
        params = params or {}
        if not "-pad" in params:
            params["-pad"] = 0
        params = [str(val) for items in params.items() for val in items]

        # Return command for applying registration transform.
        return [self.reg_resample, "-ref", fixed_path, "-flo", moving_path,
                "-res", str(Path(outdir) / 'result.nii'),
                "-trans", tfile] + params

    def read_affine(self, infile):
        """
        Read affine matrix from file. 

        **Parameter:**
        infile: str/pathlib.Path
            Path to file from which to read affine matrix.
        """
        with open(infile) as file:
            lines = [line.strip().split() for line in file.readlines()]
        return [[float(item) for item in line] for line in lines]

    def read_parameters(self, infile):
        # Read NiftyReg parameter file.
        # NiftyReg doesn't define its own file format.  Here adopt
        # format used by elastix.  For information about this,
        # see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
        return Elastix.read_parameters(self, infile)

    def set_exe_paths(self, path):
        # Set paths to NiftyReg executables.
        exe_dir = None
        sw_dir = Path(fullpath(path))
        if sw_dir.is_dir():
            if (sw_dir / "bin/reg_f3d").exists():
                exe_dir = sw_dir / "bin"
                for exe in NiftyReg.exes:
                    setattr(self, exe, exe)
            elif (sw_dir / "reg_f3d.exe").exists():
                for exe in NiftyReg.exes:
                    setattr(self, exe, f"{exe}.exe")
                exe_dir = sw_dir

        # Return the path to the directory containing the executables.
        return exe_dir

    def shift_translation_parameters(
            self, infile, dx=0, dy=0, dz=0, outfile=None):
        # Add offsets to the translation parameters
        # of a Niftyreg transform file.
        print(infile, dx, dy, dz, outfile)
        if outfile is None:
            outfile = infile

        affine = self.read_affine(infile)

        # Offset translation parameters.
        # Signs account for different conventions between NiftyReg and Elastix.
        affine[0][3] -= dx
        affine[1][3] -= dy
        affine[2][3] += dz

        self.write_affine(affine, outfile)
        return True

    def write_affine(self, affine, outfile):
        """
        Write affine matrix fo file. 

        **Parameter:**
        outfile: str/pathlib.Path
            Path to file to which to write affine matrix.
        """
        rows = [" ".join([str(val) for val in row]) for row in affine]
        with open(outfile, "w") as file:
            file.write("\n".join(rows))

    def write_parameters(self, outfile, params):
        # Write NiftyReg parameter file.
        # NiftyReg doesn't define its own file format.  Here adopt
        # format used by elastix.  For information about this,
        # see section 3.4 of elastix manual:
        # https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
        return Elastix.write_parameters(self, outfile, params)

    @staticmethod
    def get_transform_strategies():
        return ["pull"]


def adjust_parameters(infile, outfile, params, engine=None):
    """
    Modify contents of a registration parameter file.

    **Parameters:**

    infile : str
        Path to a registration parameter file.

    outfile : str
        Path to output file.

    params : dict
        Dictionary of parameter names and new values.

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.
    """
    return get_engine_cls(engine)().adjust_parameters(infile, outfile, params)

def get_data_dir():
    """Return path to data directory within the scikit-rt package."""
    return Path(resource_filename("skrt", "data"))


def get_default_pfiles(pattern="*.txt", engine=None, basename_only=False):
    """
    Get list of default parameter files.

    **Parameter:**

    basename_only : bool, default=False
        If True, return list of filenames only.  If False, return list
        of paths, as pathlib.Path objects.

    engine: str, default=None
        Name of registration engine for which path to directory
        containing default parameter files is required.  If None,
        use value set for skrt.core.Default().registration_engine.

    pattern: str, default="*.txt"
        Glob-style pattern for filtering on file names.
    """
    engine = get_engine_name(engine)
    files = sorted(list(get_default_pfiles_dir(engine).glob(pattern)))
    if basename_only:
        return [file.name for file in files]
    return files


def get_default_pfiles_dir(engine=None):
    """
    Return path to directory containing default parameter files
    for specified engine.

    **Parameter:**
    engine: str, default=None
        Name of registration engine for which path to directory
        containing default parameter files is required.  If None,
        use value set for skrt.core.Default().registration_engine.
    """
    return get_data_dir() / f"{get_engine_name(engine)}_parameter_files"

def get_engine_name(engine=None, engine_dir=None):
    """
    Get registration-engine name, given engine name or software directory.

    **Parameters:**

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.

    engine_dir: pathlib.Path/str, default=None
        Path to directory containing registration-engine software.
        It's assumed that the registration engine is a key of
        the dictionary skrt.registration.engines, that the directory
        path includes this key, and that directory path doesn't
        include any other keys of skrt.registration.engines.
    """
    # Treat case where engine is a key of engines.
    if engine in engines:
        return engine

    # Treat case where engine_dir includes a key of engines.
    for local_engine in engines:
        if local_engine in str(engine_dir):
            return local_engine

    # Return default engine name.
    return Defaults().registration_engine

def get_engine_cls(engine=None, engine_dir=None):
    """
    Get registration-engine class, given engine name or software directory.

    **Parameters:**

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.

    engine_dir: pathlib.Path/str, default=None
        Path to directory containing registration-engine software.
        It's assumed that the registration engine is a key of
        the dictionary skrt.registration.engines, that the directory
        path includes this key, and that directory path doesn't
        include any other keys of skrt.registration.engines.
    """
    return engines.get(get_engine_name(engine, engine_dir), None)


def get_image_transform_parameters(im):
    """
    Define Elastix registration-independent parameters for transforms
    to the space of a specified image.

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


def get_jacobian_colormap(col_per_band=100, sat_values={0: 1, 1: 0.5, 2: 1}):
    '''
    Return custom colour map, for highlighting features of Jacobian determinant.

    Following image registration, the Jacobian determinant of
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
    path = str(path)
    path_ok = True
    if path_must_exist:
        if not os.path.exists(path):
            path_ok = False

    if path_ok:
        if variable in os.environ:
            if os.environ[variable]:
                os.environ[variable] = os.pathsep.join(
                        [path, os.environ[variable]])
        else:
            os.environ[variable] = path


def read_parameters(infile, engine=None):
    """
    Get dictionary of parameters from a registration parameter file.

    **Parameters:**
    infile: str, pathlib.Path
        Path fo registration parameter file.

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.
    """
    return get_engine_cls(engine)().read_parameters(infile)


def set_elastix_dir(path, force=True):
    """
    Perform environment setup for using elastix software.

    Function deprecated - use set_engine_dir()

    **Parameters:**

    path : str/pathlib.Path, default=None
        Path to directory containing elastix software.

    force : bool, default=True
        If False, modify environment based on <path> only if elastix software
        can't be located in the existing environment.  If True, modify
        environment based on <path> in all cases.
    """
    warnings.warn("set_elastix_dir() deprecated - "
                  "use set_engine_dir(engine='elastix')", DeprecationWarning)
    set_engine_dir(path, "elastix", force)


def set_engine_dir(path, engine=None, force=True):
    """
    Perform environment setup for using registration software.

    path : str/pathlib.Path, default=None
        Path to directory containing registration software.

    engine : str, default=None
        Name identifying the registration engine for which
        environment setup is to be performed.  This should correspond
        to a key of the dictionary skrt.registration.engines.

    force : bool, default=True
        If False, modify environment based on <path> only if registration
        software can't be located in the existing environment.  If True, modify
        environment based on <path> in all cases.
    """
    local_engine = get_engine_name(engine, path)
    if not local_engine in engines:
        raise RuntimeError("Unable to determine registration engine "
                           f"from path: {path}, engine: {engine}; "
                           f"known engines are: {list(engines)}")
    get_engine_cls(local_engine)(path=path, force=force)


def shift_translation_parameters(
        infile, dx=0, dy=0, dz=0, outfile=None, engine=None):
    """
    Add offsets to the translation parameters in a registration transform file.
    
    **Parameters:**

    infile: str
        Path to input parameter file.

    dx, dy, dz: int, default=0
        Amounts by which to increase translation parameters, along
        x, y, z directions.

    outfile: str, default=None
        Path to output parameter file.  If None, overwrite input parameter file.

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.
    """
    get_engine_cls(engine)().shift_translation_parameters(
            infile, dx=dx, dy=dy, dz=dz, outfile=outfile)


def write_parameters(outfile, params, engine=None):
    """
    Write dictionary of parameters to a registration parameter file.

    **Parameters:**

    outfile: str
        Path to output file.

    params: dict
        Dictionary of parameters to be written to file.

    engine: str, default=None
        String identifying registration engine, corresponding to
        a key of the dictionary skrt.registration.engines.
    """
    get_engine_cls(engine)().write_parameters(outfile, params)
