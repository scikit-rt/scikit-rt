"""Tools for performing image registration."""
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess

import skrt.image
from skrt.structures import ROI, StructureSet
from skrt.core import get_logger, Data, to_list, Defaults
from skrt.dose import ImageOverlay, Dose

_ELASTIX_DIR = None
_ELASTIX = "elastix"
_transformix = "transformix"


class Registration(Data):

    def __init__(
        self, path, fixed=None, moving=None, pfiles=None, auto=False,
        overwrite=False, tfiles={}, capture_output=False, log_level=None):
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

        tfiles: dict, default={}
            Dictionary of pre-defined transforms, where a keys is a
            registration step and the associated value is the path to
            a pre-defined registration transform.  This parameter is
            considered only if pfiles is null.

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
        self.jacobians = {}
        self.deformation_fields = {}
        if isinstance(pfiles, str):
            pfiles = [pfiles]
        if pfiles is not None:
            self.add_pfiles(pfiles)
        else:
            self.load_pfiles()

        if not self.pfiles:
            self.tfiles = tfiles
            for step in sorted(tfiles):
                self.steps.append(step)

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
            Category of image ("fixed" or "moving").

        force : bool, default=True
            If True, the image file within self.path will be overwritten by
            this image even if it already exists.
        """

        if category not in ["fixed", "moving"]:
            raise RuntimeError(
                f"Unrecognised image category {category}; "
                "should be either fixed or moving"
            )

        if not isinstance(im, skrt.image.Image):
            im = skrt.image.Image(im)
        path = getattr(self, f"{category}_path")
        if not os.path.exists(path) or force:
            skrt.image.Image.write(im, path)
        setattr(self, f"{category}_image", skrt.image.Image(path))

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
                print(
                    f"Warning: no {category} image found at {path}! "
                    f"Make sure you run Registration.set_{category}_image"
                    " before running a registration."
                )
                return
            self.set_image(path, category, force=False)

    def add_pfile(self, pfile, name=None, params=None):
        """Add a single parameter file to the list of registration steps.
        This parameter file can optionally be modified by providing a dict
        of parameter names and new values in <params>.

        **Parameters:**

        pfile : str/dict
            Path to the elastix parameter file to copy into the registration
            directory. Can also be a dict containing parameters and values,
            which will be used to create a parameter file from scratch. In this
            case, the <name> argument must also be provided.

        name : str, default=None
            Name for this step of the registration. If None, a name will
            be taken from the parameter file name.

        params : dict, default=None
            Dictionary of parameter names and replacement values with which
            the input parameter file will be modified.
        """

        # Infer name from parameter file name if name is None
        if name is None:
            if isinstance(pfile, dict):
                print("If passing parameters from dict, <name> must be set.")
                return
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

        # If pfile already exists, copy it into output dir
        path = f"{outdir}/InputParameters.txt"
        shutil.copy(pfile, path)
        self.pfiles[name] = path

        # Modify the pfile if custom parameters are given
        if params is not None:
            self.adjust_pfile(name, params)

        # Rewrite text file containing list of steps
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
            self.add_pfile(p, name=name)

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
            print(f"Default file {name} not found. Available files:")
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
            print(f"Default file {name} not found. Available files:")
            self.list_default_pfiles(self)
            return

        full_files = get_default_pfiles(False)
        pfile = full_files[files.index(filename)]
        self.add_pfile(pfile, params=params)

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
                print(
                    f"Warning: no output directory ({self.path}/{step}) "
                    f"found for registration step {step} listed in "
                    f"{self.steps_file}. This step will be ignored."
                )
                continue

            pfile = os.path.join(outdir, "InputParameters.txt")
            if not os.path.exists(pfile):
                print(
                    f"Warning: no parameter file ({outdir}/InputParameters.txt) "
                    f"found for registration step {step} listed in "
                    f"{self.steps_file}. This step will be ignored."
                )
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
                print(
                    f"Warning: previous step {prev_step} has not yet "
                    f"been performed! Input transform file for step {step}"
                    " will not be used."
                )
            else:
                tfile = self.tfiles[prev_step]

        # Construct command
        cmd = (
            f"{_ELASTIX} -f {self.fixed_path} -m {self.moving_path} "
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
            self.register_step(step, force=force, use_previous_tfile=True)

    def register_step(self, step, force=False, use_previous_tfile=True):
        """Run a single registration step. Note that if use_previous_tfile=True,
        any prior steps that have not yet been run will be run.

        **Parameters:**

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
        if self.is_registered(step) and not force:
            return

        # Check that previous step has been run if needed
        i = self.get_step_number(step)
        step = self.get_step_name(step)
        if use_previous_tfile and i > 0:
            if not self.is_registered(i - 1):
                self.register(i, use_previous_tfile=True)

        # Run
        cmd = self.get_ELASTIX_cmd(step, use_previous_tfile)
        self.logger.info(f"Running command:\n {cmd}")
        code = subprocess.run(
                cmd.split(), capture_output=self.capture_output).returncode

        # Check whether registration succeeded
        if code:
            logfile = os.path.join(self.outdirs[step], "elastix.log")
            print(
                f"Warning: registration step {step} failed! See "
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
            print(f"Unrecognised transform input type {type(to_transform)}")

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
            im.write(im_path, verbose=False)

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
            print(
                f"Warning: image transformation failed! See "
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
            roi.write(roi_path, verbose=False)

        # Set default parameters
        default_params = {"ResampleInterpolator": '"FinalNearestNeighborInterpolator"'}
        if params is not None:
            default_params.update(params)

        # Transform the nifti file or point cloud
        result_path = self.transform_data(roi_path, step, default_params)
        if result_path is None:
            return result_path

        # Copy to output dir if outfile is set
        if outfile is not None:
            shutil.copy(result_path, outfile)
            return

        # Otherwise, return ROI object
        roi = ROI(result_path, name=roi.name, color=roi.color)
        self.rm_tmp_dir()
        if transform_points:
            if hasattr(self, 'moving_image'):
                roi.set_image(self.moving_image)
        else:
            transformed_image = self.get_transformed_image(step)
            if transformed_image is not None:
                roi.set_image(transformed_image)
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
            final.write(outfile)
            return

        # Otherwise, return structure set
        final.name = "Transformed"
        if transform_points:
            if hasattr(self, 'moving_image'):
                final.set_image(self.moving_image)
        else:
            transformed_image = self.get_transformed_image(step)
            if transformed_image is not None:
                final.set_image(transformed_image)
        return final

    def transform_moving_image(self, step=-1):
        """Transform the moving image using the output of a registration step
        and set it to self.transformed_images[step]."""

        step = self.get_step_name(step)
        outfile = os.path.join(self.outdirs[step], "result.0.nii")
        self.transform(self.moving_image, outfile=outfile)
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

    def adjust_pfile(self, step, params, reset=True):
        """Adjust parameters in an input parameter file for a given step. 

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

        reset : bool, default=True
            If True, this will remove existing registration results.
        """

        step = self.get_step_name(step)
        adjust_parameters(self.pfiles[step], self.pfiles[step], params)
        if reset and step in self.tfiles:
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
                print(
                    "This registration has more than one step. The step to "
                    "be manually adjusted must be specified when running "
                    "Registration.manually_adjust_transform()."
                )
                return
        step = self.get_step_name(step)

        # Check registration has been run
        if not self.is_registered(step):
            print(f"Registration for {step} has not yet been performed.")
            return

        # Check the tfile contains a 3-parameter translation
        pars = read_parameters(self.tfiles[step])
        if pars["Transform"] != "TranslationTransform":
            print(
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
            print(f"Registration step {step} has not yet been performed.")
            return
        return read_parameters(self.tfiles[step])

    def get_step_name(self, step):
        """Convert <step> to a string containing a step name. If <step> is
        already a string, check it's a valid step and return it; otherwise
        if <step> is an integer, return the corresponding step name."""

        if isinstance(step, str):
            if step not in self.steps:
                raise RuntimeError(f"Step {step} not a valid registration step")
            return step
        
        return self.steps[step]

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
        storage[step] = run_transformix_on_all(
            is_jac, outdir=self.outdirs[step], tfile=self.tfiles[step], 
            image=self.transformed_images[step]
        )
        return storage[step]

    def get_jacobian(self, step=-1):
        """Generate Jacobian determinant using transformix for a given 
        registration step (or return existing Jacobian object, unless 
        force=True).
        """

        return self._get_jac_or_def(step, True, force)

    def get_deformation_field(self, step=-1, force=False):
        """Generate deformation field using transformix for a given 
        registration step."""
        
        return self._get_jac_or_def(step, False, force)


class Jacobian(ImageOverlay):

    def __init__(self, *args, **kwargs):

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to Jacobian determinant
        self._default_cmap = "seismic"
        self._default_colorbar_label = "Jacobian"
        self._default_vmin = -0.5
        self._default_vmax = 2.5

    def view(self, **kwargs):
        return ImageOverlay.view(self, kwarg_name="jacobian", **kwargs)


class DeformationField:
    """Class representing a vector field."""

    def __init__(self, path, image=None, **kwargs):
        """Load vector field."""

        # Initialise own image object
        self._image = skrt.image.Image(path, **kwargs)

        # Assign an associated Image
        self.image = image

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

    def plot(
        self,
        view="x-y",
        sl=None,
        idx=None,
        pos=None,
        plot_type="quiver",
        include_image=False,
        spacing=30,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        show=True,
        save_as=None,
        scale_in_mm=True,
        **kwargs
    ):

        # Set up axes
        self._image.set_ax(view, ax, gs, figsize, zoom)
        self.ax = self._image.ax
        self.fig = self._image.fig

        # Plot the underlying image
        if include_image and self.image is not None:
            self.image.plot(view, ax=self.ax, show=False)

        # Get spacing in each direction in number of voxels
        spacing = self.convert_spacing(spacing, scale_in_mm)

        # Get vectors and positions on this slice
        data_slice = self.get_slice(view, sl=sl, idx=idx, pos=pos, 
                                    scale_in_mm=scale_in_mm)

        # Create plot
        if plot_type == "quiver":
            self._plot_quiver(view, data_slice, spacing, **kwargs)
        elif plot_type == "grid":
            self._plot_grid(view, data_slice, spacing, **kwargs)
        else:
            raise ValueError(f"Unrecognised plot type {plot_type}")

        # Label and zoom axes
        idx = self._image.get_idx(view, sl=sl, idx=idx, pos=pos)
        self._image.label_ax(view, idx=idx, scale_in_mm=scale_in_mm, **kwargs)
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
        default_kwargs = {"cmap": "jet"}
        if mpl_kwargs is not None:
            default_kwargs.update(mpl_kwargs)

        # Plot arrows
        if arrows_x.any() or arrows_y.any():
            M = np.hypot(arrows_x, arrows_y)
            self.ax.quiver(
                plot_x,
                plot_y,
                arrows_x,
                arrows_y,
                M,
                **default_kwargs
            )
        else:
            # If arrow lengths are zero, plot dots
            ax.scatter(plot_x, plot_y, c="navy", marker=".")

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

        # Plot gridlines
        for i in np.arange(0, x.shape[0], spacing[y_ax]):
            self.ax.plot(grid_x[i, :], grid_y[i, :], **default_kwargs)
        for j in np.arange(0, x.shape[1], spacing[x_ax]):
            self.ax.plot(grid_x[:, j], grid_y[:, j], **default_kwargs)

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



def run_transformix_on_all(is_jac, outdir, tfile, image=None):
    """Run transformix with either `-jac all` or `-def all` to create a
    Jacobian determinant or deformation field file, and return either
    a Jacobian or DeformationField object initialised from the output file.
    """

    # Settings
    if is_jac:
        opt = "-jac"
        dtype = Jacobian
        expected_outname = "spatialJacobian.nii"
        name = "Jacobian determinant"
    else:
        opt = "-def"
        dtype = DeformationField
        expected_outname = "deformationField.nii"
        name = "Deformation field"

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
    self.logger.info('Running command:\n {" ".join(cmd)}')
    code = subprocess.run(cmd, capture_output=self.capture_output).returncode
    if code:
        logfile = os.path.join(outdir, 'transformix.log')
        raise RuntimeError(f"Jacobian creation failed. See {logfile} for "
                           " more info.")

    # Create output object
    output_file = os.path.join(outdir, expected_outname)
    assert os.path.exists(output_file)
    return dtype(output_file, image=image, name=name)


def set_elastix_dir(path):

    # Set directory
    global _ELASTIX_DIR
    _ELASTIX_DIR = path

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
        raise RuntimeError(f"No elastix executable found in {_ELASTIX_DIR}!")


    # Cover Linux and MacOS
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

    # Read input
    original_params = read_parameters(infile)
    original_params.update(params)
    write_parameters(outfile, original_params)


def read_parameters(infile):
    """Get dictionary of parameters from an elastix parameter file."""

    lines = [line for line in open(infile).readlines() if line.startswith("(")]
    lines = [line[line.find("(") + 1 : line.find(")")].split() for line in lines]
    params = {line[0]: " ".join(line[1:]) for line in lines}
    for name, param in params.items():
        if '"' in param:
            params[name] = param.strip('"')
            if params[name] == "false":
                params[name] = False
            elif params[name] == "true":
                params[name] = True
        elif len(param.split()) > 1:
            params[name] = [p for p in param.split()]
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
        line = f"({name}"
        if isinstance(param, str):
            line += f' "{param}")'
        elif isinstance(param, list):
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
        print(
            f"Can only manually adjust a translation step. Incorrect "
            f"transform type: {pars['Transform']}"
        )
        return

    pars["TransformParameters"] = [init[0] - dx, init[1] - dy, init[2] - dz]
    write_parameters(outfile, pars)


def get_default_pfiles(basename_only=True):

    import skrt

    rel_path = "../../examples/elastix/parameter_files".split("/")
    pdir = os.path.join(skrt.__path__[0], *rel_path)
    files = [file for file in os.listdir(pdir) if file.endswith(".txt")]
    if basename_only:
        return files
    return [os.path.join(pdir, file) for file in files]
