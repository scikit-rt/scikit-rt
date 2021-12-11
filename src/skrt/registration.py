"""Tools for performing image registration."""

import os
import re
import shutil
import subprocess

from skrt.image import Image, ImageComparison


_elastix_dir = None
_elastix = "elastix"
_transformix = "transformix"


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


class Registration:
    def __init__(self, fixed, moving, pfile, outdir=".", auto_reg=True, force=False):
        """Register moving image to fixed image with a given elastix parameter
        file.
        Parameters:
        -----------
        fixed : Image/str
            Fixed image; can either be an Image object or the path to a source
            for an Image object.

        moving : Image/str
            Moving image; can either be an Image object or the path to a source
            for an Image object.

        pfile : str
            Path(s) to elastix parameter file(s). These parameter files will be
            used in series to find transforms mapping the moving image to the 
            fixed image.

        outdir : str, default='.'
            Path to output directory. If this directory already exists, data
            will be loaded in from that directory.

        auto_reg : bool, default=True
            If True, registration will be performed immediately.

        force : bool, default=False
            If <outdir> already exists and transformations with names matching
            the parameter files are found, registrations will not be rerun
            for these parameter files.
        """

        # Set up fixed and moving images
        self.fixed = fixed
        self.moving = moving
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.get_nifti_inputs(force)

        # Set parameter file and output directories for each input parameter 
        # file
        if isinstance(pfile, str):
            pfile = [pfile]
        if isinstance(pfile, dict):
            input_pfiles = pfile
        else:
            input_pfiles = {os.path.basename(p).replace(".txt", ""): p
                            for p in pfile}
        self.parameter_sets = list(input_pfiles.keys())
        self.outdirs = {p: os.path.join(self.outdir, p) for p in input_pfiles}
        for d in self.outdirs:
            if not os.path.exists(d):
                os.makedirs(d)
        self.pfiles = {}
        for p, file in input_pfiles.items():
            shutil.copy(file, f'{self.outdirs[p]}/InputParameters.txt')

        # Expected locations of outputs
        self.tfiles = [
            os.path.join(d, "TransformParameters.0.txt") for d in self.outdirs
        ]
        self.tfile = self.tfiles[-1]
        self.out_path = os.path.join(self.outdirs[-1], "result.0.nii")
        self.transformed_images = {}
        self.final_image = None

        # Perform registration
        if auto_reg:
            self.register(force)

    def get_nifti_inputs(self, force=False):
        """Write fixed and moving images to nifti files at the top level of
        self.outdir if they do not yet exist."""

        for im in ["fixed", "moving"]:
            if not hasattr(self, f"{im}_path") or force:

                path = os.path.join(self.outdir, f"{im}.nii.gz")
                if not os.path.exists(path) or force:
                    setattr(self, im, ensure_image(getattr(self, im)))
                    path = get_nifti_path(getattr(self, im), path, force)
                setattr(self, f"{im}_path", path)

    def create_elastix_command(self, pfile, outdir, tfile=None):
        """Create elastix command for a given parameter file, output directory,
        and optional initial transform file. This command will be cached in 
        self.cmd.

        Parameters
        ----------
        pfile : str
            String containing the path to an elastix parameter file.

        outdir : str
            Directory in which output will be saved.

        tfile : str, default=None
            Optional path to an elastix transform parameter file which will
            be applied to the moving image before the registration is performed.
        """

        # Create elastix command
        cmd = (
            f"{_elastix} -f {self.fixed_path} -m {self.moving_path} "
            f"-p {pfile} -out {outdir}"
        )
        if tfile is not None:
            self.cmd += f" -t0 {tfile}"

        # Replace all backslashes
        cmd = cmd.replace("\\", "/")
        self.cmd = cmd
        return cmd

    def is_registered(self, p):
        """Check whether registration has already been performed for a given
        parameter set."""

        if isinstance(p, int):
            p = self.parameter_sets[p]
        return os.path.exists(self.tfiles[p])

    def register(self, p=None, force=False, apply=False):
        """Run a registration. By default the registration will be run for
        all parameter sets in self.param_sets, but can optionally be run for
        just one set.

        Parameters
        ----------
        p : int/str, default=None
            Name or number of the parameter set for which the registration 
            should be performed. By default, registration will be performed 
            for all parameter sets in series, using the previous set's
            output transform as input for the next. Available parameter sets 
            are listed in self.parameter_sets.

        force : bool, default=None
            If False and a registration has already been performed for the 
            given parameter (sets), the registration will not be re-run.

        apply : bool, default=True
            If True, the resultant transform will be applied to the moving
            image in order to create a transformed image. If multiple 
            registrations are performed in series, only the final transform
            will be applied.
        """

        # Make list of transforms to apply
        if p is None:
            params = self.param_sets
        elif not isinstance(p, list):
            params = [p]
        else:
            params = p

        # Run elastix for each parameter file
        for par in params:

            # Check if this step has already been done
            if self.is_registered(par) and not force:
                print(f"Registration {par} has already been performed.")
                continue

            # Get input transform file from previous step
            i = self.parameter_sets.index(par)
            tfile = None
            if i != 0:
                prev = self.parameter_sets[i - 1]
                if not is_registered(prev):
                    print(f"Warning: previous registration {prev} has not yet "
                          f"been performed! Registration {par} will be "
                          "performed without input transform."
                         )
                else:
                    tfile = self.tfiles[prev]

            # Run elastix
            self.create_elastix_command(
                pfile, self.outdirs[i], tfile=tfile, force=force_nii_creation
            )
            print("Running command:", self.cmd)
            subprocess.call(self.cmd.split())

        # Get output image
        if apply:
            self.get_result()

    def transform_image(self, im, p=-1, outfile=None):
        """Transform an image using a transform. By default, the final 
        transform in the sequence will be used.

        Parameters
        ----------
        im : Image/str
            Image to transform, or path that can be used to create an image
            object.

        p : int/str
            Name or number of the parameter set whose transform should be used.
            By default, the last parameter set in self.parameter_sets is used.

        outfile : str, default=None
            If set, the image will be written to an output file rather than
            returned.
        """

        # Make temporary output directory
        tmp_dir = f"{self.outdir}/_tmp"
        os.path.mkdir(tmp_dir)

        # Check the transformation for this parameter set exists
        if not self.is_registered(p):
            print(f"Registration {p} has not yet been performed. Run "
                  "self.register() before transforming an image.")
            return

        # Ensure output format in transform file is nifti
        set_parameters(
            self.tfile, {"ResultImageFormat": '"nii"', 
                         "CompressResultImage": '"false"'}
        )

        # Create command
        cmd = f"{_transformix} -in {im_path} -out {outdir} -tp {self.tfile}"
        cmd = cmd.replace("\\", "/")
        print("Running command:", cmd)
        subprocess.call(cmd.split())

        # Copy output to file if requested
        tmp_file = f"{tmp_dir}/result.nii"
        if outfile is not None:
            shutil.copy(tmpfile, outfile)
            shutil.rmdir(tmp_dir)
            return

        # Otherwise, return transformed Image object
        im = Image(tmp_file)
        shutil.rmtree(tmp_dir)
        return im

    def transform_moving_image(self, p=-1):
        """Transform the moving image using a given parameter set and write
        to that parameter set's directory."""

        self.transform_image(
            p=p, outfile=f"{self.outdirs[p]}/transformed_moving.nii.gz")

    def get_final_image(self, force=False):
        """Get moving image with final transform applied."""

        if not is_registered(self.parameter_sets[-1]):
            self.register()
        if self.final_image is not None and not force:
            return self.final_image

        if self.parameter_sets[-1] not in self.transformed_images or force:
            self.transform_moving_image(p=-1)
        self.final = Image(self.out_path)

    def get_comparison(self):

        self.get_final_image()
        self.comparison = ImageComparison(self.fixed, self.final)
        return self.comparison

    def plot_comparison(self, outdir=None, **kwargs):

        if outdir is None:
            outdir = self.outdir

        self.get_comparison()
        for view in ["x-y", "y-z", "x-z"]:
            self.comparison.plot_overlay(
                view, save_as=os.path.join(outdir, f"overlay_{view}.pdf"), **kwargs
            )

    def adjust_pfile(self, params, idx=-1, make_copy=True):
        """Adjust the parameters in a parameter file.

        Parameters:
        -----------
        params : dict
            Dictionary of parameter names and new parameter values.
        idx : int, default=-1
            Index of the parameter file to adjust (by default, this will be the
            last parameter file.
        make_copy : bool, default=True
            If True, a new parameter file will be created in self.outdir rather
            than overwriting the original parameter file.
        """

        # Find name of output parameter file
        if make_copy:
            basename = os.path.basename(self.pfiles[idx]).replace(".txt", "")
            outfile = os.path.join(self.outdir, basename + "_copy.txt")
        else:
            outfile = None

        # Update parameters
        set_parameters(self.pfiles[idx], params, outfile)
        if outfile is not None:
            self.pfiles[idx] = outfile

    def view_comparison(self, **kwargs):
        """View comparison of fixed image and final transformed moving 
        image."""

        from skrt.viewer import QuickViewer

        self.get_final_image()
        if "comparison" not in kwargs:
            kwargs["comparison"] = True
        if "hu" not in kwargs:
            self.fixed = ensure_image(self.fixed)
            kwargs["hu"] = self.fixed.default_window
        if "title" not in kwargs:
            kwargs["title"] = ["Fixed", "Transformed moving"]
        QuickViewer([self.fixed_path, self.out_path], **kwargs)

    def view_init(self, **kwargs):
        """View comparison of initial fixed and moving images."""

        from skrt.viewer import QuickViewer

        if "hu" not in kwargs:
            self.fixed = ensure_image(self.fixed)
            kwargs["hu"] = self.fixed.default_window
        if "match_axes" not in kwargs:
            kwargs["match_axes"] = "y"
        if "title" not in kwargs:
            kwargs["title"] = ["Fixed", "Moving"]
        if "comparison" not in kwargs:
            kwargs["comparison"] = True
        QuickViewer([self.fixed_path, self.moving_path], **kwargs)


def ensure_image(im):
    if isinstance(im, Image):
        return im
    return Image(im)


def get_nifti_path(im_source, outname, force=False):
    """Get path to nifti version of a given image."""

    if os.path.exists(outname) and not force:
        return outname

    if im_source is not None:
        im = ensure_image(im_source)
        if im.source_type == "nifti":
            return im.path
        im.write(outname)
        return outname
    else:
        raise TypeError("Must give a valid input image!")


def set_parameters(tfile, params, output_tfile=None):
    """Replace value(s) of parameter(s) in an elastix transform file."""

    # Overwrite input file if no output given
    if output_tfile is None:
        output_tfile = tfile

    # Read input
    with open(tfile) as file:
        lines = "".join(file.readlines())

    # Modify
    for name, value in params.items():
        lines = re.sub(f"\({name}.*\)", f"({name} {value})", lines)

    # Write to output
    with open(output_tfile, "w") as file:
        file.write(lines)

