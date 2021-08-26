'''Tools for performing image registration.'''

import os
import re
import subprocess

from skrt.image import Image, ImageComparison


class Registration:
    def __init__(
        self,
        fixed,
        moving,
        pfile,
        outdir='.',
        auto_reg=True,
        force=False
    ):
        '''Register moving image to fixed image with a given elastix parameter
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
            Path to an elastix parameter file.
        outdir : str, default='.'
            Path to output directory.
        auto_reg : bool, default=True
            If True, registration will be performed immediately.
        '''

        # Set up fixed and moving images
        self.fixed = fixed
        self.moving = moving
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.get_nifti_inputs()
        self.registered = False

        # Parameter file(s) and output location(s)
        self.pfiles = pfile
        if isinstance(pfile, str):
            self.pfiles = [pfile]
        self.outdirs = [
            os.path.join(self.outdir, os.path.basename(p).replace('.txt', ''))
            for p in self.pfiles
        ]
        for d in self.outdirs:
            if not os.path.exists(d):
                os.makedirs(d)

        # Expected locations of outputs
        self.tfiles = [
            os.path.join(d, 'TransformParameters.0.txt') for d in self.outdirs
        ]
        self.tfile = self.tfiles[-1]
        self.out_path = os.path.join(self.outdirs[-1], 'result.0.nii')

        # Perform registration
        if auto_reg:
            self.register(force)

    def get_nifti_inputs(self, force=False):
        '''Ensure nifti versions of fixed and moving images exist.'''

        for im in ['fixed', 'moving']:
            if not hasattr(self, f'{im}_path') or force:

                path = os.path.join(self.outdir, f'{im}.nii.gz')
                if not os.path.exists(path):
                    setattr(self, im, ensure_image(getattr(self, im)))
                    path = get_nifti_path(getattr(self, im), path, force)

                setattr(self, f'{im}_path', path)

    def create_elastix_command(self, pfile, outdir, tfile=None, force=False):
        '''Create elastix command.'''

        # Create elastix command
        self.cmd = (
            f'elastix -f {self.fixed_path} -m {self.moving_path} '
            f'-p {pfile} -out {outdir}'
        )
        if tfile is not None:
            self.cmd += f' -t0 {tfile}'

    def register(self, force=False, apply=False):

        # Run elastix
        if not self.registered:
            for i, pfile in enumerate(self.pfiles):

                # Check if this step has already been done
                if os.path.exists(self.tfiles[i]) and not force:
                    print(f'Transform {self.tfiles[i]} already exists')
                    continue

                # Create elastix command
                force_nii_creation = force and i == 0
                tfile = None if i == 0 else self.tfiles[i - 1]
                self.create_elastix_command(
                    pfile, self.outdirs[i], tfile=tfile, 
                    force=force_nii_creation
                )

                # Run elastix
                print('Running command:', self.cmd)
                subprocess.call(self.cmd.split())

        self.registered = True

        # Get output image
        if apply:
            self.get_final_image()

    def transform_image(self, im_path, outdir=None, force=False):
        '''Transform an image at a given path using own final transform.'''

        out_path = os.path.join(self.outdirs[-1], 'result.nii')
        if os.path.exists(out_path) and not force:
            return out_path

        if not self.registered:
            self.register()
        if outdir is None:
            outdir = self.outdirs[-1]

        # Ensure output format is nifti
        set_parameters(
            self.tfile, {'ResultImageFormat': ''nii'', 
                         'CompressResultImage': ''false''}
        )

        cmd = f'transformix -in {im_path} -out {outdir} -tp {self.tfile}'
        print('Running command:', cmd)
        subprocess.call(cmd.split())
        return out_path

    def get_final_image(self, force=False):
        '''Get transformed moving image.'''

        if not self.registered:
            self.register()
        if hasattr(self, 'final') and not force:
            return self.final

        if not os.path.exists(self.out_path) or force:
            self.out_path = self.transform_image(
                self.moving_path, self.outdirs[-1], force
            )
        self.final = Image(self.out_path)

    def get_comparison(self):

        self.get_final_image()
        self.comparison = ImageComparison(self.fixed, self.final)
        return self.comparison

    def plot_comparison(self, outdir=None, **kwargs):

        if outdir is None:
            outdir = self.outdir

        self.get_comparison()
        for view in ['x-y', 'y-z', 'x-z']:
            self.comparison.plot_overlay(
                view, save_as=os.path.join(outdir, f'overlay_{view}.pdf'), 
                **kwargs
            )

    def adjust_pfile(self, params, idx=-1, make_copy=True):
        '''Adjust the parameters in a parameter file.

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
        '''

        # Find name of output parameter file
        if make_copy:
            basename = os.path.basename(self.pfiles[idx]).replace('.txt', '')
            outfile = os.path.join(self.outdir, basename + '_copy.txt')
        else:
            outfile = None

        # Update parameters
        set_parameters(self.pfiles[idx], params, outfile)
        if outfile is not None:
            self.pfiles[idx] = outfile

    def view_comparison(self, **kwargs):
        '''View comparison of fixed image and final transformed moving 
        image.'''

        from skrt.viewer import QuickViewer

        self.get_final_image()
        if 'comparison' not in kwargs:
            kwargs['comparison'] = True
        if 'hu' not in kwargs:
            self.fixed = ensure_image(self.fixed)
            kwargs['hu'] = self.fixed.default_window
        if 'title' not in kwargs:
            kwargs['title'] = ['Fixed', 'Transformed moving']
        QuickViewer([self.fixed_path, self.out_path], **kwargs)

    def view_init(self, **kwargs):
        '''View comparison of initial fixed and moving images.'''

        from skrt.viewer import QuickViewer

        if 'hu' not in kwargs:
            self.fixed = ensure_image(self.fixed)
            kwargs['hu'] = self.fixed.default_window
        if 'match_axes' not in kwargs:
            kwargs['match_axes'] = 'y'
        if 'title' not in kwargs:
            kwargs['title'] = ['Fixed', 'Moving']
        QuickViewer([self.fixed_path, self.moving_path], **kwargs)


def ensure_image(im):
    if isinstance(im, Image):
        return im
    return Image(im)


def get_nifti_path(im_source, outname, force=False):
    '''Get path to nifti version of a given image.'''

    if os.path.exists(outname) and not force:
        return outname

    if im_source is not None:
        im = ensure_image(im_source)
        if im.source_type == 'nifti':
            return im.path
        im.write(outname)
        return outname
    else:
        raise TypeError('Must give a valid input image!')


def set_parameters(tfile, params, output_tfile=None):
    '''Replace value(s) of parameter(s) in an elastix transform file.'''

    # Overwrite input file if no output given
    if output_tfile is None:
        output_tfile = tfile

    # Read input
    with open(tfile) as file:
        lines = ''.join(file.readlines())

    # Modify
    for name, value in params.items():
        lines = re.sub(f'\({name}.*\)', f'({name} {value})', lines)

    # Write to output
    with open(output_tfile, 'w') as file:
        file.write(lines)
