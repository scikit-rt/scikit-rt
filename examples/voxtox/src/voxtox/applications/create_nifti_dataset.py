'''
Application for converting patient data to NIfTI format.
'''

import glob
import os
import shutil
import sys

import numpy as np

from skrt.application import Algorithm, Application
from skrt.core import fullpath

import voxtox.roi_names.head_and_neck_roi_names as h_names
import voxtox.roi_names.prostate_roi_names as p_names

class CreateNiftiDataset(Algorithm):
    '''
    Algorithm subclass, for converting a patient's imaging data
    and structure sets to NIfTI format.

    Methods:
        __init__ -- Return instance of CreateNiftiDataset class,
                    with properties set according to options dictionary.
        execute  -- Convert patient's imaging data and structure sets
                    to NIfTI format.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of CreateNIfTIDataset class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        # Output directory.
        self.out_dir = './nifti_dataset'

        # Flag indicating whether to recreate output directory.
        self.recreate_out_dir = False

        # Constrain types of imaging data to be saved
        # (None to save all, empty list to exclude all)
        self.image_types = None

        # For each imaging type, list of identifiers of objects to be saved
        # in NIfTI format
        # 0    -- least recent
        # -1   -- most recent
        # None -- all
        self.files = None

        # For each imaging type, list of indices of structure-set objects
        # to be saved for each image object
        self.times = None

        # Identifier of studies to be considered
        # 0    -- least recent only
        # -1   -- most recent only
        # None -- all
        self.studies = None

        # Value bandings to be applied before saving imaging data;
        # for example:
        #     self.bands{'mvct' : {-100: -1024, 100: 0, 1e10: 1024}}
        # will band an image of type 'mvct':
        #     value <= -100 => -1024
        #     -100 < value <= 100 => 0
        #     100 < value <= 1e10 => 1024
        self.bands = {}

        # Fill value to use when interpolating image values outside
        # data area; if set to None, the lowest value of the original
        # image is used
        self.fill_value = None

        # Image types for which data are to be written
        # only if associated structure-set found
        self.require_structure_set = []

        # ROI names (list, dictionary, None), as accepted by
        # voxtox.rtstruct.save_structs_as_nifti()
        self.roi_names = None

        # If True, force writing of dummy NIfTI file for all named ROIs
        # not found in structure-set data.  If False, disregard these
        # ROIs.
        self.force_roi_nifti = False

        # List of ROIs that should be split into left and right parts.
        self.bilateral_names = []

        # Image dimensions (dx, dy, dz) in voxels for image resizing
        # prior to writing.  If None, original image size is kept.
        self.image_size = None

        # Voxel dimensions (dx, dy, dz) in mm for image resizing
        # prior to writing.  If None, original voxel size is kept.
        self.voxel_size = None

        # Flag indicating whether to report progress.
        self.verbose = True

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Obtain full path to output directory.
        self.out_dir = fullpath(self.out_dir)

        # Create, or recreate, the output directory as needed.
        if self.recreate_out_dir:
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Convert to single-element lists where needed.
        if isinstance(self.studies, int):
            self.studies = [self.studies]
        if isinstance(self.image_types, str):
            self.image_types = [self.image_types]

    def execute(self, patient=None):

        '''
        Convert patient's imaging data and structure sets to NIfTI format.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        '''

        # Print details of current patient.
        print(f'Patient id: {patient.id}')
        print(f'Folder path: {patient.path}')

        # Create, or recreate, the patient's output directory.
        out_dir_patient = f'{self.out_dir}/{patient.id}'
        if os.path.exists(out_dir_patient):
            shutil.rmtree(out_dir_patient)

        # Define list of studies to be processed
        if self.studies is None:
            studies = patient.studies
        else:
            studies = [patient.studies[idx] for idx in self.studies]

        # For each study, convert data from DICOM to NIfTI
        for study in studies:
            out_dir_study = f'{out_dir_patient}/{study.timestamp}'
            os.makedirs(out_dir_study)
            
            study.save_images_as_nifti(out_dir=out_dir_study,
                    image_types = self.image_types, times=self.times,
                    verbose=self.verbose, image_size=self.image_size,
                    voxel_size=self.voxel_size, fill_value=self.fill_value,
                    bands=self.bands,
                    require_structure_set=self.require_structure_set)

            study.save_structure_sets_as_nifti(out_dir=out_dir_study,
                    image_types = self.image_types, times=self.times,
                    files = self.files, verbose=self.verbose,
                    image_size=self.image_size, voxel_size=self.voxel_size,
                    roi_names=self.roi_names,
                    force_roi_nifti=self.force_roi_nifti,
                    bilateral_names=self.bilateral_names)

        return self.status


if '__main__' == __name__:

    # Create list of paths to patient data
    path = '/r02/voxtox/kh_synthetic/transform_derangement_000/head_and_neck/consolidation/VT1_H_22EEAK1L'
    path = '/Users/karl/data/head_and_neck/partiii/VT1_H_A98302K1'
    sys.argv.append(path)
    arg_ok = (len(sys.argv) > 1)
    if arg_ok:
        path = sys.argv[1]
        arg_ok = isinstance(path, str)
    if arg_ok:
        paths = glob.glob(path, recursive=True)
        paths.sort()
        arg_ok = (len(paths) > 0)

    # Create and run algorithm to save data in NIfTI format
    if arg_ok:
        opts = {}
        opts['image_types'] = ['ct']
        opts['structure_set_types_save'] = ['ct']
        opts['roi_names'] = h_names.head_and_neck_plan
        opts['voxel_size'] = None
        opts['recreate_out_dir'] = False
        opts['times'] = {'ct': [-1]}
        opts['files'] = {'ct': [0]}
        opts['verbose'] = True
        algs = [CreateNiftiDataset(opts)]
        app = Application(algs)
        app.run(paths)

    # Print usage information
    else:
        print(
                '\nUsage: python create_nifti_dataset.py <path>'
                '\n\n       <path> -- path identifying patient folders,'
                '\n       to be expanded as list by glob.glob(<path>)'
            )

        if path and not paths:
            print(f'\nNo data found at path \'{path}\'')
