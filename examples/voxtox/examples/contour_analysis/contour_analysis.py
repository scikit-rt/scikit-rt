'''
Application for analysing contours for VoxTox planning scans.
'''

import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

from skrt.application import Algorithm, Application
from skrt.core import fullpath

from voxtox.data.djn_253 import djn_253
from voxtox.data.jes_109 import jes_109
from voxtox.roi_names import get_roi

class ContourAnalysis(Algorithm):
    '''
    Algorithm subclass, for contour analysis.

    Methods:
        __init__ -- Return instance of ContourAnalysis class,
                    with properties set according to options dictionary.
        execute  -- Perform analysis.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of ContourAnalysis class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''

        # Names of ROIs for which analysis is to be performed.
        self.roi_names = []

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        self.roi_data = []

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

        for study in patient.studies:
            if len(study.ct_structure_sets) < 2:
                continue
            study.ct_structure_sets.sort()
            ss_plan = study.ct_structure_sets[0]
            ss_voxtox = study.ct_structure_sets[-1]

            for roi_name in self.roi_names:
                roi1 = get_roi(ss_plan, roi_name, 'plan')
                roi2 = get_roi(ss_voxtox, roi_name, 'voxtox')
                if roi1 is None or roi2 is None:
                    continue
                # Exclude cases where two ROIs are probably the same.
                if roi1.get_dice(roi2) > 0.99:
                    continue
                z_coords = set(roi1.get_polygons().keys()).intersection(
                        set(roi2.get_polygons().keys()))
                for z in sorted(z_coords):
                    dx, dy = roi1.get_centroid_distance(
                        roi2, single_slice=True, pos=z)
                    dice = roi1.get_dice(roi2, single_slice=True, pos=z)
                    area1 = roi1.get_area(pos=z)
                    area2 = roi2.get_area(pos=z)
                    unsigned_msd = roi1.get_mean_surface_distance(
                            roi2, single_slice=True, pos=z)
                    signed_msd = roi1.get_mean_surface_distance(
                            roi2, signed=True, single_slice=True, pos=z)
                    rmssd = roi1.get_rms_surface_distance(
                            roi2, single_slice=True, pos=z)
                    self.roi_data.append({
                            'id': patient.id, 'study': study.timestamp,
                            'roi': roi1.name, 'z': z, 'dx': dx, 'dy': dy,
                            'area1': area1, 'area2': area2,
                            'area_ratio': area2 / area1, 'dice': dice,
                            'umsd': unsigned_msd, 'smsd': signed_msd,
                            'rmssd': rmssd
                            })

        return self.status

    def finalise(self):

        df = pd.DataFrame(self.roi_data)
        df.to_csv('roi_data.csv', index=False)

        return self.status


def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    opts['roi_names'] = ['spinal_cord']

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = ContourAnalysis(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

def get_paths():
    # Define the patient data to be analysed
    data_dir = '/Users/karl/data/head_and_neck/partiii'
    #paths = glob.glob(f'{data_dir}/VT*')
    paths = djn_253

    return paths

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the patient data to be analysed
    paths = get_paths()

    # Run application for the selected data
    app.run(paths)

if 'Ganga' in __name__:
    # Define script for setting analysis environment
    setup_script = fullpath('skrt_conda.sh')

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), setup_script)

    # Define the patient data to be analysed
    paths = get_paths()
    input_data = PatientDataset(paths=paths)

    # Define processing system
    backend = Local()
    #backend = Condor()

    # Define how job should be split into subjobs
    splitter = PatientDatasetSplitter(patients_per_subjob=65)

    # Define merging of subjob outputs
    merger = SmartMerger()
    merger.files = ['stderr', 'stdout']
    merger.ignorefailed = True
    postprocessors = [merger]

    # Define job name
    name = 'contour_analysis'

    # Define list of outputs to be saved
    outbox = []

    # Create the job, and submit to processing system
    j = Job(application=ganga_app, backend=backend, inputdata=input_data,
            outputsandbox=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
