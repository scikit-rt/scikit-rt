'''
Script for copying InnerEye outputs to allow loading by scikit-rt Patient.
'''
from pathlib import Path
from shutil import copy2, rmtree
from time import strftime

import pandas as pd

from skrt import StructureSet
from skrt.core import generate_timestamp

def copy_data(row, indir=None, outdir=None, timestamp=None):
    '''
    Copy data based on single row in a dataframe.

    **Parameters:**

    row : pandas.Series
        Series corresponding to a row in a data index as required
        by InnerEye:
        https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/creating_dataset.md

    indir : patlib.Path, default=None
        Path to directory from which data are to be copied.

    outdir : pathlib.Path, default=None
        Path to directory to which data are to be copied.

    timestamp : str, default=None
        Timestamp appended to sub-directories for copied data.
    '''
    # Define path to data for patient referenced in current row.
    indir_patient = indir / f'{row.subject:03}'
    if not indir_patient.exists():
        return

    # Extract information from path to data file referenced in current row.
    patient_id, study_timestamp, filename = row.filePath.split('/')
    print(f'Patient: {indir_patient.name} {patient_id}')
    elements = filename.split('_')
    modality = Path(*elements[0: 2])
    modality_timestamp = '_'.join(elements[2: 5])
    modality_dir = (outdir / patient_id / study_timestamp / modality
            / modality_timestamp)
    nifti_dir = outdir / patient_id / study_timestamp / 'NIfTI'

    # Loop over InnerEye results in patient directory.
    for inpath in indir_patient.glob('*.nii.gz'):
        # Determine result type.
        result_type = inpath.with_suffix('').stem.split('_')[0]

        # Skip the multi-label segmentation - will save the ROI binary masks.
        if 'segmentation' == result_type:
            continue

        if result_type not in ['posterior', 'uncertainty']:
            # Result is a binary map.
            outpath = modality_dir / f'innereye_{timestamp}' / inpath.name
        else:
            # Result is a map of posterior probability or Shannon entropy
            outpath = nifti_dir / inpath.name

        # Create output directory if it doesn't exist, and copy file.
        outpath.parent.mkdir(parents=True, exist_ok=True)
        copy2(inpath, outpath)

def select_and_copy(topdir=None, experiments=None, outdir=None,
        recreate_outdir=True):
    '''
    Select and copy data from InnerEye results directory.

    **Parameters:**

    topdir : pathlib.Path/str, default=None
        Path to top-level directory containing InnerEye experiment directories.

    experiments : dict, default=None
        Dictionary where keys are sub-directories to be created in outdir,
        and values are run directories in topdir.

    outdir : pathlib.Path/str, default=None
        Path to directory to which data are to be copied.

    recreate_outdir : bool, default=True
        If True, delete and recreate outdir if it already exists.
    '''

    # Ensure that paths to directories are pathlib.Path objects.
    topdir = Path(topdir)
    outdir = Path(outdir)

    # Ensure that output directory exists, deleting and recreating if required.
    if outdir.exists() and recreate_outdir:
        rmtree(outdir)
    outdir.mkdir(parents=True)


    # Generate timestamp with which to tag outputs.
    timestamp = generate_timestamp()

    # Loop over experiments.
    if experiments is None:
        experiments = {}
    for experiment, run_id in experiments.items():
        indir = topdir / run_id / 'best_validation_epoch/Test'
        dataset_csv = indir / 'dataset.csv'
        # Load data index into pandas dataframe.
        df = pd.read_csv(dataset_csv)
        # Extract information for ROIs.
        df_rois = df[df['channel'] != 'ct'].copy()
        # Keep only one row per patient.
        df_rois.drop_duplicates(subset='subject', inplace=True)
        # Copy patient data.
        df_rois.apply(copy_data, axis=1, indir=indir,
                      outdir=outdir / experiment, timestamp=timestamp)

if '__main__' == __name__:

    # Path to top-level directory containing InnerEye experiment directories.
    topdir = Path('/Users/karl/Desktop/ExperimentRun')

    # Dictionary where keys are sub-directories to be created in output
    # directory and values are run directories in input directory.
    experiments = {
        'head_and_neck_smg':
        'dcid.HD_37d263b2-0978-4426-9010-af56cd04bd0c',
        'head_and_neck_multiple5':
        'dcid.HD_fe7ddde0-f20b-4cb0-90a1-a9c693b8be34',
        }

    # Path to directory to which data are to be copied.
    outdir = Path('/Users/karl/data/innereye_results')

    # Select and copy data from InnerEye results directory.
    select_and_copy(topdir, experiments, outdir)
