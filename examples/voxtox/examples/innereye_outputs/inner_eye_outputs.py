'''
Script for copying InnerEye outputs to allow loading by scikit-rt Patient.
'''
from pathlib import Path
from shutil import copy2, rmtree
from time import strftime

import pandas as pd

from skrt import StructureSet

def copy_data(row, indir=None, outdir=None):
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
    innereye_dir = outdir / patient_id / study_timestamp / 'InnerEye'

    # Loop over InnerEye results in patient directory.
    for inpath in indir_patient.glob('*.nii.gz'):
        # Determine result type.
        result_type = inpath.with_suffix('').stem.split('_')[0]

        # Skip the multi-label segmentation - will save the ROI binary masks.
        if 'segmentation' == result_type:
            continue

        if result_type not in ['posterior', 'uncertainty']:
            # Result is a binary map.
            outpath = (modality_dir / f'innereye_{modality_timestamp}' /
                    inpath.name)
        else:
            # Result is a map of posterior probability or Shannon entropy
            outpath = innereye_dir / f'{modality_timestamp}_{inpath.name}'

        # Create output directory if it doesn't exist, and copy file.
        outpath.parent.mkdir(parents=True, exist_ok=True)
        copy2(inpath, outpath)

def select_and_copy(topdir=None, subdir=None, experiments=None, outdir=None,
        recreate_outdir=True):
    '''
    Select and copy data from InnerEye results directory.

    **Parameters:**

    topdir : pathlib.Path/str, default=None
        Path to top-level directory containing InnerEye experiment directories.

    subdir : pathlib.Path/str, default=None
        Outputs subdirectory in best_validation_epoch.  If None, take
        the subdirectory to have the same name as the last element of
        the output directory.

    experiments : dict, default=None
        Dictionary where keys are sub-directories to be created in outdir,
        and values are run directories, or lists of run directories, in topdir.

    outdir : pathlib.Path/str, default=None
        Path to directory to which data are to be copied.

    recreate_outdir : bool, default=True
        If True, delete and recreate outdir if it already exists.
    '''

    # Ensure that paths to directories are pathlib.Path objects,
    # and that subdir is defined.
    topdir = Path(topdir)
    outdir = Path(outdir)
    if subdir is None:
        subdir = outdir.name

    # Ensure that output directory exists, deleting and recreating if required.
    if outdir.exists() and recreate_outdir:
        rmtree(outdir)
    outdir.mkdir(parents=True)

    # Ensure that dictionary values are lists
    if experiments is None:
        experiments = {}
    for key in experiments.keys():
        if not isinstance(experiments[key], list):
            experiments[key] = [experiments[key]] if experiments[key] else None

    # Loop over experiments.
    for experiment, run_ids in experiments.items():
        for run_id in run_ids:
            indir = (topdir / 'ExperimentRun' / run_id
                    / 'best_validation_epoch' / subdir)
            dataset_csv = indir / 'dataset.csv'
            # Load data index into pandas dataframe.
            df = pd.read_csv(dataset_csv)
            # Extract information for ROIs.
            df_rois = df[df['channel'] != 'ct'].copy()
            # Keep only one row per patient.
            df_rois.drop_duplicates(subset='subject', inplace=True)
            # Copy patient data.
            df_rois.apply(copy_data, axis=1, indir=indir,
                    outdir=outdir / experiment)

if '__main__' == __name__:

    # Path to top-level directory containing InnerEye experiment directories.
    topdir = Path('/Users/karl/data/InnerEye/head_and_neck/azure')
    topdir = Path('/Users/karl/data/azure_downloads')

    # Dictionary where keys are sub-directories to be created in output
    # directory and values are run directories in input directory.
    experiments = {
        'head_and_neck_smg_dice': 'smgs_dice',
        'head_and_neck_smg_focal': 'smgs_focal',
        'head_and_neck_pg_sc_dice': 'parotids_dice',
        'head_and_neck_pg_sc_focal': 'parotids_focal',
        }

    experiments = {
        'head_and_neck_smg_dice':
        'dcid.HD_37d263b2-0978-4426-9010-af56cd04bd0c',
        'head_and_neck_smg_focal':
        'dcid.HD_78aa040d-35c4-478d-b523-02abc3e6475a',
        'head_and_neck_pg_sc_dice':
        'dcid.HD_14630e58-671a-47a0-b4ed-927c98bb1864',
        'head_and_neck_pg_sc_focal':
        'dcid.HD_ba6d07c9-18ba-4704-b363-886dcf11c2fa',
        }

    for experiment, run_id in experiments.items():
        run_ids = []
        for idx in range(4):
            run_ids.append(f'{run_id}_{idx}')
        experiments[experiment] = run_ids

    # Path to directory to which data are to be copied.
    outdir = Path(
            '/Users/karl/data/InnerEye/head_and_neck/innereye_results/Test')
    outdir = Path(
            '/Users/karl/data/InnerEye/head_and_neck/innereye_results/Val')

    # Select and copy data from InnerEye results directory.
    select_and_copy(topdir=topdir, experiments=experiments, outdir=outdir)
