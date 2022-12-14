'''
Script for cataloguing NIfTI dataset, in form expected by InnerEye.
'''

from pathlib import Path
import sys

import nibabel

def catalogue_dataset(
        dataset_dir='./nifti_dataset', out_csv='dataset.csv',
        roi_names=None, nz_min=16, all_in_scan=False):
    '''
    Catalogue files of NIfTI dataset, writing information to
    file of comma-separated values, in the format required for
    use with the InnerEye software.  The values stored are:

    - subject: unique positive integer;
    - filePath: file path from top-level directory of dataset;
    - channel: string identifying file as scan data or roi-speficic label mask;
    - seriesId: left blank;
    - institutionId: left blank;
    - DIM_X: size of image array along x-axis;
    - DIM_Y: size of image array along y-axis;
    - DIM_Z: size of image array along z-axis.

    **Parameters:**

    dataset_dir : str, default='./nifti_dataset'
        Path to directory containing NIfTI dataset.

    out_csv : str, default='dataset.csv'
        Name for output CSV file, which will be placed inside dataset_dir.

    roi_names : list, default=None
        List of ROI names.  if non-empty, require and catalogue
        listed ROIs.  If empty, catalogue all ROIs.

    nz_min : int, default=16
        Minimum number of image slices required for data to be catalogued.

    all_in_scan : bool, default=False
        If True, require that all ROIs listed in roi_names be fully inside
        scan.  This can be useful for compact ROIs, for example salivary
        glands, but less useful for extended ROIs, for example spinal cord.
    '''

    # Convert string defining dataset directory to a pathlib Path.
    dataset_dir = Path(dataset_dir)

    # Create dictionary containing values to store for each patient,
    # and list of paths to patient data.
    patients = {}
    patient_paths = sorted(list(dataset_dir.iterdir()))

    # Loop over paths.
    idx = 0
    for patient_path in patient_paths:
        # Skip paths that aren't data directories.
        if not patient_path.is_dir():
            continue
        idx += 1
        patient_id = patient_path.name
        print(f' Patient {idx:5} - {patient_id}')

        # Only consider initial study (i.e. exclude replan data).
        study_path = sorted(list(patient_path.iterdir()))[0]

        # Deal with a single modality.
        modality_paths = sorted(list(study_path.iterdir()))
        assert len(modality_paths) == 1
        modality_path = modality_paths[0]

        # Deal with a single timestamp.
        timestamp_paths = sorted(list(modality_path.iterdir()))
        assert len(timestamp_paths) == 1
        timestamp_path = timestamp_paths[0]

        # Loop over paths to NIfTI files.
        nifti_paths = sorted(list(timestamp_path.iterdir()))
        for nifti_path in nifti_paths:
            # Determine and check channel.
            elements = nifti_path.name.split('.')[0].split('_')
            channel_ok = True
            if 'RTSTRUCT' == elements[0]:
                channel = '_'.join(elements[5:])
                channel_ok = (channel in roi_names) or (not roi_names)
            elif elements[0] in ['MVCT', 'CTHD']:
                channel = 'ct'
            else:
                channel = elements[0].lower()
            if not channel_ok:
                continue

            # Load NIfTI data.
            nifti_object = nibabel.load(nifti_path)
            fdata = nifti_object.get_fdata()
            nx, ny, nz = fdata.shape

            # Require number of image slices above a minimum.
            if nz >= nz_min:
                # Store data cor each channel.
                if not patient_id in patients:
                    patients[patient_id] = {}
                    patients[patient_id]['channels'] = []
                patients[patient_id][channel] = (
                    nifti_path.relative_to(dataset_dir), '', '', nx, ny, nz)
                patients[patient_id]['channels'].append(channel)
            else:
                print(f'Patient {patient_id} - {nz} slices - skipping')
                break

            # Require that named ROIs be at partly or, if all_in_scan is True,
            # fully inside scan.
            if (roi_names) and (channel in roi_names):
                in_scan = True
                if not int(fdata.max() + 0.1):
                    in_scan = False
                elif all_in_scan:
                    if int(fdata[:,:,0].max() + 0.1):
                        in_scan = False
                    if int(fdata[:,:,-1].max() + 0.1):
                        in_scan = False

                if not in_scan:
                    print(f'Patient {patient_id} - {channel} not in scan')
                    patients[patient_id]['channels'].remove(channel)

        # Delete patient information if not all listed ROIs present.
        if roi_names:
            if patient_id in patients:
                missing_list = []
                for roi_name in roi_names:
                    if not roi_name in patients[patient_id]['channels']:
                        missing_list.append(roi_name)
            else:
                missing_list = list(roi_names)

            if missing_list:
                print(f'Patient {patient_id} - '
                      f'missing {missing_list} - skipping')
                if patient_id in patients:
                    del patients[patient_id]

    # Write to CSV file values required by InnerEye software
    lines = ['subject,filePath,channel,seriesId,institutionId,'
             'DIM_X,DIM_Y,DIM_Z']
    for index, patient_id in enumerate(sorted(patients)):
        subject = index + 1
        for channel in patients[patient_id]['channels']:
            file_path, series_id, institution_id, nx, ny, nz = (
                    patients[patient_id][channel])
            lines.append(f'{subject},{file_path},{channel},{series_id},'
                         f'{institution_id},{nx},{ny},{nz}')

    out_path = dataset_dir / out_csv
    out_file = open(out_path, 'w')
    out_file.write('\n'.join(lines))
    out_file.close()

    return None

if '__main__' == __name__:

    # Accept values for dataset_dir and out_csv given at command line,
    # or use defaults.
    default_dataset_dir = '/r02/voxtox/workshop/ie_datasets/prostate'
    default_out_csv = 'dataset.csv'
    default_roi_names = ['bladder', 'femoral_head_left', 'femoral_head_right',
            'prostate', 'rectum']
    default_dataset_dir = '/r02/voxtox/workshop/ie_datasets/head_and_neck'
    default_out_csv = 'dataset.csv'
    default_roi_names = ['brainstem', 'mandible', 'parotid_left',
            'parotid_right', 'smg_left', 'smg_right', 'spinal_cord']

    if (len(sys.argv) > 1):
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = default_dataset_dir
    if (len(sys.argv) > 2):
        out_csv = sys.argv[2]
    else:
        out_csv = default_out_csv
    if (len(sys.argv) > 3):
        roi_names = sys.argv[3]
    else:
        roi_names = default_roi_names

    if Path(dataset_dir).is_dir():
        catalogue_dataset(dataset_dir, out_csv, roi_names)

    # Print usage information
    else:
        print(
                '\nUsage: python catalogue_nifti_dataset.py [<dataset_dir>]'
                ' [<out_csv]'
                '\n\n       <dataset_dir> -- path to dataset directory'
                f' (default: {default_dataset_dir})'
                '\n       <out_csv> -- name of CSV output file'
                f' (default: {default_out_csv})'
             ) 
