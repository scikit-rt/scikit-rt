'''
Module for creating lists of ROI names.

This modules provides functions for extracting all names used for all
ROIs in a collection of datasets:

- get_config():
  Retreive configuration information for some datasets of interest.
- get_info():
  Retrieve information for dataset on patient numbers and on ROI names.
- print_info():
  Print out, for each disease, patient numbers and ROI names.
- run():
  Perform processing, taking into account command-line arguments.

The idea is that the output should be directed to an output file
with extension .py, to that the lists of ROI names can be processed
using the functions of sort_rois.py.
'''

from pathlib import Path
from sys import argv

from skrt import Patient
from voxtox.data import djn_253, jes_109

def get_config(mode='ct'):
    '''
    Retreive configuration information for some datasets of interest.

    **Parameter:**

    mode : str, default='ct'
        Identifier of dataset to be considered.  Recognised values are:

        - 'ct': structure sets for all planning scans;
        - 'jes_109': structure sets for planning scans outlined by JES;
        - 'djn_253': structure sets for planning scans outlined by DJN;
        - 'mvct' : structure sets for manually outlined treatment scans.
    '''

    top_dir = Path('/r02/voxtox/data')
    cohorts = ['consolidation', 'discovery', 'error_cases/consolidation',
               'error_cases/discovery', 'special_cases/consolidation',
               'special_cases/discovery']
    diseases = ['head_and_neck', 'prostate']
    ids = []
    image_type = 'ct'

    if 'jes_109' == mode:
        diseases = ['prostate']
        ids = jes_109
    elif 'djn_253' == mode:
        diseases = ['head_and_neck']
        ids = djn_253
    elif 'mvct' == mode:
        image_type = 'mvct'
        cohorts = ['vspecial']
    
    return(top_dir, diseases, cohorts, image_type, ids)

def get_info(top_dir= '.', diseases=[], cohorts=[], image_type='ct',
             ids=[], max_patient=1000, verbose=True):
    '''
    Retrieve information for dataset on patient numbers and on ROI names.

    **Parameters:**

    top_dir : str, default='.'
        Path to top level of directory tree (top_dir/disease/cohort)
        containing datasets.

    diseases : list, default=[]
        List of disease subdirectories to be considered.

    cohorts : list, default=[]
        List of cohort subdirectories to be considered.

    image_type : str, default='ct'
        Type of image for which associated structure sets are
        to be examined to extract ROI names.  Valid values are
        'ct' and 'mvct'.

    ids : list, default=[]
        If non-empty, only datasets with ids listed are to be considered.

    max_patient : int, default=1000
        Maximum number of datasets to be processed.

    verbose, bool, default=True
        If True, print out the id of the dataset being processed.
    '''

    # Convert path to a pathlib.Path object
    top_dir = Path(top_dir)

    # Initialise dictionaries
    n_patient = {'all_patients': 0}
    rois_plan = {}
    rois_voxtox = {}

    # Loop over diseases
    for disease in diseases:
        n_patient[disease] = 0
        rois_plan[disease] = []
        if 'mvct' == image_type:
            rois_voxtox[disease] = {}
        else:
            rois_voxtox[disease] = []
        # Loop over cohorts
        for cohort in cohorts:
            cohort_dir = top_dir / disease / cohort
            if 'mvct' == image_type:
                cohort_paths = [x for x in  cohort_dir.glob('*')
                                if x.is_dir()]
            else:
                cohort_paths = [cohort_dir]
            for cohort_path in cohort_paths:
                if ('autocontouring' in str(cohort_path)
                    or 'tmp' in str(cohort_path)
                    or 'sinogram' in str(cohort_path)):
                    continue
                if 'mvct' == image_type:
                    # For 'mvct' image type, recursively search directories
                    # for patient directories.
                    patient_dirs = cohort_path.glob('**/VT*')
                else:
                    # For 'ct' image type, only search cohort-level
                    # directory for patient directories.
                    patient_dirs = cohort_dir.glob('VT*')

                # Loop over patient directories.
                for patient_dir in patient_dirs:
                    if n_patient['all_patients'] >= max_patient:
                        break
                    if ids and patient_dir.name not in ids:
                        continue
                    if verbose:
                        print(patient_dir)
                    n_patient['all_patients'] += 1
                    n_patient[disease] += 1
                    # For 'mvct' image type take directory containing
                    # data to define a category.
                    if 'mvct' == image_type:
                        category = patient_dir.parent.name
                        rois_voxtox[disease][category] = []
                    p = Patient(str(patient_dir))
                    # Loop over studies.
                    for s in p.studies:
                        if not hasattr(s, f'{image_type}_structure_sets'):
                            continue
                        if 'mvct' == image_type:
                            # For 'mvct' image type, extract ROI names
                            # for all of an image's structure sets
                            for mvct_ss in s.mvct_structure_sets:
                                rois_voxtox[disease][category].extend(
                                    mvct_ss.get_roi_names())
                        else:
                            # For 'ct' image type, extract ROI names
                            # for an image's first structure set
                            # (clinical planning data) and for the
                            # image's last structure set (outlining
                            # for VoxTox).
                            ct_structure_sets = sorted(
                                list(s.ct_structure_sets))
                            rois_plan[disease].extend(
                                ct_structure_sets[0].get_roi_names())
                            if len(ct_structure_sets) > 1:
                                rois_voxtox[disease].extend(
                                    ct_structure_sets[-1].get_roi_names())

        # Ensure that each ROI name is recorded only once;
        # sort ROIs alphabetically, ignoring case.
        if 'mvct' == image_type:
            for category, rois in rois_voxtox[disease].items():
                rois = list(set(rois))
                rois_voxtox[disease][category] = sorted(
                    rois, key=str.casefold)
        else:
            rois_plan[disease] = list(set(rois_plan[disease]))
            rois_plan[disease].sort(key=str.casefold)
            rois_voxtox[disease] = list(set(rois_voxtox[disease]))
            rois_voxtox[disease].sort(key=str.casefold)

    return (n_patient, rois_plan, rois_voxtox)

def print_info(n_patient={}, rois_plan={}, rois_voxtox={}):
    '''
    Print out, for each disease, patient numbers and ROI names.

    **Parameters:**
    n_patient : dict, default={}
        Dictionary where keys are disease identifiers and values
        are patient numbers.

    rois_plan : dict, default={}
        Dictionary where keys are disease identifiers and values
        are ROI names used in clinical plans.

    rois_voxtox : dict, default={}
        Except for 'mvct' image type, dictionary where keys are
        disease identifiers and values are ROI names from
        VoxTox outlining.  For 'mvct' image type, dictionary where
        keys are disease identifiers and values are themselves
        dictionaries.  The latter have keys that are category
        identifiers and values that are ROI names from VoxTox outlining.
    '''

    print('\'\'\'Lists of names labelling ROIs.\'\'\'')

    for disease, n in sorted(n_patient.items()):
        print(f'{disease} = {n}')

    for disease in sorted(rois_voxtox):
        if isinstance(rois_voxtox[disease], list):
            print()
            print(f'{disease}_plan =\\')
            print(sorted(rois_plan[disease]))
            print()
            print(f'{disease}_voxtox =\\')
            print(sorted(rois_voxtox[disease]))
        elif isinstance(rois_voxtox[disease], dict):
            n_category = 0
            for category in sorted(rois_voxtox[disease]):
                print()
                print(f'# {category}:')
                if n_category:
                    print(f'{disease}_mvct +=\\')
                else:
                    print(f'{disease}_mvct =\\')
                print(sorted(rois_voxtox[disease][category]))
                n_category += 1

def run():
    '''Perform processing, taking into account command-line arguments.'''

    # List of valid modes and usage information.
    valid_modes = ['ct', 'djn_253', 'jes_109', 'mvct']
    usage = f'\npython get_roi_names.py <mode> <max_patient> <verbose>'\
            f'\n    <mode> valid values: {valid_modes}'\
            f'\n    <max_patient>: maximum number of datasets to process'\
            f'\n    <verbose>: True to print dataset being processed'\

    if (len(argv) > 1) and (argv[1] in ['-h', '-help', '--help']
                            or argv[1] not in valid_modes):
        # Print usage information.
        print(usage)
    else:
        # Define parameters, taking into account command-line arguments.
        mode = argv[1] if len(argv) > 1 else 'ct'
        max_patient = int(argv[2]) if len(argv) > 2 else 1000
        verbose = (True if len(argv) > 3 and argv[3].lower() == 'true'
                   else False)

        # Extract configuration information.
        top_dir, diseases, cohorts, image_type, ids = get_config(mode)

        # Retrieve patient numbers and ROI names.
        n_patient, rois_plan, rois_voxtox = get_info(
            top_dir, diseases, cohorts, image_type, ids,
            max_patient, verbose)

        # Print patient numbers and ROI names.
        print_info(n_patient, rois_plan, rois_voxtox)

if '__main__' == __name__:
    run()
