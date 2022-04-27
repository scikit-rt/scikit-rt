'''
Application for extracting summary data for VoxTox datasets.
'''

from itertools import chain
from pathlib import Path

import pandas as pd

from skrt.application import Algorithm, Application

class ExtractInfo(Algorithm):
    '''
    Algorithm subclass, for extracting summary data.

    Methods:
        __init__ -- Return instance of ExtractInfo class,
                    with properties set according to options dictionary.
        execute  -- Extract summary data.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of ExtractInfo class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''

        # List of strings indicating types of image for which information
        # is to be retrieved, for example ['ct', 'mr'].  If None,
        # information is retrieved for all image types in the patient dataset.
        self.image_types = None

        # Name of output csv files.
        self.summary_csv = 'summary_data.csv'
        self.images_csv = 'image_data.csv'
        self.plans_csv = 'plan_data.csv'
        self.doses_csv = 'dose_data.csv'

        # Define patient collections.
        self.collections = {}
        self.collections['cohort'] = ['consolidation', 'discovery']
        self.collections['sample'] = ['error_cases', 'special_cases']

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        self.summary_data = []
        self.image_data = []
        self.plan_data = []
        self.dose_data = []

    def execute(self, patient=None):

        '''
        Extract summary data for patient's dataset.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        '''

        # Print details of current patient.
        print(f'Patient id: {patient.id}')
        print(f'Folder path: {patient.path}')

        summary_info = patient.get_info(collections=self.collections,
                plan_image_type=self.plan_image_type,
                treatment_image_type=self.treatment_image_type)
        image_info = patient.get_image_info(image_types=self.image_types)
        plan_info = patient.get_plan_info()
        dose_info = patient.get_dose_info()
        df1 = patient.get_info(collections=self.collections,
                plan_image_type=self.plan_image_type,
                treatment_image_type=self.treatment_image_type, df=True)
        df2 = patient.get_image_info(image_types=self.image_types, df=True)
        df3 = patient.get_plan_info(df=True)
        df4 = patient.get_dose_info(df=True)
        #print(df1.to_string(index=False))
        #print(df2.to_string(index=False))
        #print(df3.to_string(index=False))
        #print(df4.to_string(index=False))

        self.summary_data.append(summary_info)
        self.image_data.extend(image_info)
        self.plan_data.extend(plan_info)
        self.dose_data.extend(dose_info)

        return self.status

    def finalise(self):

        df1 = pd.DataFrame(self.summary_data)
        df1.to_csv(self.summary_csv, index=False)

        df2 = pd.DataFrame(self.image_data)
        df2.to_csv(self.images_csv, index=False)

        df3 = pd.DataFrame(self.plan_data)
        df3.to_csv(self.plans_csv, index=False)

        df4 = pd.DataFrame(self.dose_data)
        df4.to_csv(self.doses_csv, index=False)

        return self.status


def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}

    #opts['image_types'] = ['ct', 'mvct']
    opts['plan_image_type'] = 'ct'
    opts['treatment_image_type'] = 'mvct'
    opts['collections'] = {}
    #opts['collections']['cohort'] = ['import_high', 'import_low']
    opts['collections']['cohort'] = ['consolidation', 'discovery']
    opts['collections']['sample'] = ['error_cases', 'special_cases']

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = ExtractInfo(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

def get_paths():
    # Define the patient data to be analysed
    data_dir = Path('/Users/karl/data/head_and_neck/vspecial/30_patients__spinal_cord__1_mv')
    paths = data_dir.glob('VT*')
    #data_dir = Path('/Users/karl/data/20220331_import_data_selection')
    #paths = chain(data_dir.glob('import_high/H*'),
    #              data_dir.glob('import_low/L*'))

    return list(sorted(paths))

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the patient data to be analysed
    paths = get_paths()

    # Run application for the selected data
    app.run(paths[0:5])

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
    splitter = PatientDatasetSplitter(patients_per_subjob=50)

    # Define merging of subjob outputs
    merger = SmartMerger()
    merger.files = ['stderr', 'stdout', 'summary_data.csv', 'image_data.csv',
            'plan_data.csv', 'dose_data.csv']
    merger.ignorefailed = True
    postprocessors = [merger]

    # Define job name
    name = 'contour_analysis'

    # Define list of outputs to be saved
    outbox = ['summary_data.csv', 'image_data.csv', 'plan_data.csv',
            'dose_data.csv']

    # Create the job, and submit to processing system
    j = Job(application=ganga_app, backend=backend, inputdata=input_data,
            outputfiles=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
