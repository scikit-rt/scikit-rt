'''
Application for copying DICOM files, with optional sorting.
'''
import platform
import sys
import timeit

from pathlib import Path

from skrt.application import Algorithm, Application, get_paths
from skrt.core import fullpath, get_qualified_class_name, is_list
from skrt.patient import Patient

class CopyDicom(Algorithm):
    '''
    Algorithm subclass, for copying DICOM files.

    Methods:
        __init__ -- Return instance of CopyDicom class,
                    with properties set according to options dictionary.
        execute  -- Sort DICOM files.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of CopyDicom class.

        **Parameters:**
        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        # For more details of specifying items to copy, see documentation of:
        # skrt.patient.Patient.copy_dicom()
        # skrt.patient.Study.copy_dicom()
        # The default values are set to copy all studies, images,
        # strucure sets, doses, plans, sorting according to VoxTox scheme.

        # Specification of studies to copy.
        self.studies_to_copy = None

        # Specification of images to copy.
        self.images_to_copy = None

        # Specification of structure sets to copy.
        self.structure_sets_to_copy = None

        # Specification of doses to copy.
        self.doses_to_copy = None

        # Specification of plans to copy.
        self.plans_to_copy = None

        # Sort output files according to VoxTox scheme.
        self.sort = True

        # Path to output directory.
        self.outdir = "dicom"

        # Indicate whether to overwrite any pre-existing patient directory.
        self.overwrite = True

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Expand path to output directory.
        self.outdir = Path(fullpath(self.outdir))

    def execute(self, patient=None):
        '''
        Map ROIs from relapse scan to planning scan.

        **Parameter:**

        patient: skrt.patient.Patient, default=None
            Object providing access to patient dataset.
        '''

        # Print details of current patient.
        self.logger.info(f"Patient id: {patient.id}")
        self.logger.info(f"Input folder: {patient.path}")
        self.logger.info(f"Initialisation time: {patient._init_time:.2f} s")
        self.logger.info(f"Output folder {self.outdir}/{patient.id}")

        tic = timeit.default_timer()
        patient.copy_dicom(
                outdir=self.outdir,
                studies_to_copy=self.studies_to_copy,
                overwrite=self.overwrite,
                images_to_copy=self.images_to_copy,
                structure_sets_to_copy=self.structure_sets_to_copy,
                doses_to_copy=self.doses_to_copy,
                plans_to_copy=self.plans_to_copy,
                sort=self.sort)
        toc = timeit.default_timer()
        self.logger.info(f"Copying time: {toc - tic:.2f} s\n")

        return self.status


def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    # Specification of studies to copy.
    opts["studies_to_copy"] = None

    # Specification of images to copy.
    opts["images_to_copy"] = None

    # Specification of structure sets to copy.
    opts["structure_sets_to_copy"] = None

    # Specification of doses to copy.
    opts["doses_to_copy"] = None

    # Specification of plans to copy.
    opts["plans_to_copy"] = None

    # Sort output files according to VoxTox scheme.
    opts["self.sort"] = True

    # Path to output directory.
    opts["outdir"] = fullpath("dicom")

    # Indicate whether to overwrite any pre-existing patient directory.
    opts["overwrite"] = True

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = CopyDicom(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

def get_data_locations():
    """
    Specify locations of patient datasets.
    """
    # Define the patient data to be analysed
    if "Linux" == platform.system():
        data_dirs = Path("/r02/radnet/import").glob("Arch*")
    else:
        data_dirs = [Path("~/data/20220613_import_cam").expanduser()]

    patterns = ["import_high/R*", "import_high/H*", "import_low/L*"]

    return {data_dir: patterns for data_dir in data_dirs}

def get_data_loader():
    # Define class and options for loading patient data.
    PatientClass = Patient
    patient_class = get_qualified_class_name(PatientClass)
    patient_opts = {"unsorted_dicom": True}

    return (PatientClass, patient_class, patient_opts)


if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define class and options for loading patient datasets.
    PatientClass, patient_class, patient_opts = get_data_loader()

    # Define the patient data to be analysed.
    paths = get_paths(get_data_locations())

    # Run application for the selected data.
    app.run(paths, PatientClass, **patient_opts)


if 'Ganga' in __name__:
    # Define script for setting analysis environment.
    setup_script = fullpath('skrt_conda.sh')

    # Define class and options for loading patient datasets.
    PatientClass, patient_class, patient_opts = get_data_loader()

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), setup_script,
            patient_class, patient_opts)
    opts = ganga_app.algs[0].opts

    # Define the patient data to be analysed.
    if "Linux" == platform.system():
        paths = get_paths(get_data_locations())
    else:
        paths = get_paths(get_data_locations(), 1)
    input_data = PatientDataset(paths=paths)

    # Define processing system.
    if "Linux" == platform.system():
        backend = Condor()
        backend.cdf_options["request_memory"]="4G"
    else:
        backend = Local()

    # Define how job should be split into subjobs.
    splitter = PatientDatasetSplitter(patients_per_subjob=1)

    # Set up output merging.
    merger = SmartMerger()
    merger.files = ['stderr', 'stdout']
    merger.ignorefailed = True
    postprocessors = [merger]

    # Define list of outputs to be saved.
    outbox = []

    # Define job name.
    name = "copy_dicom"

    # Create the job, and submit to processing system.
    j = Job(application=ganga_app, backend=backend,
            inputdata=input_data,
            outputfiles=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
