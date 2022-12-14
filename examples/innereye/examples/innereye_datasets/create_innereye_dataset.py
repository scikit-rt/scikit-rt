'''
Application to write patient images and structure sets for InnerEye.
'''

import platform
import sys

from innereye import write_patient
from skrt.application import Algorithm, Application, get_paths
from skrt.core import fullpath, make_dir
from voxtox.roi_names.head_and_neck_roi_names import head_and_neck_plan
from voxtox.roi_names.prostate_roi_names import prostate_plan

# Define global variable specifying site of interest.
global_site = "prostate"
global_site = "head_and_neck"

class CreateInnerEyeDataset(Algorithm):
    '''
    Application to write patient images and structure sets for InnerEye.

    Methods:
        __init__ -- Return instance of CreateNiftiDataset class,
                    with properties set according to options dictionary.
        execute  -- Write patient images and structure sets for InnerEye.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of CreateInnerEyeDataset class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        # If True, recreate output directory.
        self.recreate_outdir = False

        # Top-level output directory.  Within this, there will be a
        # patient sub-directory, containing a sub-directory for each
        # study, containing in turn a sub-directory for each output
        # image along with associated structure set(s).
        self.outdir = './ie_dataset'

        # List of indices of studies for which data are to be written,
        # 0 being the earliest study and -1 being the most recent.  If set to
        # None, data for all studies are written.
        self.studies_to_write = None

        # If True, report progress.
        self.verbose = True

        # If True, delete delete any pre-existing patient folder.
        self.overwrite = False

        # Images types to be saved: None to save all, otherwise a list
        # of image types to save, or a string specifying a single image
        # type to save.
        self.image_types = None

        # Dictionary where the keys are image types and the values are
        # lists of timestamp indices for the images to be saved,
        # 0 being the earliest and -1 being the most recent.  If set to
        # None, all images are saved.
        self.images_to_write = None

        # Image dimensions (dx, dy, dz) in voxels for image resizing
        # prior to writing.  If None, original image size is kept.
        self.image_size = None

        # Voxel dimensions (dx, dy, dz) in mm for image resizing
        # prior to writing.  If None, original voxel size is kept.
        self.voxel_size = None

        # Value used when extrapolating image outside data area.
        # If None, use mininum value inside data area.
        self.fill_value = None

        # Nested dictionary of value bandings to be applied before
        # image saving.  The primary key defines the type of image to
        # which the banding applies.  Secondary keys specify band limits,
        # and associated values indicte the values to be assigned.
        # For example:
        #
        # - bands{'mvct' : {-100: -1024, 100: 0, 1e10: 1024}}
        #
        # will band an image of type 'mvct':
        #
        # - value <= -100 => -1024;
        # - -100 < value <= 100 => 0;
        # - 100 < value <= 1e10 => 1024.
        self.bands = {}

        # List of image types for which data are to be written only
        # if there is an associated structure set.
        self.require_structure_set = []

        # Dictionary where the keys are image types and the values are
        # lists of file indices for structure sets to be saved for
        # a given image, 0 being the earliest and -1 being the most recent.
        # If set to None, all of an image's structure sets are saved.
        self.structure_sets_to_write = None

        # Dictionary of names for renaming ROIs, where the keys are new 
        # names and values are lists of possible names of ROIs that should
        # be assigned the new name. These names can also contain wildcards
        # with the '*' symbol
        self.roi_names = None

        # If True, force writing of dummy NIfTI file for all named ROIs
        # not found in structure set.  If False, named ROIs not found in
        # structure set are disregarded.
        self.force_roi_nifti = False

        # bilateral_names : list, default=None
        # List of names of ROIs that should be split into
        # left and right parts.
        self.bilateral_names = []

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Obtain full path to output directory.
        self.outdir = fullpath(self.outdir)

        # Create, or recreate, the output directory as needed.
        make_dir(self.outdir, overwrite=self.recreate_outdir)

    def execute(self, patient=None):

        '''
        Write patient images and structure sets for InnerEye.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        '''

        # Print details of current patient.
        print(f'Patient id: {patient.id}')
        print(f'Folder path: {patient.path}')
        write_patient(
                patient,
                outdir=self.outdir,
                studies_to_write=self.studies_to_write,
                verbose=self.verbose,
                overwrite=self.overwrite,
                image_types=self.image_types,
                images_to_write=self.images_to_write,
                image_size=self.image_size,
                voxel_size=self.voxel_size,
                fill_value=self.fill_value,
                bands=self.bands,
                require_structure_set=self.require_structure_set,
                structure_sets_to_write=self.structure_sets_to_write,
                roi_names=self.roi_names,
                force_roi_nifti=self.force_roi_nifti,
                bilateral_names=self.bilateral_names
                )

        return self.status


def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    opts['image_types'] = ['mvct']
    opts['voxel_size'] = (1.5, 1.5, None)
    opts['recreate_outdir'] = False
    opts['verbose'] = False
    if "head_and_neck" == global_site:
        opts['roi_names'] = head_and_neck_plan
    elif "prostate" == global_site:
        opts['roi_names'] = prostate_plan
    if "Linux" == platform.system():
        opts['outdir'] = f'/r02/voxtox/workshop/ie_datasets/{global_site}'
    else:
        opts['outdir'] = fullpath(f'~/data/ie_datasets/{global_site}')

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = CreateInnerEyeDataset(opts=opts, name=None, log_level=log_level)

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
        data_dirs = [f"/r02/voxtox/workshop/synthetic_mvct/{global_site}"]
    else:
        data_dirs = [fullpath(f"~/data/voxtox_check")]

    patterns = ["VT*"]

    return {data_dir: patterns for data_dir in data_dirs}

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())

    # Run application for the selected data
    app.run(paths)

if 'Ganga' in __name__:
    # Define script for setting analysis environment
    setup_script = fullpath('skrt_conda.sh')

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), setup_script)

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())
    input_data = PatientDataset(paths=paths)

    # Define processing system.
    if "Linux" == platform.system():
        backend = Condor()
        backend.cdf_options["request_memory"]="8G"
    else:
        backend = Local()

    # Define how job should be split into subjobs
    splitter = PatientDatasetSplitter(patients_per_subjob=20)

    # Define merging of subjob outputs
    merger = SmartMerger()
    merger.files = ['stderr', 'stdout']
    merger.ignorefailed = True
    postprocessors = [merger]

    # Define job name
    name = f'{global_site}_ied'

    # Define list of outputs to be saved
    outbox = []

    # Create the job, and submit to processing system
    j = Job(application=ganga_app, backend=backend, inputdata=input_data,
            outputsandbox=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
