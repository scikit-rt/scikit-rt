'''
Application to write patient images and structure sets for nnU-net.

For information about nnU-net (non-new U-net), see:
    https://doi.org/10.1038/s41592-020-01008-z
    https://github.com/MIC-DKFZ/nnUNet/
'''

from json import dump
from pathlib import Path
from platform import system
from random import random
from sys import argv

from skrt.application import Algorithm, Application, get_paths
from skrt.core import filter_on_paths, fullpath, make_dir

# Import ROI dictionaries for head-and-neck cohort.
from voxtox.roi_names.head_and_neck_roi_names import (
        head_and_neck_plan, head_and_neck_voxtox, head_and_neck_iov,
        head_and_neck_mvct, head_and_neck_parotid_fiducials, head_and_neck_tre)

# Import ROI dictionaries for prostate cohort.
from voxtox.roi_names.prostate_roi_names import (
        prostate_plan, prostate_voxtox, prostate_iov, prostate_mvct)

SITE = "head_and_neck"

class CreateNnunetDataset(Algorithm):
    """
    Application to write patient images and structure sets for nnU-net.

    Methods:
        __init__ -- Return instance of CreateNnunetDataset class,
                    with properties set according to options dictionary.
        execute  -- Write patient images and structure sets for nnU-net.
        finalise -- Write metadata for training dataset.
    """

    def __init__(self, opts=None, name=None, log_level=None):
        """
        Create instance of CreateNnunetDataset class.

        opts: dict, default=None
            Dictionary for setting algorithm attributes.  If null,
            set to empty dictionary.

        name: str, default=None
            Name for identifying algorithm instance.  If null,
            set to empty dictionary.

        log_level: str/int/None, default=None
            Severity level for event logging.  If None, log_level is
            set to the value of skrt.core.Defaults().log_level.
        """
        # If True, create a dataset for training and evaluation
        # (images and segmentations); if False, create a dataset
        # for inference (images only).
        self.training_set = True

        # Fraction of input cases that will be randomly assigned to
        # test set when creating a dataset for training and evaluation.
        self.test_fraction = 0.2

        # Type of images to be written.  Multiple image types per case
        # can be handled by nnU-net, but only a single image type
        # is considered here.
        self.image_type = "ct"

        # Image dimensions (dx, dy, dz) in voxels for image resizing
        # prior to writing.  If None, original image size is kept.
        self.image_size = None

        # Voxel dimensions (dx, dy, dz) in mm for image resizing
        # prior to writing.  If None, original voxel size is kept.
        self.voxel_size = None

        # Value used when extrapolating image outside data area.
        # If None, use mininum value inside data area.
        self.fill_value = None

        # Dictionary of value bandings to be applied to image data.
        # Keys specify band limits, and values indicte the values
        # to be assigned.  For example:
        #     bands{-100: -1024, 100: 0, 1e10: 1024}
        # will result in the following banding:
        #    value <= -100 : -1024;
        #    -100 < value <= 100 : 0;
        #    100 < value <= 1e10 : 1024.
        self.bands = None

        # Dictionary of names for ROIs to be written in output structure set.
        # Keys are names to be assigned to ROIs, and values are list of names
        # with which ROIs may be labelled in the input data.  These names can
        # also contain wildcards with the '*' symbol.  The output structure
        # set is a multi-label NIfTI file, where ROIs are numbered
        # consecutively, starting from 1, in the order in which they are
        # listed in the dictionary.  Before numbering, bilateral ROIs will
        # be split into left and right components.  If None, no structure
        # set is written in output.  Ignored if self.training_set is False.
        self.roi_names = None

        # If True, skip cases where all of the ROIs identified by
        # self.roi_names are present in the image-associated structure set.
        # Ignored if self.training_set is False.
        self.require_all_rois = True

        # List of names of ROIs that should be split into
        # left and right parts, following initial name resolution.
        # Ignored if self.training_set is False.
        self.bilateral_names = []

        # Top-level directory for nnU-net, within which there will be
        # sub-directories for raw data, preprocessing, and results.
        self.topdir = './data'

        # Sub-directory for raw data.  Within this, there will be a dataset
        # sub-directory.  When creating a dataset for training, this will
        # contain sub-directories for training images (imagesTr) and training
        # segmentations (labelsTr).  If self.test_fraction is greater than
        # zero, the dataset sub-directory will also contain sub-directories
        # for test images (imagesTs) and test segmentations (labelsTs).
        # When creating a dataset for inference, images will be placed
        # directly in the dataset sub-directory.
        self.raw_subdir = 'nnUNet_raw'

        # Identifier to be assigned to dataset.
        # This should be an integer of up to three digits.
        # All datasets within the top-level data directory must have
        # a different identifier.
        self.dataset_id = 1

        # Name to be assigned to dataset.
        self.dataset_name = "Test"

        # If True, delete delete any pre-existing dataset folder
        # (but not top-level output directory)
        self.overwrite = False

        # List of indices of studies for which data are to be written,
        # 0 being the earliest study and -1 being the most recent.  If set to
        # None, data for all studies are written.
        self.studies_to_write = None

        # index of image-associated structure set for which data
        # are to be written, 0 being the earliest structure set and -1 being
        # the most recent.  Ignored if self.training_set is False.
        self.structure_set_to_write = 0

        # Suffix for names of output files.
        self.suffix = ".nii.gz"

        # Dictionary where keys are patient identifiers,
        # and values are lists of paths to images for processing.
        # This dictionary will be updated if Ganga is used to
        # run the application, with splitting via PatientImageSplitter.
        self.images = {}

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Define required associated data for images to be written.
        self.associations = "structure_sets" if self.training_set else None

        # Ensure that image type is upper case.
        self.image_type = self.image_type.upper()

        # Initialise counter for number of images written.
        self.n_image = 0

        # Initialise counter for number of images written for training.
        self.n_train = 0

        # Initialise counter for number of images written for testing.
        self.n_test = 0

        # Define list of ROI names for output structure set.
        # This is the same as the input list of names, except that
        # bilateral ROIs are split into left and right components.
        self.out_roi_names = list(self.roi_names)
        if self.bilateral_names:
            for name in self.bilateral_names:
                if name in self.out_roi_names:
                    idx = self.out_roi_names.index(name)
                    self.out_roi_names[idx : idx + 1] = [
                            f"{name}_left", f"{name}_right"]
        self.out_roi_names = tuple(self.out_roi_names)

        # Ensure top-level directory exists.
        self.topdir = Path(self.topdir).resolve()
        self.topdir.mkdir(parents=True, exist_ok=True)

        # Ensure raw-data directory exists.
        self.raw_dir = make_dir(self.topdir / self.raw_subdir,
                                overwrite=False)

        # Ensure dataset directory exists.
        self.dataset_dir = make_dir(self.raw_dir / 
                                    f"Dataset{self.dataset_id:03}"
                                    f"_{self.dataset_name}",
                                    overwrite=self.overwrite)

        # Ensure that self.test_fraction is in the closed interval [0, 1].
        self.test_fraction = min(max(0, self.test_fraction), 1)

        # Ensure that sub-directories for training dataset exist.
        if self.training_set:
            self.images_tr_dir = make_dir(
                    self.dataset_dir / "imagesTr", overwrite=self.overwrite)
            self.labels_tr_dir = make_dir(
                    self.dataset_dir / "labelsTr", overwrite=self.overwrite)
            if self.test_fraction:
                self.images_ts_dir = make_dir(
                        self.dataset_dir / "imagesTs", overwrite=self.overwrite)
                self.labels_ts_dir = make_dir(
                        self.dataset_dir / "labelsTs", overwrite=self.overwrite)

        # Initialse output directories for dataset files.
        if self.training_set:
            self.images_dir = self.images_tr_dir
            self.labels_dir = self.labels_tr_dir
        else:
            self.images_dir = self.dataset_dir
            self.labels_dir = None

    def execute(self, patient=None):
        """
        Write patient images and structure sets for nnU-net.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        """

        # Print details of current patient.
        print(f'Patient id: {patient.id}')
        print(f'Folder path: {patient.path}')

        # Retrieve studies for processing.
        studies = patient.get_studies()
        if self.studies_to_write:
            studies = [studies[idx] for idx in self.studies_to_write]

        for study in studies:
            # Retrieve images for processing.
            images = filter_on_paths(
                    study.get_images(self.image_type, self.associations),
                    self.images.get(patient.id, None))

            for image in images:
                # Perform any image standardisation requested.
                image.resize(image_size=self.image_size,
                             voxel_size=self.voxel_size)
                image.apply_banding(self.bands)


                # For training set, process image-associated structure set.
                if self.training_set:
                    sset = image.get_structure_sets()[
                            self.structure_set_to_write]

                    # Standardise ROI names.
                    sset.rename_rois(self.roi_names, keep_renamed_only=True)

                    # Check that structure set is non-empty,
                    # and contains any required ROIs.
                    if sset is None or (self.require_all_rois and
                        len(sset.get_roi_names()) != len(self.roi_names)):
                        continue

                    # Split bilateral ROIs into left and right components.
                    sset.split_rois_in_two(self.bilateral_names)

                    # Ensure that ROI masks have same geometry as image.
                    if not sset.get_image().has_same_geometry(image):
                        sset.set_image(image)

                # Randomly divide images of training set
                # between images for training and images for evaluation.
                if self.training_set and self.test_fraction:
                    if random() < self.test_fraction:
                        self.images_dir = self.images_ts_dir
                        self.labels_dir = self.labels_ts_dir
                        self.n_test += 1
                    else:
                        self.images_dir = self.images_tr_dir
                        self.labels_dir = self.labels_tr_dir
                        self.n_train += 1
                self.n_image += 1

                # Define case-specific part of output file names.
                case_identifier = (f"{patient.id}_{study.timestamp}_"
                                   f"{self.image_type}_{image.timestamp}")

                # Write image.
                image.write(self.images_dir /
                                f"{case_identifier}_0000{self.suffix}")

                # Write structure set for training and evaluation.
                if self.training_set:
                    sset.set_image(image)
                    for roi in sset.get_rois():
                        roi.create_mask()
                    sset.write(f"{case_identifier}{self.suffix}",
                               self.labels_dir, multi_label=True,
                               names=self.out_roi_names)

        return self.status

    def finalise(self):
        """
        Write metadata for training set.
        """
        if self.training_set:
            out_roi_names = ["background"] + list(self.out_roi_names)
            metadata = {
                    "channel_names": {"0": self.image_type.upper()},
                    "labels": {roi_name: idx for idx, roi_name in
                               enumerate(out_roi_names)},
                    "numTraining": self.n_train, 
                    "file_ending": self.suffix,
             }
            with open(self.dataset_dir / "dataset.json", "w") as out_json:
                dump(metadata, out_json, indent=4, separators=None)

        return self.status

def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''
    opts = {}

    if "head_and_neck" == SITE:
        if "Linux" == system():
            roi_names = [
                    "brainstem",
                    "mandible",
                    "parotid_left",
                    "parotid_right",
                    "smg_left",
                    "smg_right",
                    "spinal_cord",
                    ]
            roi_lookup = head_and_neck_plan
        else:
            roi_names = ["spinal_cord"]
            roi_lookup = head_and_neck_mvct
    elif "prostate" == SITE:
        roi_names = ["rectum"]
        roi_lookup = prostate_plan

    opts["roi_names"] = {roi_name: roi_lookup[roi_name]
                         for roi_name in roi_names}

    opts["training_set"] = True
    opts["image_type"] = "MVCT"

    opts["voxel_size"] = (1.5, 1.5, 1.5)
    opts["image_size"] = (256, 256, None)

    opts["topdir"] = str(Path("~/nnUNet/data").expanduser())

    opts["dataset_id"] = 1
    opts["dataset_name"] = "Test"

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = CreateNnunetDataset(opts=opts, name=None, log_level=log_level)

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
    if "Linux" == system():
        data_dirs = [f"/r02/voxtox/workshop/synthetic_mvct/{SITE}"]
        #data_dirs = [f"/r02/voxtox/data/head_and_neck/vspecial/30_patients__spinal_cord__1_mv"]
        patterns = ["VT*"]
    else:
        data_dirs = [fullpath(f"~/data/voxtox_check/{SITE}")]
        patterns = ["*/VT*"]
        data_dirs = [fullpath(f"~/data/head_and_neck/vspecial/30_patients__spinal_cord__1_mv")]
        patterns = ["VT*"]

    return {data_dir: patterns for data_dir in data_dirs}

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())

    # Run application for the selected data
    app.run(paths[0:5])

if 'Ganga' in __name__:
    # Define script for setting analysis environment
    setup_script = fullpath('skrt_conda.sh')

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), setup_script)

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())
    input_data = PatientDataset(paths=paths)

    # Define processing system.
    if "Linux" == system():
        backend = Condor()
        backend.cdf_options["request_memory"]="8G"
    else:
        backend = Local()

    # Define how job should be split into subjobs
    splitter = PatientDatasetSplitter(patients_per_subjob=5)
    splitter = None

    # Define merging of subjob outputs
    merger = SmartMerger()
    merger.files = ['stderr', 'stdout']
    merger.ignorefailed = True
    postprocessors = [merger]
    postprocessors = []

    # Define job name
    name = f'nnunet_{SITE}'

    # Define list of outputs to be saved
    outbox = []

    # Create the job, and submit to processing system
    j = Job(application=ganga_app, backend=backend, inputdata=input_data,
            outputsandbox=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
