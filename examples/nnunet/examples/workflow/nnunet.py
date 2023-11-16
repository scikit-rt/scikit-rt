"""
Application to write patient images and structure sets for nnU-net.

For information about nnU-net (non-new U-net), see:
    https://doi.org/10.1038/s41592-020-01008-z
    https://github.com/MIC-DKFZ/nnUNet/
"""

from json import dump
from pathlib import Path
from platform import system
from random import random
from sys import argv

from skrt.application import Algorithm, Application, get_paths
from skrt.core import filter_on_paths, fullpath, get_indexed_objs, make_dir

# Import ROI dictionaries for head-and-neck cohort.
from voxtox.roi_names.head_and_neck_roi_names import (
    head_and_neck_plan,
    head_and_neck_voxtox,
    head_and_neck_iov,
    head_and_neck_mvct,
    head_and_neck_parotid_fiducials,
    head_and_neck_tre,
)

# Import ROI dictionaries for prostate cohort.
from voxtox.roi_names.prostate_roi_names import (
    prostate_plan,
    prostate_voxtox,
    prostate_iov,
    prostate_mvct,
)

SITE = "prostate"


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
        # Directory to which to write output.
        self.outdir = "./data"

        # Dataset identifier.  If an integer, the dataset to be written
        # is taken to be a training set: images and associated labels
        # are written to the sub-directories "imagesTr" and "labelsTr"
        # <outdir>/nnUNet_raw/Dataset<dataset_id>_<dataset_name>.
        # If None, the dataset is taken to be for inference: images
        # are written directly to <outdir>.
        self.dataset_id = 1

        # Name to be associated with a training set, and used in the
        # construction of the path to the training set images and labels.
        # Ignored if self.dataset_id is None (not a training set).
        self.dataset_name = "Test"

        # Type of images to be written.  Multiple image types per case
        # can be handled by nnU-net, but only a single image type
        # is considered here.
        self.image_type = "ct"

        # Dictionary where keys are patient identifiers,
        # and values are lists of paths to images for processing.
        # This dictionary will be updated if Ganga is used to
        # run the application, with splitting via PatientImageSplitter.
        self.images = {}

        # Image dimensions (dx, dy, dz) in voxels for image resizing
        # prior to writing.  If None, original image size is kept.
        # If an individual dimension is None, and the corresponding
        # voxel dimension for resizing isn't None, the image dimension
        # in voxels will be set so as to retain the original dimension
        # in mm.
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

        # index of image-associated structure set for which data
        # are to be written, 0 being the earliest structure set and -1 being
        # the most recent.  Ignored if self.dataset_id is None
        # (not a training set)
        self.structure_set_to_write = -1

        # Dictionary of names for ROIs to be written in output structure set.
        # Keys are names to be assigned to ROIs, and values are list of names
        # with which ROIs may be labelled in the input data.  These names can
        # also contain wildcards with the '*' symbol.  The output structure
        # set is a multi-label NIfTI file, where ROIs are numbered
        # consecutively, starting from 1, in the order in which they are
        # listed in the dictionary.  Before numbering, bilateral ROIs will
        # be split into left and right components.  If None, no structure
        # set is written in output.  Ignored if self.dataset_id is None
        # (not a training set).
        self.roi_names = None

        # If True, skip cases where all of the ROIs identified by
        # self.roi_names are present in the image-associated structure set.
        # Ignored if self.dataset_id is None (not a training set).
        self.require_all_rois = True

        # List of names of ROIs that should be split into
        # left and right parts, following initial name resolution.
        # Ignored if self.dataset_id is None (not a training set).
        self.bilateral_names = []

        # List of indices of studies for which data are to be written,
        # 0 being the earliest study and -1 being the most recent.  If set to
        # None, data for all studies are written.
        self.studies_to_write = None

        # Suffix for names of output files.
        self.suffix = ".nii.gz"

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Initialise counter for number of images written.
        self.n_image = 0

        # Ensure output directory exists.
        self.outdir = fullpath(self.outdir, pathlib=True)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def execute(self, patient=None):
        """
        Write patient images and structure sets for nnU-net.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        """

        # Print details of current patient.
        print(f"Patient id: {patient.id}")
        print(f"Folder path: {patient.path}")

        n_image += write_patient(
            patient,
            studies_to_write=self.studies_to_write,
            outdir=self.outdir,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            image_type=self.image_type,
            images=self.images,
            bands=self.bands,
            structure_set_to_write=self.structure_set_to_write,
            roi_names=self.roi_names,
            require_all_rois=self.require_all_rois,
            bilateral_names=self.bilateral_names,
            suffix=self.suffix,
        )

        return self.status

    def finalise(self):
        """
        Write metadata for training set.
        """
        if self.training_set:
            out_roi_names = ["background"] + list(self.out_roi_names)
            metadata = {
                "channel_names": {"0": self.image_type.upper()},
                "labels": {
                    roi_name: idx for idx, roi_name in enumerate(out_roi_names)
                },
                "numTraining": self.n_train,
                "file_ending": self.suffix,
            }
            with open(self.dataset_dir / "dataset.json", "w") as out_json:
                dump(metadata, out_json, indent=4, separators=None)

        return self.status


def get_dataset_path(indir, dataset_id, dataset_name=None):
    if dataset_name is None:
        return list(
            fullpath(indir, pathlib=True).glob(f"Dataset{dataset_id:03}_*")
        )[0]
    return (
        fullpath(indir, pathlib=True) / f"Dataset{dataset_id:03}_{dataset_name}"
    )


def get_nnunet_env(topdir="./nnunet"):
    """
    Define environment variables for running nnU-net.
    """
    topdir = fullpath(topdir, pathlib=True)
    return {
        f"nnUNet_{var}": str(topdir / f"nnUNet_{var}")
        for var in ["raw", "preprocessed", "results"]
    }


def get_training_dirs(topdir, dataset_id, dataset_name=None):
    dataset_dir = get_dataset_path(
        get_nnunet_env(topdir)["nnUNet_raw"], dataset_id, dataset_name
    )
    return (dataset_dir / "imagesTr", dataset_dir / "labelsTr")


def write_patient(patient, studies_to_write=True, **kwargs):
    """
    Write patient data for use by nnU-net.

    For a training set, images and associated structure sets are written.
    Otherwise, only images are written.

    **Parameters:**

    patient - skrt.patient.Patient
        Patient object for which data are to be written.

    studies_to_write : list/bool default=True
        List of indices of studies for which data are to be written,
        0 being the earliest study and -1 being the most recent.  If set to
        True, data for all studies are written.

    **kwargs
        Keyword arguments passed to
            nnunet.write_study().
        For details, see nnunet.write_study() documentation.
    """
    # Initialise counter for number of images written.
    n_image = 0

    # Process selected studies.
    for study in get_indexed_objs(patient.get_studies(), studies_to_write):
        n_image += write_study(study, **kwargs)

    return n_image


def write_study(
    study,
    outdir="./nnunet_datasets",
    dataset_id=None,
    dataset_name="Test",
    image_type="ct",
    images=None,
    image_size=None,
    voxel_size=None,
    fill_value=None,
    bands=None,
    structure_set_to_write=-1,
    roi_names=None,
    require_all_rois=False,
    bilateral_names=None,
    suffix=".nii.gz",
):
    """
    Write study data for use by nnU-net.

    For a training set, images and associated structure sets are written.
    Otherwise, only images are written.

    **Parameters:**

    study : skrt.patient.Study
        Study object for which data are to be written.

    outdir : str, default='.'
        Directory to which to write output.

    dataset_id : int/None, default=None
        Dataset identifier.  If an integer, the dataset to be written
        is taken to be a training set: images and associated labels
        are written to the sub-directories "imagesTr" and "labelsTr"
        <outdir>/nnUNet_raw/Dataset<dataset_id>_<dataset_name>.
        If None, the dataset is taken to be for inference: images
        are written directly to <outdir>.

    dataset_name: str, default="Test"
        Name to be associated with a training set, and used in the
        construction of the path to the training set images and labels.
        Ignored if <dataset_id> is None (not a training set).

    image_type : str, default="ct"
        Type of images to be written.  Multiple image types per case
        can be handled by nnU-net, but only a single image type
        is considered here.

    images : dict/None, default=None
        Dictionary where keys are patient identifiers,
        and values are lists of paths to images for processing.
        Allows for selective processing of a study's images.  If None,
        all images are processed.

    image_size : tuple, default=None
        Image dimensions (dx, dy, dz) in voxels for image resizing
        prior to writing.  If None, original image size is kept.
        If an individual dimension is None, and the corresponding
        voxel dimension for resizing isn't None, the image dimension
        in voxels will be set so as to retain the original dimension
        in mm.

    voxel_size : tuple, default=None
        Voxel dimensions (dx, dy, dz) in mm for image resizing
        prior to writing.  If None, original voxel size is kept.

    fill_value : int/float, default=None
        Value used when extrapolating image outside data area.
        If None, use mininum value inside data area.

    bands : dict, default=None
        Nested dictionary of value bandings to be applied before
        image saving.  The primary key defines the type of image to
        which the banding applies.  Secondary keys specify band limits,
        and associated values indicte the values to be assigned.
        For example:

        - bands{'mvct' : {-100: -1024, 100: 0, 1e10: 1024}}

        will band an image of type 'mvct':

        - value <= -100 => -1024;
        - -100 < value <= 100 => 0;
        - 100 < value <= 1e10 => 1024.

    structure_set_to_write: int, default=-1
        index of image-associated structure set for which data
        are to be written, 0 being the earliest structure set and -1 being
        the most recent.  Ignored if <dataset_id> is None (not a training set).

    roi_names : dict, default=None
        Dictionary of names for ROIs to be written in output structure set.
        Keys are names to be assigned to ROIs, and values are list of names
        with which ROIs may be labelled in the input data.  These names can
        also contain wildcards with the '*' symbol.  The output structure
        set is a multi-label NIfTI file, where ROIs are numbered
        consecutively, starting from 1, in the order in which they are
        listed in the dictionary.  Before numbering, bilateral ROIs will
        be split into left and right components.  If None, no structure
        set is written in output.  Ignored if <dataset_id> is None
        (not a training set).

    require_all_rois : bool, default=False
        If True, skip cases where not all of the ROIs identified by
        roi_names are present in the image-associated structure set.
        Ignored if <dataset_id> is None (not a training set).

    bilateral_names : list, default=None
        List of names of ROIs that should be split into
        left and right parts, following initial name resolution.
        Ignored if <dataset_id> is False (not a training set).

    suffix : str, default=".nii.gz"
        Suffix for names of output files.
    """
    # Initialise counter for number of images written.
    n_image = 0

    # Retrieve images for processing.
    associations = "structure_sets" if dataset_id else None
    images_to_write = filter_on_paths(
        study.get_images(image_type, associations),
        (images or {}).get(study.patient.id, None),
    )

    if images_to_write:
        if dataset_id is None:
            images_dir = fullpath(outdir, pathlib=True)
        else:
            dataset_dir = get_dataset_path(
                get_nnunet_env(outdir)["nnUNet_raw"], dataset_id
            )
            images_dir, labels_dir = get_training_dirs(
                outdir, dataset_id, dataset_name
            )
            make_dir(labels_dir, overwrite=False)

        make_dir(images_dir, overwrite=False)

    for image in images_to_write:
        # If an individual image dimension for resizing is None,
        # and the corresponding voxel dimension isn't None,
        # set the image dimension in voxels so as to retain
        # the original image dimension in mm.
        if image_size is not None and voxel_size is not None:
            for idx in range(len(image_size)):
                if image_size[idx] is None and voxel_size[idx] is not None:
                    image_size[idx] = int(
                        image.get_n_voxels()[idx]
                        * image.get_voxel_size()[idx]
                        / self.voxel_size[idx]
                    )

        # Perform any image standardisation requested.
        image.resize(image_size=image_size, voxel_size=voxel_size)
        image.apply_banding(bands)

        # For training set, process image-associated structure set.
        if training_set:
            sset = image.get_structure_sets()[structure_set_to_write]
            sset.set_image(image)

            # Standardise ROI names.
            sset.rename_rois(roi_names, keep_renamed_only=True)

            # Remove any ROIs with zero volume inside the image area.
            to_remove = [
                roi.name
                for roi in sset.get_rois()
                if roi.get_volume(method="mask") < 0.1
            ]
            if to_remove:
                sset.filter_rois(to_remove=to_remove)

            # Check that structure set contains any required ROIs.
            if require_all_rois and len(sset.get_roi_names()) != len(roi_names):
                continue

            # Define list of ROI names for output structure set.
            # This is the same as the input list of names, except that
            # bilateral ROIs are split into left and right components.
            out_roi_names = list(roi_names)
            if bilateral_names:
                for name in bilateral_names:
                    if name in out_roi_names:
                        idx = out_roi_names.index(name)
                        out_roi_names[idx : idx + 1] = [
                            f"{name}_left",
                            f"{name}_right",
                        ]
                out_roi_names = tuple(out_roi_names)

            # Split bilateral ROIs into left and right components.
            sset.split_rois_in_two(bilateral_names)

            # Ensure that ROI masks have same geometry as image.
            if not sset.get_image().has_same_geometry(image):
                sset.set_image(image)

            # Define case-specific part of output file names.
            case_identifier = (
                f"{study.patient.id}_{study.timestamp}_"
                f"{image_type.upper()}_{image.timestamp}"
            )

            # Write image.
            image.write(images_dir / f"{case_identifier}_0000{suffix}")
            n_image += 1

            # Write structure set for training.
            if training_set:
                sset.set_image(image)
                for roi in sset.get_rois():
                    roi.create_mask(force=True)
                    sset.write(
                        f"{case_identifier}{self.suffix}",
                        labels_dir,
                        multi_label=True,
                        names=out_roi_names,
                    )

    return n_image


def get_app(setup_script=""):
    """
    Define and configure application to be run.
    """
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
        if "Linux" == system():
            roi_names = [
                "bladder",
                "femoral_head_left",
                "femoral_head_right",
                "prostate",
                "rectum",
                "seminal_vesicles",
            ]
        else:
            roi_names = ["rectum"]

        roi_lookup = prostate_plan

    opts["roi_names"] = {
        roi_name: [roi_name] + roi_lookup[roi_name] for roi_name in roi_names
    }

    opts["training_set"] = True

    opts["voxel_size"] = (1.5, 1.5, 3.0)
    opts["image_size"] = (256, 256, None)

    # opts["topdir"] = str(Path("~/nnunet/data").expanduser())
    opts["topdir"] = "/work/harrison/workshop/nnunet/data"

    opts["image_type"] = "MVCT"
    opts["dataset_id"] = 21
    opts["dataset_name"] = "pk"
    opts["test_fraction"] = 0.20

    if "Ganga" in __name__:
        opts["alg_module"] = fullpath(argv[0])

    # Set the severity level for event logging
    log_level = "INFO"

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
        data_dirs = [f"/r02/voxtox/workshop/synthetic_mvct/k_{SITE}"]
        # data_dirs = [f"/r02/voxtox/data/head_and_neck/vspecial/30_patients__spinal_cord__1_mv"]
        patterns = ["VT*"]
    else:
        data_dirs = [fullpath(f"~/data/voxtox_check/{SITE}")]
        patterns = ["*/VT*"]
        data_dirs = [
            fullpath(
                f"~/data/head_and_neck/vspecial/30_patients__spinal_cord__1_mv"
            )
        ]
        data_dirs = [fullpath(f"~/data/synthetic_mvct/{SITE}")]
        patterns = ["VT*"]

    return {data_dir: patterns for data_dir in data_dirs}


if "__main__" == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())

    # Run application for the selected data
    app.run(paths)

if "Ganga" in __name__:
    # Define script for setting analysis environment
    setup_script = fullpath("skrt_conda.sh")

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), setup_script)

    # Define the patient data to be analysed
    paths = get_paths(get_data_locations())
    input_data = PatientDataset(paths=paths)

    # Define processing system.
    if "Linux" == system():
        backend = Condor()
        backend.cdf_options["request_memory"] = "8G"
    else:
        backend = Local()

    # Define how job should be split into subjobs
    splitter = PatientDatasetSplitter(patients_per_subjob=5)
    splitter = None

    # Define merging of subjob outputs
    merger = SmartMerger()
    merger.files = ["stderr", "stdout"]
    merger.ignorefailed = True
    postprocessors = [merger]
    postprocessors = []

    # Define job name
    name = f"nnunet_{SITE}"

    # Define list of outputs to be saved
    outbox = []

    # Create the job, and submit to processing system
    j = Job(
        application=ganga_app,
        backend=backend,
        inputdata=input_data,
        outputsandbox=outbox,
        splitter=splitter,
        postprocessors=postprocessors,
        name=name,
    )
    j.submit()
