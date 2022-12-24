'''
Application for mapping ROIs between reference frames, for IMPORT data.
'''
import platform
import sys
import timeit

from itertools import chain
from pathlib import Path

import pandas as pd

from skrt import Patient
from skrt.application import Algorithm, Application, Status, get_paths
from skrt.core import Data, fullpath, is_list, tic, toc
from skrt.image import match_image_voxel_sizes
from skrt.registration import (get_default_pfiles_dir, Registration,
        set_elastix_dir)

global_side = "right"

class RoiTransform(Algorithm):
    '''
    Algorithm subclass, for mapping ROIs between reference frames.

    Methods:
        __init__ -- Return instance of RoiTransform class,
                    with properties set according to options dictionary.
        execute  -- Perform ROI mappings.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of RoiTransform class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''
        # Path to Elastix installation.
        self.elastix_dir = None

        # Dictionary of names for renaming ROIs, where the keys are new
        # names and values are lists of possible names of ROIs that should
        # be assigned the new name.
        self.roi_names = None

        # Dictionary of value bandings to be applied to image data.
        # Keys are floats specifying intensities to be assigned, and
        # values are two-element tuples indicating lower and upper
        # band limits.  If the first element is None, no lower limit
        # is applied; if the second element is None, no upper limit
        # is applied.
        self.bands = None

        # Paths to patient datasets against which to register.
        self.atlases = None

        # Structure on which to align for cropping and initial translation.
        # If the value given is the name of an ROI present in the structure
        # sets of the two images being aligned, cropping is performed
        # around this ROI.  Otherwise the structure is used only for alignment.
        self.alignment = "__top__"

        # Buffers ((x1, x2), (y1, y2), (z1, z1))  to leave
        # when cropping around structure used in alignment.
        # If None, no cropping is performed.
        self.crop_buffers = ((0, 0), (0, 0), (0, 0))

        # Define whether planning scan and relapse scan should be cropped
        # to the same size prior to registration.  Size matching is performed
        # after cropping of relapse scan to structure sets.
        self.crop_to_match_size = False

        # Define voxel size for image resizing prior to registration.
        # Possible values are:
        # None - no resizing performed;
        # "dz_max" - the image with smaller slice thickness is resized
        #     to have the same voxel size as the image sith larger
        #     slice thickness;
        # "dz_min" - the image with larger slice thickness is resized
        #     to have the same voxel size as the image sith smaller
        #     slice thickness;
        # (dx, dy, dz) - images are resized to have voxels with the
        #     specified dimensions in mm.
        self.voxel_size = None

        # Path to registration output directory.
        self.registration_outdir = "registration_results"

        # Obtain path to default Elastix parameter files.
        self.pfiles_dir = get_default_pfiles_dir()

        # Define whether to overwrite any previous registration results.
        self.overwrite = True

        # Define whether to suppress Elastix output.
        self.capture_output = True

        # Parameter files to be used in registration.
        self.pfiles={
                "translation": self.pfiles_dir / "MI_Translation.txt",
                "bspline": self.pfiles_dir / "MI_BSpline30.txt",
            }

        # Strategy for transforming ROIs from frame of relapse scan
        # to frame of planning scan:
        # "push": push ROI contour points from frame of
        # relapse scan (fixed image) # to frame of planning scan (moving image);
        # "pull": pull ROI masks from frame of relapse scan (moving image)
        # to frame of planning scan (fixed image).
        self.strategy = "pull"

        # Metrics to be evaluated for comparing transformed ROIs
        # and reference ROIs.
        self.metrics = None

        # File to which to write dataframe of comparison metrics.
        self.analysis_csv = "analysis.csv"

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        # Set path to Elastix installation.
        if isinstance(self.elastix_dir, str):
            self.elastix_dir = Path(fullpath(self.elastix_dir))
            set_elastix_dir(self.elastix_dir)

        # Check that strategy for transforming ROIs is valid.
        valid_strategies = ["pull", "push"]
        if self.strategy not in valid_strategies:
            raise RuntimeError(
                    f"Invalid transformation strategy: {self.strategy}; "
                    f"strategy should be one of {valid_strategies}")

        # Ensure that output file is always created, even if empty.
        Path(self.analysis_csv).touch()

        # Dataframe for comparison metrics.
        self.df_analysis = None

    def execute(self, patient=None):
        '''
        Map ROIs from relapse scan to planning scan.

        **Parameter:**

        patient: import_analysis.patient.ImportPatient, default=None
            Object providing access to patient dataset.
        '''

        # Print details of current patient.
        self.logger.info(f"Patient id: {patient.id}")
        self.logger.info(f"Folder path: {patient.path}")

        # Retreive and check patient image and structure set.
        self.im1, self.ss1, status = self.get_image_and_structure_set(patient)
        self.status.copy_attributes(status)
        if not self.status.ok():
            return self.status

        # Perform banding of image grey levels.
        self.im1.apply_selective_banding(self.bands)

        self.patient1 = patient
        for atlas in self.atlases:
            self.patient2 = Patient(atlas, unsorted_dicom=True)
            self.logger.info(f"Atlas id: {self.patient2.id}")
            self.logger.info(f"Atlas path: {self.patient2.path}")

            # Retreive and check atlas image and structure set.
            self.im2, self.ss2, status = self.get_image_and_structure_set(
                    self.patient2)
            if not status.ok():
                self.logger.warning("Unable to obtain image and structure set: "
                        f"{atlas}")
                self.logger.warning(status.reason)
                continue

            # Perform banding of image grey levels.
            self.im2.apply_selective_banding(self.bands)

            # Register patient image and atlas image.
            status = self.register()
            if not status.ok():
                self.warning(f"{self.patient1.id} vs {self.patient2.id} - "
                        f"{status.reason.lower()}")
                continue

            # Transform the atlas structure set.
            self.ss2_transformed = self.reg.transform(
                    self.ss2, transform_points=(self.strategy == "push"))
            self.ss2_transformed.set_image(self.im1)

            if self.metrics:
                self.compare_rois()

        return self.status

    def finalise(self):
        # Write comparison table in CSV format.
        if self.df_analysis is not None:
            self.df_analysis.to_csv(self.analysis_csv)

        return self.status

    def register(self):
        """
        Perform registration for current images and configuration.
        """
        # Crop primary image to region around alignment structure.
        if (self.alignment in self.ss1.get_roi_names()
                and self.alignment in self.ss2.get_roi_names()
                and self.crop_buffers is not None):
            roi_extents = self.ss1[self.alignment].get_extents()
            for idx1 in range(3):
                 for idx2 in range(2):
                   roi_extents[idx1][idx2] += self.crop_buffers[idx1][idx2]

            self.im1.crop(*roi_extents)

        # Crop images to same size.
        if self.crop_to_match_size:
            self.im2.crop_to_image(self.im1, alignment=self.alignment)

        # Resample images to same voxel size.
        if self.voxel_size:
            match_image_voxel_sizes(self.im1, self.im2, self.voxel_size)

        # Now that any resizing has been performed, set structure-set images.
        self.ss1.set_image(self.im1)
        self.ss2.set_image(self.im2)

        # Define fixed image based on strategy for contour propagation.
        if "push" == self.strategy:
            fixed = self.im2
            moving = self.im1
        else:
            fixed = self.im1
            moving = self.im2

        # Define the registration strategy.
        self.reg = Registration(
                Path(f"{self.registration_outdir}/{global_side}/"
                     f"{self.patient1.id}_{self.patient2.id}"),
                fixed = fixed,
                moving = moving,
                initial_alignment = self.alignment,
                pfiles=self.pfiles,
                overwrite=True,
                capture_output=True,
                keep_tmp_dir = True,
                )
        self.reg.register()

        # Exit if registration failed for any step:
        status = Status()
        failures = []
        for step in self.reg.steps:
            if not step in self.reg.tfiles:
                status.code = 11
                status.name = "RegistrationFailed"
                failures.append(step)
        if failures:
            status.reason = f"Registration failure(s): {failures}"

        return status

    def compare_rois(self):
        # Compare transformed ROIs and reference ROIs
        df = self.ss1.get_comparison(
                other=self.ss2_transformed, metrics=self.metrics)

        # Set "patient_id", "atlas_id" and "roi" as indices.
        df.index.name = "roi"
        df.set_index([
            pd.Series(df.shape[0] * [self.patient1.id], name="patient_id"),
            pd.Series(df.shape[0] * [self.patient2.id], name="atlas_id"),
            df.index], inplace=True)

        # Append dataframe for current patient to global dataframe.
        if self.df_analysis is None:
            self.df_analysis = df
        else:
            self.df_analysis = pd.concat([self.df_analysis, df])

    def get_image_and_structure_set(self, patient):
        """
        Get image and structure set for patient.

        Check that the patient's dataset contains only one image,
        with one associated structure set, and that this
        structure set contains all required ROIs.  Set non-zero status
        code for first check failed.
        """
        # Create local status object.
        status = Status()

        # Check that the patient dataset contains only one image.
        images = patient.combined_objs("image_types")
        if 1 != len(images):
            status.code = 1
            status.name = "images_not_1"
            status.reason = f"Number of images: {len(images)}"

        if not status.ok():
            return (None, None, status)

        # Check that the image has only one associated structure set.
        im = images[0].clone()

        if 1 != len(im.structure_sets):
            status.code = 2
            status.name = "structure_sets_not_1"
            status.reason = (
                    f"Number of structure sets: {len(im.structure_sets)}")

        if not status.ok():
            return (None, None, status)

        # Check that the structure set contains all required ROIs.
        if self.roi_names is None:
            ss = im.structure_sets[0].clone()
        else:
            ss = im.structure_sets[0].filtered_copy(names=self.roi_names,
                    keep_renamed_only=True, copy_roi_data=False)

            if len(ss.get_roi_names()) != len(self.roi_names):
                missing_rois = [roi for roi in self.roi_names
                        if roi not in ss.get_roi_names()]
                status.code = 3
                status.name = "rois_missing"
                status.reason = (f"Missing rois: {missing_rois}")

        # Assign structure set to image.
        im.assign_structure_set(ss)

        if not status.ok():
            return (None, None, status)

        return (im, ss, status)

def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    if "Linux" == platform.system():
        opts["elastix_dir"] = "~/sw20/elastix-5.0.1"
        opts["registration_outdir"] = (
                "/r02/radnet/import/registration_results")
    else:
        opts["elastix_dir"] = "~/sw/elastix-5.0.1"
        opts["registration_outdir"] = "registration_results"

    opts["roi_names"] = {
            "heart": ["heart", "Heart"],
            "imn" : ["CTV IMN", "CTV INM", "CTVn_IMN", "IMN", "ctv imn"],
            }

    opts["bands"] = {-1024:(None, 80)}
    opts["alignment"] = "heart"
    opts["crop_buffers"] = ((-10, 10), (-60, 10), (-110, 110))
    opts["crop_to_match_size"] = True
    opts["voxel_size"] = None
    opts["strategy"] = "pull"
    opts["capture_output"] = True
    opts["overwrite"] = True
    opts["metrics"] = [
            "centroid",
            "dice",
            "hausdorff_distance",
            "jaccard",
            "mean_over_contouring",
            "mean_under_contouring",
            "mean_signed_surface_distance",
            "mean_surface_distance",
            "rms_surface_distance",
            "volume_diff",
            "volume_ratio",
            ]
    opts["analysis_csv"] = "analysis.csv"

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = RoiTransform(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

def get_data_locations(side=None):
    """
    Specify locations of patient datasets.

    **Parameter:**

    side : str, default=None
        If "left" or "right", only consider data locations for tumour on
        this side.  Otherwise ignored.
    """
    # Define the patient data to be analysed
    if "Linux" == platform.system():
        data_dirs = [f"/r02/radnet/casscade"]
    else:
        data_dirs = [fullpath(f"~/data/casscade")]

    if side in ["left", "right"]:
        patterns = [f"casscade_{side}*/z*"]
    else:
        patterns = ["casscade*/z*"]

    return {data_dir: patterns for data_dir in data_dirs}

def get_to_exclude():
    """
    Datasets identified as to be excluded, pending further investigation.
    """
    # Initial dataset:
    # - tumour on left: 35 before exclusions => 29 after exclusions;
    # - tumour on right: 29 before exclusions => 25 after exclusions.

    # Tumour on left, IMNs not outlined.
    left_no_imn = ["z0k0lufhhn", "z0k0mzjjij", "z0k2rtkjkh", "z0k4oykeio"]
    # Tumour on left, two IMN outlines ('CTV IMN', 'IMN').
    left_two_imn = ["z0k4izonnq"]
    # Tumour on left, two structure sets.
    left_two_ss = ["z0k4pxmkji"]
    # Tumour on right, IMNs not outlined.
    right_no_imn = ["z0k0mzlefq", "z0k3owkffh", "z0k3oylnij"]
    # Tumour on right, two structure sets and IMNs not outlined.
    right_two_ss = ["z0k4owmnlj"]
    
    return (left_no_imn + left_two_imn + left_two_ss
            + right_no_imn + right_two_ss)

def get_data_loader():
    # Define class and options for loading patient data.
    PatientClass = Patient
    patient_opts = {"unsorted_dicom": True}

    # Determine qualified name if PatientClass is a class.
    if isinstance(PatientClass, type):
        patient_class = f"{PatientClass.__module__}.{PatientClass.__name__}"
    else:
        patient_class = None

    return (PatientClass, patient_class, patient_opts)


if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define class and options for loading patient datasets.
    PatientClass, patient_class, patient_opts = get_data_loader()

    # Define the patient data to be analysed.
    paths = get_paths(get_data_locations(global_side),
            None, None, get_to_exclude())
    if "Linux" == platform.system():
        input_data = paths[0:]
    else:
        input_data = paths[0:1]

    # Set paths to atlases.
    app.algs[0].atlases = paths[0:]

    # Run application for the selected data.
    app.run(input_data, PatientClass, **patient_opts)

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
    paths = get_paths(get_data_locations(), None, None, get_to_exclude())
    if "Linux" == platform.system():
        input_data = PatientDataset(paths=paths[0:])
    else:
        input_data = PatientDataset(paths=paths[0:1])

    # Set paths to atlases.
    opts["atlases"] = paths

    # Define processing system.
    if "Linux" == platform.system():
        backend = Condor()
        backend.cdf_options["request_memory"]="8G"
    else:
        backend = Local()

    # Define how job should be split into subjobs.
    splitter = PatientDatasetSplitter(patients_per_subjob=1)

    # Set up output merging.
    merger = SmartMerger()
    merger.ignorefailed = True
    postprocessors = [merger]

    # Define output directory.
    registration_outdir = Path(opts["registration_outdir"])

    # Loop over transformation options.
    for strategy in ["push", "pull"]:
        for alignment in ["heart"]:
            for crop_buffers in [((-10, 10), (-60, 10), (-110, 110))]:
                for crop_to_match_size in [True]:
                    for voxel_size in [None, (2, 2, 2)]:
                        # Define job name.
                        cb = "".join([f"{str(xyz1)}_{str(xyz2)}"
                            for xyz1, xyz2 in crop_buffers])
                        if is_list(voxel_size):
                            vs = "x".join([str(dxyz) for dxyz in voxel_size])
                        else:
                            vs = str(voxel_size)
                        name = (f"{global_side}_{strategy}_"
                                f"{alignment.replace('_', '')}_"
                                f"{cb}_{crop_to_match_size}_{vs}")

                        # Update algorithm options.
                        opts["strategy"] = strategy
                        opts["alignment"] = alignment
                        opts["crop_buffers"] = crop_buffers
                        opts["crop_to_match_size"] = crop_to_match_size
                        opts["voxel_size"] = voxel_size
                        opts["registration_outdir"] = str(registration_outdir
                            / name)
                        opts["analysis_csv"] = f"{name}.csv"

                        # Define list of outputs to be saved.
                        outbox = [opts["analysis_csv"]]

                        # Define files to be merged.
                        merger.files = ['stderr', 'stdout',
                                opts["analysis_csv"]]

                        # Create the job, and submit to processing system.
                        j = Job(application=ganga_app, backend=backend,
                                inputdata=input_data,
                                outputfiles=outbox, splitter=splitter,
                                postprocessors=postprocessors, name=name)
                        j.submit()
