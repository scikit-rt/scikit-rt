'''
Application for atlas-based segmentation.
'''
import platform
import shutil
import sys
import timeit

from itertools import chain
from pathlib import Path

import pandas as pd

from skrt import Patient, StructureSet
from skrt.application import Algorithm, Application, Status, get_paths
from skrt.core import Data, fullpath, is_list, tic, toc
from skrt.image import match_image_voxel_sizes
from skrt.registration import get_default_pfiles_dir, set_elastix_dir
from skrt.segmentation import SingleAtlasSegmentation

global_side = "right"

class Segmentation(Algorithm):
    '''
    Algorithm subclass, for atlas-based segmentation.

    Methods:
        __init__ -- Return instance of Segmentation class,
                    with properties set according to options dictionary.
        execute  -- Perform atlas-based segmentation.
    '''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Return instance of Segmentation class.

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
        self.bands1 = None
        self.bands2 = None

        # Paths to patient datasets against which to register.
        self.atlases = None

        # Structure on which to align for cropping and initial translation.
        # If the value given is the name of an ROI present in the structure
        # sets of the two images being aligned, cropping is performed
        # around this ROI.  Otherwise the structure is used only for alignment.
        self.initial_alignment = "__top__"

        self.initial_transform_name = None

        # Buffers ((x1, x2), (y1, y2), (z1, z1))  to leave
        # when cropping around structure used in alignment.
        # If None, no cropping is performed.
        self.initial_alignment_crop = True
        self.initial_alignment_crop_margins = None
        self.default_crop_margins = 0
        self.roi_crop_buffers = None

        # Define whether planning scan and relapse scan should be cropped
        # to the same size prior to registration.  Size matching is performed
        # after cropping of relapse scan to structure sets.
        self.crop_to_match_size1 = False
        self.crop_to_match_size2 = False

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
        self.voxel_size1 = None
        self.voxel_size2 = None

        # Path to registration output directory.
        self.workdir = "segmentation_workdir"

        # Obtain path to default Elastix parameter files.
        self.pfiles_dir = get_default_pfiles_dir()

        # Define whether to overwrite any previous registration results.
        self.overwrite = True

        # Define whether to suppress Elastix output.
        self.capture_output = True

        self.keep_tmp_dir = False

        # Parameter files to be used in registration.
        self.pfiles1={
                "translation": self.pfiles_dir / "MI_Translation.txt",
                "bspline": self.pfiles_dir / "MI_BSpline30.txt",
            }

        self.pfiles2 = dict(self.pfiles1)

        self.most_points1 = True
        self.most_points2 = True

        self.auto_step = -1
        self.log_level = None

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

        # Specification of statistics to be calculated relative
        # to slice-by-slice metric values.  Used for all metrics with
        # suffix "_slice_stats" in the self.metrics list.
        self.slice_stats = None

        # Default specification of slices to be considered when
        # calculating slice-by-slice statistics.
        self.default_by_slice = None

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
        Perform atlas-based segmentation

        **Parameter:**

        patient: skrt.patient.Patient, default=None
            Object providing access to patient dataset.
        '''

        # Print details of current patient.
        self.logger.info(f"Patient id: {patient.id}")
        self.logger.info(f"Folder path: {patient.path}")

        im1 = patient.combined_objs("ct_images")[0]

        for atlas in self.atlases:
            patient2 = Patient(atlas, unsorted_dicom=True)
            self.logger.info(f"Atlas id: {patient2.id}")
            self.logger.info(f"Atlas path: {patient2.path}")
            im2 = patient2.combined_objs("ct_images")[0]

            self.sas = SingleAtlasSegmentation(
                im1=im1,
                im2=im2,
                ss1=None,
                ss2=None,
                log_level=self.log_level,
                workdir=Path(self.workdir) / f"{patient.id}_{patient2.id}",
                overwrite=self.overwrite,
                auto=True,
                auto_step=self.auto_step,
                strategy=self.strategy,
                roi_names=self.roi_names,
                initial_alignment=self.initial_alignment,
                initial_alignment_crop=self.initial_alignment_crop,
                initial_alignment_crop_margins=\
                        self.initial_alignment_crop_margins,
                initial_transform_name=self.initial_transform_name,
                crop_to_match_size1=self.crop_to_match_size1,
                voxel_size1=self.voxel_size1,
                bands1=self.bands1,
                pfiles1=self.pfiles1,
                most_points1=self.most_points1,
                roi_crop_margins=self.roi_crop_margins,
                default_crop_margins=self.default_crop_margins,
                crop_to_match_size2=self.crop_to_match_size2,
                voxel_size2=self.voxel_size2,
                bands2=self.bands2,
                pfiles2=self.pfiles2,
                most_points2=self.most_points2,
                capture_output=self.capture_output,
                keep_tmp_dir=self.keep_tmp_dir,
            )

            if self.metrics:
                self.compare_rois(patient, patient2)

        return self.status

    def finalise(self):
        # Write comparison table in CSV format.
        if self.df_analysis is not None:
            self.df_analysis.to_csv(self.analysis_csv)

        return self.status

    def compare_rois(self, patient1, patient2):
        print("Hello - compare_rois()", patient1.id, patient2.id)
        print(self.sas.ss1_filtered.path)
        print(self.sas.get_segmentation(step=self.auto_step).path)
        # Compare transformed ROIs and reference ROIs
        df = self.sas.ss1_filtered.get_comparison(
                other=self.sas.get_segmentation(step=self.auto_step),
                metrics=self.metrics, slice_stats=self.slice_stats,
                default_by_slice=self.default_by_slice)

        # Set "patient_id", "atlas_id" and "roi" as indices.
        df.index.name = "roi"
        df.set_index([
            pd.Series(df.shape[0] * [patient1.id], name="patient_id"),
            pd.Series(df.shape[0] * [patient2.id], name="atlas_id"),
            df.index], inplace=True)

        # Append dataframe for current patient to global dataframe.
        if self.df_analysis is None:
            self.df_analysis = df
        else:
            self.df_analysis = pd.concat([self.df_analysis, df])

def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    if "Linux" == platform.system():
        opts["elastix_dir"] = "~/sw20/elastix-5.0.1"
        opts["workdir"] = ("/r02/radnet/casscade/workdir")
    else:
        opts["elastix_dir"] = "~/sw/elastix-5.0.1"
        opts["workdir"] = ("segmentation_workdir")

    opts["strategy"] = "pull"
    opts["roi_names"] = {
            "heart": ["heart", "Heart"],
            "imn" : ["CTV IMN", "CTV INM", "CTVn_IMN", "IMN", "ctv imn"],
            }
    opts["initial_alignment"] = "heart"
    opts["initial_alignment_crop"] = True
    opts["initial_alignment_crop_margins"] = ((-10, 10), (-60, 10), (-110, 110))
    opts["crop_to_match_size1"] = True
    opts["voxel_size1"] = (1, 1, None)
    opts["bands1"] = {-1024:(None, 80)}
    opts["default_crop_margins"] = (10, 10, 5)
    opts["roi_crop_margins"] = {"heart": (10, 30, 10)}
    opts["crop_to_match_size2"] = True
    opts["voxel_size2"] = opts["voxel_size1"]
    opts["bands2"] = None
    opts["log_level"] = "WARNING"

    opts["metrics"] = [
            "centroid_slice_stats",
            "abs_centroid_slice_stats",
            "area_ratio_slice_stats",
            "centroid",
            "abs_centroid",
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
    
    opts["slice_stats"] = {"intersection": ["mean", "stdev"]}

    opts["analysis_csv"] = "analysis.csv"

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = Segmentation(opts=opts, name=None, log_level=log_level)

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
        input_data = paths[0:2]

    # Set paths to atlases.
    app.algs[0].atlases = paths[0:3]

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
    paths = get_paths(get_data_locations(global_side),
            None, None, get_to_exclude())
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
    workdir = Path(opts["workdir"])

    # Loop over transformation options.
    for strategy in ["push", "pull"]:
        for alignment in ["heart"]:
            for crop_margins in [((-10, 10), (-60, 10), (-110, 110))]:
                for crop_to_match_size in [True]:
                    for voxel_size in [None, (2, 2, None)]:
                        # Define job name.
                        cm = "".join([f"{str(xyz1)}_{str(xyz2)}"
                            for xyz1, xyz2 in crop_margins])
                        if is_list(voxel_size):
                            vs = "x".join([str(dxyz) for dxyz in voxel_size])
                        else:
                            vs = str(voxel_size)
                        name = (f"{global_side}_{strategy}_"
                                f"{alignment.replace('_', '')}_"
                                f"{cm}_{crop_to_match_size}_{vs}")

                        # Update algorithm options.
                        opts["strategy"] = strategy
                        opts["initial_alignment"] = alignment
                        opts["initial_alignment_crop_margins"] = crop_margins
                        opts["crop_to_match_size1"] = crop_to_match_size
                        opts["crop_to_match_size2"] = crop_to_match_size
                        opts["voxel_size1"] = voxel_size
                        opts["voxel_size2"] = voxel_size
                        opts["workdir"] = str(workdir) / name
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
