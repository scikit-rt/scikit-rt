'''
Application for evaluating atlas-based segmentation.
'''
import platform
import random
import sys

from pathlib import Path

import pandas as pd

from skrt import Image, Patient, StructureSet
from skrt.application import Algorithm, Application, get_paths
from skrt.core import Defaults, fullpath, is_list, qualified_name, tic, toc
from skrt.registration import get_default_pfiles, get_engine_name
from skrt.segmentation import SasTuner

global_side = "right"

class SasEvaluation(Algorithm):
    '''
    Algorithm subclass, for evaluating single-atlas segmentation.

    Methods:
        __init__ -- Return instance of SasEvaluation class,
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
        # Path to directory for registration output.
        self.workdir = "segmentation_workdir"

        # Paths to patient datasets against which to register.
        self.atlases = None

        # Number of atlases to consider, from those of self.atlases.
        # If None, no limit is placed on the number of atlases.
        self.n_atlas = None

        # Random number seed (relevant for random selection of atlases).
        self.seed = None

        # Flag indicating whether to exclude target as atlas for registrations
        # (i.e. to exclude self-registration).
        self.exclude_target = False

        # Type of selection to be performed for atlases.
        # Possible values are:
        # - "random": random selection;
        # - "mutual_information": select in order of mutual information;
        # - "fidelity": select in order of fidelity;
        # - "correlation_quality": select in order of correlation quality;
        # - None: select in order of input.
        self.atlas_selection = None

        # Registration engine, and path to associated software directory.
        self.engine = None
        self.engine_dir = None

        # Dictionary of names for renaming ROIs, where the keys are new
        # names and values are lists of possible names of ROIs that should
        # be assigned the new name.
        self.roi_names = None

        # Initial crop focus, and margins to leave when cropping.
        self.initial_crop_focus = None
        self.initial_crop_margins = None
        self.initial_crop_about_centre = False
        self.default_crop_margins = 0
        self.roi_crop_margins = None
        self.default_roi_crop_about_centre = False
        self.roi_crop_about_centre = None

        # Define whether target and atlas images should be cropped
        # to the same size prior to registration.  Size matching is performed
        # after cropping about focus.
        self.crop_to_match_size1 = True
        self.crop_to_match_size2 = True

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

        # Dictionaries of value bandings to be applied to image data.
        # Keys of a banding dictionary are floats specifying intensities
        # to be assigned, and values are two-element tuples indicating
        # lower and upper band limits.  If the first element is None,
        # no lower limit is applied; if the second element is None,
        # no upper limit is applied.
        #
        # self.bands1 : banding dictionary for global registration;
        # self.default_roi_bands : default banding dictionary for
        #     local registration;
        # self.roi_bands : dictionary where keys are ROI names and values
        #     are banding dictionaries for local registration focused on
        #     the specified ROI.
        self.bands1 = None
        self.default_roi_bands = None
        self.roi_bands = None

        # Structure on which to align for initial registration,
        # and name to be assigned to the associated registration step.
        self.initial_alignment = "_top_"
        self.initial_transform_name = None

        # Parameter files to be used in registration.
        self.pfiles1= {}
        self.pfiles2 = {}
        self.roi_pfiles = None

        # Variations to evaluate in registration parameters.
        self.pfiles1_variations = None
        self.pfiles2_variations = None

        # Indicate whether to retain for each segmented ROI only the contour
        # with most points in each image slice.
        self.most_points1 = True
        self.most_points2 = True

        # Define whether to suppress Elastix output.
        self.capture_output = True

        # Define whether to keep directory used for ROI transformations.
        self.keep_tmp_dir = False

        # Level for message logging.
        self.log_level = None
        if log_level is not None:
            Defaults({"log_level": log_level})

        # Segmentation and registration steps to consider in comparisons.
        self.steps = -1
        self.reg_steps = -1

        # Strategy / strategies for mapping ROIs from atlas to target.
        # to frame of planning scan:
        # "push": push ROI contour points from atlas (fixed) to target (moving);
        # "pull": pull ROI masks from atlas (moving) to target (fixed).
        self.strategies = "pull"

        # Names of ROIs to consider in evaluations.
        self.to_keep = None
        
        # Voxel size for evaluations.
        self.voxel_size = None

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

        # Try to ensure that name of registration engine is set.
        self.engine = get_engine_name(self.engine, self.engine_dir)


        # If parameter files not specified, set engine-dependent defaults.
        if not self.pfiles1 :
            self.pfiles1 = {
                    "bspline": get_default_pfiles("*BSpline15*", self.engine)[0]
                    }
            self.pfiles1 = {} # Hello
        if not self.pfiles2:
            self.pfiles2 = dict(self.pfiles1)

        # Ensure that output file is always created, even if empty.
        Path(self.analysis_csv).touch()

        # Dataframe for comparison metrics.
        self.df_analysis = None

        # Set random-number seed.
        random.seed(self.seed)

    def execute(self, patient=None):
        '''
        Perform and evaluate single-atlas segmentations.

        **Parameter:**

        patient: skrt.patient.Patient, default=None
            Object providing access to patient dataset.
        '''

        # Print details of current patient.
        self.logger.info(f"Patient id: {patient.id}")
        self.logger.info(f"Folder path: {patient.path}")

        self.im1 = self.get_target_image(patient)
        exclude_ids = [patient.id] if self.exclude_target else None

        atlas_objs = self.select_atlases(
                self.n_atlas, exclude_ids, self.atlas_selection)

        for atlas_id in atlas_objs:
            self.logger.info(f"Atlas id: {atlas_id}")
            tuner = SasTuner(
                    pfiles1_variations=self.pfiles1_variations,
                    pfiles2_variations=self.pfiles2_variations,
                    im1=self.im1,
                    im2=atlas_objs[atlas_id][0],
                    ss1=None,
                    ss2=atlas_objs[atlas_id][1],
                    log_level=self.log_level,
                    engine = self.engine,
                    engine_dir = self.engine_dir,
                    workdir=(Path(self.workdir)
                             / self.engine / f"{patient.id}_{atlas_id}"),
                    roi_names=self.roi_names,
                    initial_crop_focus=self.initial_crop_focus,
                    initial_crop_margins=self.initial_crop_margins,
                    initial_crop_about_centre=self.initial_crop_about_centre,
                    initial_alignment=self.initial_alignment,
                    initial_transform_name=self.initial_transform_name,
                    crop_to_match_size1=self.crop_to_match_size1,
                    voxel_size1=self.voxel_size1,
                    bands1=self.bands1,
                    pfiles1=self.pfiles1,
                    most_points1=self.most_points1,
                    roi_crop_margins=self.roi_crop_margins,
                    default_roi_crop_margins=self.default_roi_crop_margins,
                    roi_crop_about_centre=self.roi_crop_about_centre,
                    default_roi_crop_about_centre=(
                        self.default_roi_crop_about_centre),
                    crop_to_match_size2=self.crop_to_match_size2,
                    voxel_size2=self.voxel_size2,
                    default_roi_bands=self.default_roi_bands,
                    roi_bands=self.roi_bands,
                    pfiles2=self.pfiles2,
                    most_points2=self.most_points2,
                    capture_output=self.capture_output,
                    keep_tmp_dir=self.keep_tmp_dir,
                    id1=patient.id,
                    id2=atlas_id,
                    strategies=self.strategies,
                    steps=self.steps,
                    to_keep=self.to_keep,
                    metrics=self.metrics,
                    slice_stats=self.slice_stats,
                    default_by_slice=self.default_by_slice,
                    voxel_size=self.voxel_size,
                    )
            print(tuner.df)

            # Append dataframe for current comparision to global dataframe.
            if self.df_analysis is None:
                self.df_analysis = tuner.df
            else:
                self.df_analysis = pd.concat(
                        [self.df_analysis, tuner.df], ignore_index=True)

        return self.status

    def finalise(self):
        # Write comparison table in CSV format.
        if self.df_analysis is not None:
            self.df_analysis.to_csv(self.analysis_csv)

        return self.status
    
    def get_target_image(self, patient):
        """
        Obtain target image for segmentation.

        This implementation returns the first image of any type associated
        with the patient, with the assumption that the image has an
        associated structure set.

        This function may need to be overridden in a subclass.

        **Parameter:**

        patient : skrt.patient.Patient
            Object providing access to patient dataset.
        """
        return patient.combined_objs("image_types")[0]

    def get_atlas_objs(self):
        """
        Obtain dictionary of atlas objects against which to register.

        Dictionary keys are identifiers, and values are tuples of
        Image and StructureSet.  If the StructureSet to consider is
        the first element of the Image object's structure_sets attribute,
        the second element of the tuple may be set to None.

        This implementation returns the first image of any type
        associated with each atlas.

        This function may need to be overridden in a subclass.
        """
        atlas_objs = {}
        for atlas in self.atlases:
            patient = Patient(atlas, unsorted_dicom=True)
            im = patient.combined_objs("image_types")[0]
            atlas_objs[patient.id] = (im, None)
        return atlas_objs

    def select_atlases(self, n_atlas=None, exclude_ids=None, selection=None):

        allowed_selections = {None, "random", "mutual_information",
                              "fidelity", "correlation_quality"}
        if selection not in allowed_selections:
            raise RuntimeError(f"Selection {selection} not allowed; "
                               f"allowed selections: {allowed_selections}")

        atlas_objs = self.get_atlas_objs()
        if n_atlas is None and exclude_ids is None and selection is None:
            return atlas_objs

        if exclude_ids:
            if not is_list(exclude_ids):
                exclude_ids = [exclude_ids]
            atlas_ids = [atlas_id for atlas_id in atlas_objs
                         if atlas_id not in exclude_ids]
        else:
            atlas_ids = list(atlas_objs)

        n_atlas = n_atlas if n_atlas is not None else len(atlas_ids)

        if n_atlas > len(atlas_ids):
            raise RuntimeError(f"Request selection of {n_atlas} atlases "
                               f"but only {len(atlas_ids)} atlases "
                               f"after exclusions: {atlas_ids}")

        if selection in ["random", None]:
            if "random" == selection:
                atlas_ids = random.sample(atlas_ids, n_atlas)
            return {atlas_id: atlas_objs[atlas_id]
                    for atlas_id in list(atlas_ids)[: n_atlas]}

        if selection in ["mutual_information", "fidelity",
                         "correlation_quality"]:
            scores = {}
            for atlas_ids in atlas_objs:
                sas = SingleAtlasSegmentation(
                        im1=self.im1,
                        im2=atlas_objs[atlas_id][0],
                        ss1=None,
                        ss2=atlas_objs[atlas_id][1],
                        log_level=self.log_level,
                        engine = self.engine,
                        engine_dir = self.engine_dir,
                        workdir=(Path(self.workdir)
                                 / self.engine / "atlas_selection"),
                        overwrite=True,
                        auto=True,
                        auto_reg_setup_only=True,
                        roi_names=self.roi_names,
                        initial_crop_focus=self.initial_crop_focus,
                        initial_crop_margins=self.initial_crop_margins,
                        initial_crop_about_centre=(
                            self.initial_crop_about_centre),
                        initial_alignment=self.initial_alignment,
                        initial_transform_name=self.initial_transform_name,
                        crop_to_match_size1=self.crop_to_match_size1,
                        voxel_size1=self.voxel_size1,
                        bands1=self.bands1,
                        pfiles1=({} if self.initial_alignment
                                 else self.pfiles1),
                        capture_output=self.capture_output,
                        )
                reg = sas.get_registration(step=0)
                score = getattr(reg, f"get_{selection}")(reg_step=0)
                if score not in scores:
                    scores[score] = []
                scores[score].append(atlas_id)

            atlas_objs2 = {}
            i_atlas = 0
            for score, atlas_ids2 in sorted(scores.items(), reverse=True):
                for atlas_id in sorted(atlas_ids2):
                    atlas_objs2[atlas_id] = atlas_objs[atlas_id]
                    i_atlas += 1
                    if i_atlas >= n_atlas:
                        return atlas_objs2

def get_app(setup_script='', engine="elastix"):
    '''
    Define and configure application to be run.
    '''

    opts = {}
    if "Linux" == platform.system():
        if "elastix" == engine:
            opts["engine_dir"] = "~/sw20/elastix-5.0.1"
        opts["workdir"] = ("/r02/radnet/casscade/workdir")
    else:
        if "elastix" == engine:
            opts["engine_dir"] = "~/sw/elastix-5.0.1"
        elif "niftyreg" == engine:
            opts["engine_dir"] = "~/sw/niftyreg"
        opts["workdir"] = ("segmentation_workdir")

    opts["strategies"] = "pull"
    opts["roi_names"] = {
            "heart": ["heart", "Heart"],
            "imn" : ["CTV IMN", "CTV INM", "CTVn_IMN", "IMN", "ctv imn"],
            }
    opts["initial_alignment"] = "heart"
    opts["initial_crop_focus"] = "heart"
    opts["initial_crop_margins"] = ((-10, 10), (-60, 10), (-110, 110))
    opts["voxel_size1"] = None
    opts["bands1"] = {-1024:(None, 80)}
    opts["default_roi_crop_margins"] = (20, 10, 10)
    opts["roi_crop_margins"] = {"heart": (10, 30, 10), "imn": (40, 20, 10)}
    opts["voxel_size2"] = opts["voxel_size1"]
    opts["voxel_size"] = opts["voxel_size1"]
    opts["bands2"] = None
    opts["log_level"] = "INFO"
    opts["to_keep"] = "imn"
    opts["atlas_selection"] = None
    opts["exclude_target"] = True
    opts["seed"] = 1

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

    opts["metrics"] = [
            #"centroid_slice_stats",
            #"abs_centroid_slice_stats",
            #"area_ratio_slice_stats",
            #"centroid",
            #"abs_centroid",
            "dice",
            #"hausdorff_distance",
            #"jaccard",
            #"mean_over_contouring",
            #"mean_under_contouring",
            #"mean_signed_surface_distance",
            #"mean_surface_distance",
            #"rms_surface_distance",
            #"volume_diff",
            #"volume_ratio",
            ]
    
    opts["slice_stats"] = {"intersection": ["mean", "stdev"]}

    opts["analysis_csv"] = "analysis.csv"

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(sys.argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = SasEvaluation(opts=opts, name=None, log_level=log_level)

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
    # - tumour on left: 33 before exclusions => 29 after exclusions;
    # - tumour on right: 30 before exclusions => 26 after exclusions.
    # Numbers are after re-assignments.

    # The following were initially assigned as having tumour on left,
    # but in reality have tumour on right:
    # z0k1nxmgeo
    # z0k3qrijjo
    #
    # The following was initially assigned both as having tumour on
    # left and as having tumour on right, but in reality has tumour on right;
    # z0k3lxmiij
    #
    # The following was initially classified as having tumour on right,
    # but in reality has tumour on left:
    # z0k3ovnegl

    # Tumour on left, IMNs not outlined.
    left_no_imn = ["z0k0lufhhn", "z0k0mzjjij", "z0k2rtkjkh", "z0k4oykeio"]
    # Tumour on left, two IMN outlines ('CTV IMN', 'IMN').
    # => Two outlines seem to be the same: use either.
    # left_two_imn = ["z0k4izonnq"]
    # Tumour on left, two structure sets.
    # => At least for heart and IMNs, two structure sets seem to be the same:
    # use either.
    # left_two_ss = ["z0k4pxmkji"]
    # Tumour on right, IMNs not outlined.
    right_no_imn = ["z0k0mzlefq", "z0k3owkffh", "z0k3oylnij"]
    # Tumour on right, two structure sets and IMNs not outlined.
    right_two_ss = ["z0k4owmnlj"]
    
    return (left_no_imn + right_no_imn + right_two_ss)

def get_data_loader():
    # Define class and options for loading patient data.
    PatientClass = Patient
    patient_opts = {"unsorted_dicom": True}
    patient_class = qualified_name(PatientClass)

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
        input_data = paths
    else:
        input_data = paths[:1]

    # Set paths to atlases.
    app.algs[0].atlases = paths

    # Set number of atlases to consider
    app.algs[0].n_atlas = 2

    # Allow target among atlases.
    app.algs[0].exclude_target = False

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
        input_data = PatientDataset(paths=paths[ :])
    else:
        input_data = PatientDataset(paths=paths[:1])

    # Set paths to atlases.
    opts["atlases"] = paths

    # Set number of atlases to consider
    opts["n_atlas"] = n_atlas

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

    # Define list of outputs to be saved.
    outbox = [opts["analysis_csv"]]

    # Define files to be merged.
    merger.files = ['stderr', 'stdout', opts["analysis_csv"]]

    # Create the job, and submit to processing system.
    j = Job(application=ganga_app, backend=backend,
            inputdata=input_data,
            outputfiles=outbox, splitter=splitter,
            postprocessors=postprocessors, name=name)
    j.submit()
