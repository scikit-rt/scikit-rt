'''
Application for mapping ROIs between reference frames, for IMPORT data.
'''
import platform
import sys
import timeit

from itertools import chain
from pathlib import Path

import pandas as pd

from skrt.application import Algorithm, Application
from skrt.core import Data, fullpath, is_list
from skrt.image import match_image_voxel_sizes
from skrt.registration import (get_default_pfiles_dir, Registration,
        set_elastix_dir)
from import_analysis import ImportPatient

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

        # Alignment for cropping and initial alignment for registration.
        self.alignment = "_top_"

        # Buffer around structure sets for cropping relapse scan.
        # (If None, relapse scan isn't cropped to structure sets.)
        self.crop_buffer = None

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
                "rigid": self.pfiles_dir / "MI_Rigid.txt",
                "bspline": self.pfiles_dir / "MI_BSpline30.txt",
            }

        # Strategy for transforming ROIs from frame of relapse scan
        # to frame of planning scan:
        # "push": push ROI contour points from frame of
        # relapse scan (fixed image) # to frame of planning scan (moving image);
        # "pull": pull ROI masks from frame of relapse scan (moving image)
        # to frame of planning scan (fixed image).
        self.strategy = "pull"

        # Metrics to be evaluated for comparing transformed relapse ROIs
        # and planning ROIs.
        self.metrics = None

        # Define whether to perform dose analysis.
        self.dose_analysis = False

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

        # Dataframe for comparison metrics and/or dose analysis.
        self.df_analysis = None

    def execute(self, patient=None):
        '''
        Map ROIs from relapse scan to planning scan.

        **Parameter:**

        patient: import_analysis.patient.ImportPatient, default=None
            Object providing access to patient dataset.
        '''

        # Print details of current patient.
        print(f"Patient id: {patient.id}")
        print(f"Folder path: {patient.path}")
        self.logger.info(f"Initialisation time: {patient._init_time:.2f} s")

        # Define references to scans and structure sets.
        ct_plan = patient.get_ct_plan()
        ct_relapse = patient.get_ct_relapse()
        ss_plan = patient.get_ss_plan()
        ss_relapse = patient.get_ss_relapse()
        ss_recurrence = patient.get_ss_recurrence()

        # If alignment ROI defined,
        # check that this is contained in structure sets.
        if (isinstance(self.alignment, str)
                and self.alignment not in ["_top_", "_middle_", "_bottom_"]):
            if self.alignment not in ss_relapse.get_roi_names():
                return self.status
            if self.alignment not in ss_plan.get_roi_names():
                return self.status

        tic = timeit.default_timer()
        if self.crop_buffer is not None:
            # Crop relapse scan to include structure-set ROIs plus margin.
            ct_relapse.crop_to_roi(ss_relapse + ss_recurrence,
                    buffer=self.crop_buffer)

        if self.crop_to_match_size:
            # Crop planning and relapse scan to same size.
            ct_plan.crop_to_image(ct_relapse, alignment=self.alignment)
            ct_relapse.crop_to_image(ct_plan, alignment=self.alignment)

        if self.voxel_size:
            # Resample images to same voxel size.
            match_image_voxel_sizes(ct_plan, ct_relapse, self.voxel_size)

        # Now that any resizing has been performed, set structure-set images.
        ss_plan.set_image(ct_plan)
        ss_relapse.set_image(ct_relapse)
        ss_recurrence.set_image(ct_relapse)

        toc = timeit.default_timer()
        if (self.crop_buffer is not None or self.crop_to_match_size
                or self.voxel_size):
            self.logger.info(f"Resizing time: {toc - tic:.2f} s")

        # Choose fixed and moving image based on transform strategy.
        if "push" == self.strategy:
            fixed = ct_relapse
            moving = ct_plan
            registration_subdir = "relapse_fixed"
        elif "pull" == self.strategy:
            fixed = ct_plan
            moving = ct_relapse
            registration_subdir = "plan_fixed"

        # Define registration strategy.
        reg = Registration(
            Path(self.registration_outdir) / patient.id,
            fixed=fixed,
            moving=moving,
            initial_alignment=self.alignment,
            pfiles=self.pfiles,
            overwrite=self.overwrite,
            capture_output=self.capture_output,
        )

        # Perform registration.
        tic = timeit.default_timer()
        reg.register()
        toc = timeit.default_timer()
        self.logger.info(f"Registration time: {toc - tic:.2f} s")

        # Exit if registration failed for any step:
        for step in reg.steps:
            if not step in reg.tfiles:
                self.logger.error(f"Registration failed for step: {step}")
                return self.status

        # RoiTransform ROIs.
        tic = timeit.default_timer()
        if "push" == self.strategy:
            # Push ROI contours from relapse frame to planning frame.
            ss_relapse_transformed = reg.transform(
                    ss_relapse, transform_points=True)
        elif "pull" == self.strategy:
            # Pull ROI masks from relapse frame to planning frame.
            ss_relapse_transformed = reg.transform(ss_relapse)
        ss_relapse_transformed.set_image(ct_plan)
        toc = timeit.default_timer()
        self.logger.info(f"RoiTransformation time: {toc - tic:.2f} s")

        if self.metrics or self.dose_analysis:
            """
            if "push" == self.strategy:
                # When ROI contours have been pushed, ensure that the
                # image used for creating ROI masks is large enough 
                # to include all contours.
                dummy_image =  (
                        ss_plan + ss_relapse_transformed).get_dummy_image(
                                voxel_size=ct_plan.get_voxel_size()[0: 2],
                                slice_thickness=ct_plan.get_voxel_size()[2])
                ss_plan.set_image(dummy_image)
                ss_relapse_transformed.set_image(dummy_image)
            """

            df = None
            if self.metrics:
                # Compare transformed relapse ROIs and planning ROIs
                df = ss_plan.get_comparison(
                        other=ss_relapse_transformed, metrics=self.metrics)

            if self.dose_analysis:
                # Obtain names of ROIs defined for both plan and relapse scans.
                roi_names = set(ss_plan.get_roi_names()).intersection(
                        ss_relapse_transformed.get_roi_names())

                # Define reference to plan dose, and resize to plan image.
                dose_sum = patient.get_dose_sum()
                dose_sum.match_size(ct_plan)

                # If the relapse scan is the fixed image, transform dose from
                # frame of planning scan to frame of relapse scan.
                dose_sum_transformed = (reg.transform(dose_sum).astype("dicom")
                        if "push" == self.strategy else None)

                # Create dictionary for storing dose information.
                records = {col: {} for col in
                        ["plan_dose", "relapse_dose", "relapse_dose_diff",
                            "transformed_dose", "transformed_dose_diff"]}

                # Loop over ROIs.
                for roi_name in roi_names:
                    # Calculate mean doses for plan and relapse ROIs,
                    # and their difference.
                    records["plan_dose"][roi_name] = dose_sum.get_mean_dose(
                            ss_plan[roi_name])
                    records["relapse_dose"][roi_name] = dose_sum.get_mean_dose(
                            ss_relapse_transformed[roi_name])
                    if None in [records["plan_dose"][roi_name],
                            records["relapse_dose"][roi_name]]:
                        records["relapse_dose_diff"][roi_name] = None
                    else:
                        records["relapse_dose_diff"][roi_name] = (
                                records["relapse_dose"][roi_name] -
                                records["plan_dose"][roi_name])

                    # Use the transformed dose to recalculate mean dose
                    # for relapse ROI, and difference from mean dose for
                    # plan ROI.
                    if dose_sum_transformed is not None:
                        records["transformed_dose"][roi_name] = (
                                dose_sum_transformed.get_mean_dose(
                                    ss_relapse[roi_name]))
                        if None in [records["plan_dose"][roi_name],
                                records["transformed_dose"][roi_name]]:
                            records["transformed_dose_diff"][roi_name] = None
                        else:
                            records["transformed_dose_diff"][roi_name] = (
                                    records["transformed_dose"][roi_name] -
                                    records["plan_dose"][roi_name])

                # Create dataframe for dose information.
                df_dose = pd.DataFrame(records)

                # If metrics dataframe non-null, append dose dataframe.
                if df is not None:
                    df = pd.concat([df, df_dose], axis=1)
                else:
                    df = df_dose

            if df is not None:
                # Set "patient_id" and "roi" as indices.
                df.index.name = "roi"
                df.set_index([
                    pd.Series(df.shape[0] * [patient.id], name="patient_id"),
                    df.index], inplace=True)

                # Append dataframe for current patient to global dataframe.
                if self.df_analysis is None:
                    self.df_analysis = df
                else:
                    self.df_analysis = pd.concat([self.df_analysis, df])

        return self.status

    def finalise(self):
        # Write comparison table in CSV format.
        if self.df_analysis is not None:
            self.df_analysis.to_csv(self.analysis_csv)

        return self.status


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

    opts["alignment"] = "_top_"
    opts["crop_buffer"] = None
    opts["crop_to_match_size"] = False
    opts["strategy"] = "push"

    opts["capture_output"] = True
    opts["overwrite"] = True

    opts["metrics"] = ["area_diff_flat", "centroid", "dice_flat",
            "hausdorff_distance_flat", "mean_over_contouring_flat",
            "mean_under_contouring_flat", "mean_surface_distance_flat",
            "jaccard_flat", "rel_area_diff_flat", "rms_surface_distance_flat"]
    opts["dose_analysis"] = True
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


def get_paths(max_path=None, ids=None):
    """
    Get list of paths to patient datasets.

    **Parameter:**

    max_path : int, default=None
        Maximum number of paths to return.  If None, return paths to all
        directories in data directory (currently hardcoded in this function).

    ids : str/list, default=None
        Patient identifier, or list of patient identifiers.  If non-null,
        only return dataset paths for the specified identifier(s).
    """

    # Define the patient data to be analysed
    if "Linux" == platform.system():
        data_dir = Path(
                "/r02/radnet/import/20220726_import_data_selection_plus_cam")
    else:
        data_dir = Path("~/data/20220331_import_data_selection").expanduser()

    patterns = ["import_high/H*", "import_low/L*"]
    paths = sorted([str(path) for path in chain.from_iterable(
            [data_dir.glob(pattern) for pattern in patterns])
            if path.is_dir()])
    if ids:
        if not is_list(ids):
            ids = [ids]
        paths = [path for path in paths
                if any([Path(path).match(id) for id in ids])]

    max_path = min(max_path if max_path is not None else len(paths), len(paths))

    return paths[0: max_path]


def get_data_loader():
    # Define class and options for loading patient data.
    PatientClass = ImportPatient
    patient_opts = {
            "load_dose_sum": True,
            "load_masks": False,
            }

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
    paths = get_paths(1)

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
        paths = get_paths()
    else:
        paths = get_paths(1)
    input_data = PatientDataset(paths=paths)

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
        for alignment in ["carina", "sternum", "spinal_canal", "_top_"]:
            for crop_buffer in [20, 200]:
                for crop_to_match_size in [True]:
                    for voxel_size in [None, (2, 2, 2)]:

                        # Define job name.
                        if is_list(voxel_size):
                            vs = "x".join([str(dxyz) for dxyz in voxel_size])
                        else:
                            vs = str(voxel_size)
                        name = (f"{strategy}_{alignment.replace('_', '')}_"
                                f"{crop_buffer}_{crop_to_match_size}_{vs}")

                        # Update algorithm options.
                        opts["strategy"] = strategy
                        opts["alignment"] = alignment
                        opts["crop_buffer"] = crop_buffer
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
