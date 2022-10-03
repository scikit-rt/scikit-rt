"""
Application to compare structure sets for Osairis datasets.
"""
from pathlib import Path
import platform

import pandas as pd

from skrt.application import Algorithm, Application


class CompareStructureSets(Algorithm):
    """
    Algorithm subclass, for comparing structure sets.

    Methods:
        __init__ -- Return instance of CompareStructureSets class,
                    with properties set according to options dictionary.
        execute  -- Extract summary data.
    """

    def __init__(self, opts={}, name=None, log_level=None):
        """
        Return instance of CompareStructureSets class.

        opts: dict, default={}
            Dictionary for setting algorithm attributes.

        name: str, default=''
            Name for identifying algorithm instance.

        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        """
        # mappings between preferred and possible ROI names.
        self.roi_names = {}

        # Metrics to be evaluated for structure-set comparisons.
        self.metrics = []

        # Number of decimal places to retain for comparison metrics.
        self.decimal_places = 5

        # Capitalise headings of comparisons dataframe,
        # and use spaces rather than underscores.
        self.nice_columns = True

        # Include metric units in comparisons dataframe.
        self.units_in_header = True

        # Path to output CSV file.
        self.comparison_csv = "comparison.csv"

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        self.df = None

    def execute(self, patient=None):
        """
        Compare structure sets.

        **Parameter:**

        patient: skrt.patient.Patient/None, default=None
            Object providing access to patient information.
        """
        # Write the patient identifier.
        print(f"\n{patient.id}")

        # The data of a Patient object are grouped by study.
        # Structure sets from auto-segmentation tend not to have
        # meaningful study assignments, so the approach here is
        # to combine images and structure sets across studies.
        ct_images = patient.combined_objs("ct_images")
        structure_sets = patient.combined_objs("structure_set_types")

        # The assumption is that each data directory contains a single CT image.
        # If the assumption isn't valid, raise an exception.
        if len(ct_images) != 1:
            raise RuntimeError(f"Number of CT image in {data_dir} is "
                               f"{len(ct_images)} - don't know what to do "
                                "with number of images different from 1")

        # Perform data cleaning for structure sets.
        for ss in structure_sets:
            # Standardise ROI names, and drop ROIs not of interest.
            ss.rename_rois(names=self.roi_names, keep_renamed_only=True)
            # Name the structure set based on the name of the data directory.
            ss.name = Path(ss.path).stem
            # DICOM Structure sets from auto-segmentation don't always
            # correctly reference the image to which they relate.
            # Explicitly set the structure-set image to be the only CT image
            # in the data directory.
            ss.set_image(ct_images[0])

        # Loop over pairs of structure sets.
        for idx, ss1 in enumerate(structure_sets[:-1]):
            for ss2 in structure_sets[idx + 1:]:
                # Indicate comparison being performed.
                comparison = f"{ss1.name}_vs_{ss2.name}"
                comparison = "_".join(comparison.split())
                print(f"....{comparison}")

                # Check that the mask shape is the same
                # for all matched ROIs in the pair of structure sets.
                # Raise exception if this isn't the case.
                roi_names = set(ss1.get_roi_names()).intersection(
                        ss2.get_roi_names())
                for roi_name in roi_names:
                    if (ss1[roi_name].get_mask().shape
                            != ss2[roi_name].get_mask().shape):
                        raise RuntimeError("Mask mismatch for {roi_name}: "
                                f"{ss1[roi_name].get_mask().shape} vs "
                                f"{ss2[roi_name].get_mask().shape}")

                # Obtain dataframe of comparison metrics.
                df  = ss1.get_comparison(ss2, metrics=self.metrics,
                        name_as_index=False)
                df.insert(0, "comparison", comparison)
                df.insert(0, "id", patient.id)

                # Add to the global dataframe.
                if self.df is None:
                    self.df = df
                else:
                    self.df = pd.concat([self.df, df], ignore_index=True)

        return self.status

    def finalise(self):
        # Write comparison table in CSV format.
        if self.df is not None:
            self.df.to_csv(self.comparison_csv, index=False)

        return self.status

def get_app():
    '''
    Define and configure application to be run.
    '''
    # Define run-time options.
    opts = {}

    # Mappings between preferred and possible ROI names.
    opts["roi_names"] = get_roi_names("names_hn")

    # Metrics to be evaluated for structure-set comparisons.
    opts["metrics"] = ["dice", "abs_centroid", "volume_diff",
            "rel_volume_diff", "hausdorff_distance", "mean_surface_distance",
            "mean_under_contouring", "mean_over_contouring",
            "mean_distance_to_conformity", "jaccard"]

    opts["metrics"] = ["dice", "abs_centroid"]

    # Number of decimal places to retain for comparison metrics.
    opts["decimal_places"] = 5

    # Capitalise headings of comparisons dataframe,
    # and use spaces rather than underscores.
    opts["nice_columns"] = True

    # Include metric units in comparisons dataframe.
    opts["units_in_header"] = True

    # Path to output CSV file.
    opts["comparison_csv"] = "comparison.csv"

    # Set the severity level for event logging.
    log_level = "INFO"

    # Create algorithm object.
    alg = CompareStructureSets(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run.
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

def get_paths(max_path=None, top_dir=".", data_dir="*"):
    # Determine the paths to data directories.
    paths = sorted(list(top_dir.glob(data_dir)))
    max_path = min(max_path if max_path is not None else len(paths),
            len(paths))
    return paths[0: max_path]

def get_roi_names(names_id="names_hn"):
    roi_names = {}

    # Define mappings between preferred and possible ROI names.
    roi_names["names_hn"] = {
        'External': ['zz_external REVIEW BEFORE CLINICAL USE', 'body', 'external', 'Skin', 'externalbody', 'external*body'],
        'SPC': ['spc', 'zz_spc_muscle REVIEW BEFORE CLINICAL USE', 'Sup_Constrict_Musc', 'spc*muscle'],
        'Left Parotid': ['left*parotid', 'parotid*left', 'parotid*l', 'zz_parotid_l REVIEW BEFORE CLINICAL USE', 'Parotid_L (NL004)', 'L*Parotid',],
        'Right Parotid': ['right*parotid', 'parotid*right', 'parotid*r', 'zz_parotid_r REVIEW BEFORE CLINICAL USE', 'Parotid_R (NL004)', 'R*Parotid'],
        'MPC': ['mpc', 'zz_mpc_muscle REVIEW BEFORE CLINICAL USE', 'Mid_Constrict_Musc', 'mpc*muscle'],
        'Left SMG': ['zz_smg_l REVIEW BEFORE CLINICAL USE', 'Glnd*Submand*L', 'Submandibular_L (NL004)', 'L*Submand*Glnd'],
        'Right SMG': ['zz_smg_r REVIEW BEFORE CLINICAL USE', 'Glnd*Submand*R', 'Submandibular_R (NL004)', 'R*Submand*Glnd'],
        'Spinal Cord': ['zz_spinal_cord REVIEW BEFORE CLINICAL USE', 'spinalcord', 'spinal*cord', 'SpinalCord (NL004)'],
        'Brainstem': ['zz_brainstem REVIEW BEFORE CLINICAL USE', 'brainstem', 'Brainstem (NL004)'],
        'Left Globe': ['zz_globe_l REVIEW BEFORE CLINICAL USE', 'eye*left', 'eye*l', 'l*eye', 'left*eye', 'Orbit_L (UK013)'],
        'Right Globe': ['zz_globe_r REVIEW BEFORE CLINICAL USE', 'eye*right', 'eye*r', 'r*eye', 'right*eye', 'Orbit_R (UK013)'],
        'Mandible': ['zz_mandible REVIEW BEFORE CLINICAL USE', 'Bone*Mandible', 'Mandible (NL004)', 'Mandible*Bone'],
        'Left Cochlea': ['zz_cochlea_l REVIEW BEFORE CLINICAL USE', 'cochlea*l', 'Cochlea_L (UK014)', 'L*Cochlea'],
        'Right Cochlea': ['zz_cochlea_r REVIEW BEFORE CLINICAL USE', 'cochlea*r', 'Cochlea_R (UK014)', 'R*Cochlea'],
        'Left Lens': ['zz_lens_l REVIEW BEFORE CLINICAL USE', 'lens*l', 'Lens_L (UK014)', 'l*lens'],
        'Right Lens': ['zz_lens_r REVIEW BEFORE CLINICAL USE', 'lens*r', 'Lens_R (UK014)', 'r*lens'],
        'Optic Chiasm': ['zz_optic_chiasm REVIEW BEFORE CLINICAL USE', 'OpticChiasm', 'optic*chiasm', 'Optic_Chiasm (UK014)'],
        'Left Optic Nerve': ['zz_optic_nerve_l REVIEW BEFORE CLINICAL USE', 'opticnrv*l', 'Opt_N_L (UK014)', 'L*OpticNrv'],
        'Right Optic Nerve': ['zz_optic_nerve_r REVIEW BEFORE CLINICAL USE', 'opticnrv*r', 'Opt_N_R (UK014)', 'R*OpticNrv'],
        'Pituitary Gland': ['zz_pituitary_gland REVIEW BEFORE CLINICAL USE', 'pituitary', 'Pituitary (UK013)'],
        'Left Lacrimal Gland': ['zz_lacrimal_gland_l REVIEW BEFORE CLINICAL USE', 'Glnd*Lacrimal*L', 'Lacrimal_L (UK013)', 'L*Lacrimal*Glnd'],
        'Right Lacrimal Gland': ['zz_lacrimal_gland_r REVIEW BEFORE CLINICAL USE', 'Glnd*Lacrimal*R', 'Lacrimal_R (UK013)', 'R*Lacrimal*Glnd']
    }

    roi_names["names_prostate"] = {
        'Left Femoral Head': ['zz_Femur_L REVIEW BEFORE CLINICAL USE', 'zz_Femur_L REVIEW BEFORE CLINICAL USE_old', 'zz_Femur_L REVIEW BEFORE CLINICAL USE_current'],
        'Right Femoral Head': ['zz_Femur_R REVIEW BEFORE CLINICAL USE', 'zz_Femur_R REVIEW BEFORE CLINICAL USE_old', 'zz_Femur_R REVIEW BEFORE CLINICAL USE_current'],
        'Rectum': ['zz_Rectum REVIEW BEFORE CLINICAL USE', 'zz_Rectum REVIEW BEFORE CLINICAL USE_old', 'zz_Rectum REVIEW BEFORE CLINICAL USE_current'],
        'Bladder': ['zz_Bladder REVIEW BEFORE CLINICAL USE', 'zz_Bladder REVIEW BEFORE CLINICAL USE_old', 'zz_Bladder REVIEW BEFORE CLINICAL USE_current'],
        'External': ['zz_External REVIEW BEFORE CLINICAL USE', 'zz_External REVIEW BEFORE CLINICAL USE_old', 'zz_External REVIEW BEFORE CLINICAL USE_current'],
        'Prostate': ['zz_Prostate REVIEW BEFORE CLINICAL USE', 'zz_Prostate REVIEW BEFORE CLINICAL USE_old', 'zz_Prostate REVIEW BEFORE CLINICAL USE_current'],
        'Seminal Vesicles': ['zz_SeminalVesicles REVIEW BEFORE CLINICAL USE', 'zz_SeminalVesicles REVIEW BEFORE CLINICAL USE_old', 'zz_SeminalVesicles REVIEW BEFORE CLINICAL USE_current'],
    }
    return roi_names.get(names_id, [])

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Define the path to the top-level directory.
    if "Windows" == platform.system:
        top_dir = Path("C:/Users/pytorch/Desktop/ProKnow_Evaluation")
    else:
        top_dir = Path("~/data/Osairis/ProKnow_Evaluation").expanduser()

    # Define the patient data to be analysed.
    paths = get_paths(max_path=None, top_dir=top_dir, data_dir="Osairis*")

    # Run application for the selected data.
    app.run(paths, unsorted_dicom=True)
