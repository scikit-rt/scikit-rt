import timeit

from skrt.core import get_associated_image
from skrt.dose import remove_duplicate_doses
from skrt.patient import Patient


class ImportPatient(Patient):
    """
    Class that adds to the methods of skrt.patient.Patient.

    The additional functionality provided is aimed at simplifying
    analysis of data from the IMPORT study.

    Different from skrt.patient.Patient, the default is to load
    data as unsorted DICOM.
    """

    def __init__(self, path=None, exclude=None, unsorted_dicom=True,
            id_mappings=None, load_dose_sum=True, load_masks=True):
        """
        Create instance of ImportPatient class.

        **Parameters:**

        path : str/pathlib.Path, default=None
            Relative or absolute path to a directory containing patient data.
            If None, the path used is the path to the current working
            directory.

        exclude : list, default=None
            List of first-level sub-directories to disregard when
            loading patient data organised accoding to the VoxTox model
            (patient -> studies -> modalites).  If None, then set to
            ["logfiles"].  Ignored if <unsorted_dicom> set to True.

        unsorted_dicom : bool, default=True
            If True, don't assume that patient data are organised
            accoridng to the VoxTox model, and creates data hierarchy
            based on information read from DICOM files.

        id_mappings : dict, default=None
            By default, the patient identifier is set to be the name
            of the directory that contains the patient data.  The
            id_mappings dictionary allows mapping from the default
            name (dictionary key) to a different name (associated value).

        load_dose_sum : bool, default=True
            If True, load planned dose, summed over contributions,
            at initialisation time.  If False, leave loading until
            dose sum is requested via call to getter method (get_dose_sum()).

        load_masks : bool, default=True
            If True, create foreground masks for patient's plan and
            relapse CT images at initialisation time, using default
            parameters.  If False, leave mask creation until requested
            via vall to getter method (get_mask_plan(), get_mask_relapse()).
        """
        # Record start time
        tic = timeit.default_timer()

        # By default, data are treated as unsorted DICOM.
        unsorted_dicom = True if unsorted_dicom is None else unsorted_dicom

        # Perform most of the initialisation via the parent class.
        super().__init__(path, exclude, unsorted_dicom, id_mappings)

        # Perform loading of key data relevant to IMPORT study.
        self.key_data_loaded = False
        self.load_key_data(load_dose_sum=load_dose_sum)

        # Record end time, then store initialisation time.
        toc = timeit.default_timer()
        self._init_time = (toc - tic)

    def get_ct_plan(self):
        """Get CT scan for treatment planning"""
        return self.ct_plan

    def get_ct_relapse(self):
        """Get CT scan for relapse"""
        return self.ct_relapse

    def get_dose_sum(self, image=None):
        """
        Get planned dose, summed over contributions.

        **Parameter:**

        image : skrt.image.Image, default=None
            Image to be associated with the dose sum.  If None,
            the associated image will be the CT planning scan (self.ct_plan).
        """
        image = image or self.ct_plan
        if (self.dose_sum is None or
                not self.dose_sum.image.has_same_geometry(image)):
            Patient.get_dose_sum(self, set_image=image)
        return self.dose_sum

    def get_dose_summation_types(self, remove_duplicates=True):
        """
        Get list of summation types for doses associated with this patient.

        **Parameter:**
        remove_duplicates : bool, default=True
            If True, disregard any duplicate dose objects (same array
            shape and dose values).
        """
        if remove_duplicates:
            doses = remove_duplicate_doses(self.combined_objs('dose_types'))
        else:
            doses = self.combined_objs('dose_types')
        return [dose.get_dose_summation_type() for dose in doses]

    def get_mask_plan(self, threshold=-500, convex_hull=False,
            fill_holes=True, dxy=5, force=False):
        """
        Get foreground mask for CT planning scan.

        **Parameters:**

        threshold : int/float, default=-500
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.
    
        convex_hull : bool, default=False
            If False, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=True
            If False, fill holes in the slice foreground masks initially
            obtained.

        force : bool, default=False
            If True, force mask creation, overwriting any mask
            created previously.
        """
        if self.ct_plan and (force or not self.mask_plan):
            self.mask_plan = self.ct_plan.get_foreground_mask(
                    threshold, convex_hull, fill_holes, dxy)
        return self.mask_plan

    def get_mask_relapse(self, threshold=-500, convex_hull=False,
            fill_holes=True, dxy=5, force=False):
        """
        Get foreground mask for CT planning scan.

        **Parameters:**

        threshold : int/float, default=-500
            Intensity value above which pixels in a slice are assigned to
            regions for determination of foreground.
    
        convex_hull : bool, default=False
            If False, create mask from the convex hulls of the
            slice foreground masks initially obtained.

        fill_holes : bool, default=True
            If False, fill holes in the slice foreground masks initially
            obtained.

        force : bool, default=False
            If True, force mask creation, overwriting any mask
            created previously.
        """
        if self.ct_relapse and (force or not self.mask_relapse):
            self.mask_relapse = self.ct_relapse.get_foreground_mask(
                    threshold, convex_hull, fill_holes, dxy)
        return self.mask_relapse

    def get_ss_clinical(self, image=None):
        """
        Get clinical structure set for CT planning scan.

        **Parameter:**

        image : skrt.image.Image, default=None
            Image to be associated with the structure set.  If None,
            the associated image will be the CT planning scan (self.ct_plan).
        """
        self.ss_clinical.set_image(image or self.get_ct_plan())
        return self.ss_clinical

    def get_ss_plan(self, image=None):
        """
        Get research structure set for CT planning scan.

        **Parameter:**

        image : skrt.image.Image, default=None
            Image to be associated with the structure set.  If None,
            the associated image will be the CT planning scan (self.ct_plan).
        """
        self.ss_plan.set_image(image or self.get_ct_plan())
        return self.ss_plan

    def get_ss_relapse(self, image=None):
        """
        Get research structure set for CT relapse scan.

        **Parameter:**

        image : skrt.image.Image, default=None
            Image to be associated with the structure set.  If None,
            the associated image will be the CT relapse scan (self.ct_relapse).
        """
        self.ss_relapse.set_image(image or self.get_ct_relapse())
        return self.ss_relapse

    def load_key_data(self, load_dose_sum=True, load_masks=True, force=False):
        """
        Perform loading of key data relevant to IMPORT study.

        This method sets the following:

        self.ct_plan : skrt.image.Image
            CT scan used in treatment planning.

        self.ss_plan : skrt.structures.StructureSet
            Research structure set, containing control structures
            outlined on the planning scan.

        self.ss_clinical : skrt.structures.StructureSet
            Clinical structure set, containing structures used in
            treatment planning.

        self.dose_sum : skrt.dose.Dose
            Planned dose, summed over contributions.  Initialised
            to None if <load_dose_sum> is False.

        self.ct_relapase : skrt.image.Image
            CT scan recorded at the time of relapse.

        self.ss_relapase : skrt.structures.StructureSet
            Research structure set, containing control structures
            and recurrence outlined on the relapse scan.

        **Parameters:**

        load_dose_sum : bool, default=True
            If True, load planned dose, summed over contributions,
            at initialisation time.  If False, leave loading until
            dose sum is requested via call to getter method (get_dose_sum()).

        load_masks : bool, default=True
            If True, create foreground masks for patient's plan and
            relapse CT images at initialisation time.  If False, leave
            mask creation until requested via vall to getter method
            (get_mask_plan(), get_mask_relapse()).

        force : bool, default=False
            If True, load data from source, even if previously loaded.
        """
        # Return if key data already loaded, and not forcing reload.
        if self.key_data_loaded and not force:
            return

        if load_dose_sum:
            # Calculate planned dose, summed over contributions.
            # the result is assigned to self.dose_sum.
            Patient.get_dose_sum(self)
            self.ct_plan = self.dose_sum.image
        else:
            # Initialise dose sum to None.
            self.dose_sum = None

            # Take CT planning scan to be the largest image
            # referenced by a Dose object, with image size measured
            # in terms of number of voxels.
            self.ct_plan = get_associated_image(
                    self.combined_objs("dose_types"))

        # Obtain structure sets, sorted by number of ROIs.
        structure_sets = sorted(self.combined_objs("structure_set_types"),
                key=(lambda ss : len(ss.get_roi_names())))

        # Identify the control structures, which should usually be outlined
        # in the research structure sets.
        control_rois = [["Carina"], ["Spinal canal", "Spinal Canal",
            "Spinal cord"], ["Top of sternum", "Top of Sternum"]]
        additional_control_rois = [
                ["Spinal canal on the same slice as bottom of sternum"],
                ["Bottom of sternum"]]
    
        # Remove from the list of structure sets the cases where
        # the number of ROIs saved is less than the number of control
        # structures.
        # If one of the structure sets contains a single ROI,
        # labelled "recurrence", save this structure set in case it's
        # the only one with the recurrence outlined.
        ss_recurrence = None
        while len(structure_sets[0].get_roi_names()) == 1:
            if "recurrence" == structure_sets[0].get_roi_names()[0]:
                ss_recurrence = structure_sets[0]
            structure_sets.pop(0)   
        while len(structure_sets[0].get_roi_names()) < len(control_rois):
            structure_sets.pop(0)
        
        # The plan research structure set should be the one with fewest ROIs.
        self.ss_plan = structure_sets[0]
        assert len(self.ss_plan.get_roi_names()) == len(control_rois)
        for names in control_rois:
            assert [name in self.ss_plan.get_roi_names()
                    for name in names].count(True) == 1

        # Remove from the list of structure sets the cases where the number
        # of ROIs saved is equal to the number of control structures.
        while len(structure_sets[0].get_roi_names()) == len(control_rois):
            structure_sets.pop(0)
        
        # Among the remaining structure sets, the relapse research
        # structure set should include the control structures and
        # at least one recurrence outline.
        # Among the structure sets that don't include the control
        # structures, the clinical structure set should be the largest.
        self.ss_relapse = self.ss_clinical = None
        for structure_set in structure_sets:
            has_control_rois = has_additional_control_rois = True
            for names in control_rois:
                has_control_rois *= ([name in structure_set.get_roi_names()
                    for name in names].count(True) == 1)
            for names in additional_control_rois:
                has_additional_control_rois *= (
                        [name in structure_set.get_roi_names()
                            for name in names].count(True) == 1)

            # Deal with structure sets where additional control structures
            # are present.
            # In practice, in the cases, a plan structure set contains both
            # standard and additional control structures; a relapse structure set
            # contains additional control structures only.
            if has_additional_control_rois:
                if has_control_rois:
                    self.ss_plan = structure_set
                else:
                    self.ss_relapse = structure_set

            # Deal with standard cases: relapse structure set contains control
            # structures and recurrence outline; clinical structure set doesn't
            # contain control structures.
            elif has_control_rois:
                if ((not self.ss_relapse) or
                        (len(self.ss_relapse.get_roi_names())
                            > len(structure_set.get_roi_names()))):
                    self.ss_relapse = structure_set
            else:
                if ((not self.ss_clinical) or
                        (len(self.ss_clinical.get_roi_names())
                            > len(structure_set.get_roi_names()))):
                    self.ss_clinical = structure_set
                
        # If relapse structure set doesn't include recurrence outline,
        # substitute for it the structure set containing
        # recurrence outline only.
        if ss_recurrence and (not self.ss_relapse
                or ss_recurrence.get_roi_names()[0]
                not in self.ss_relapse.get_roi_names()):
            self.ss_relapse = ss_recurrence

        # Take CT image for relapse structure set to be the first image
        # that has at least 20 slices, and doesn't have the same number
        # of slices as the image associated with the dose.  (For the IMPORT
        # data, images for a patient that have the same number of slices
        # tend to be the same image, possible resampled.)
        for image in self.combined_objs("image_types"):
            if (len(image.dicom_paths) >= 20 and image.get_n_voxels()[2]
                    != self.ct_plan.get_n_voxels()[2]):
                self.ct_relapse = image
                break

        # Set structure-set images.
        self.ss_plan.set_image(self.ct_plan)
        self.ss_relapse.set_image(self.ct_relapse)
        self.ss_clinical.set_image(self.ct_plan)

        # Optionally create foreground masks.
        self.mask_plan = None
        self.mask_relapse = None
        if load_masks:
            self.get_mask_plan()
            self.get_mask_relapse()

        self.key_data_loaded = True
