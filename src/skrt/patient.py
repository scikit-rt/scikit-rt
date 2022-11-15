"""Classes related to Patients and Studies."""

import os
from pathlib import Path
import pydicom
import shutil
import time
import timeit

import pandas as pd

import skrt.core
from skrt.core import fullpath, get_data_indices, get_indexed_objs
from skrt.image import _axes, Image
from skrt.structures import StructureSet
from skrt.dose import Dose, Plan, remove_duplicate_doses, sum_doses

# Default mode for reading patient data.
skrt.core.Defaults({"unsorted_dicom": False})


class Study(skrt.core.Archive):
    """Class representing a single study; can contain images, dose fields,
    and structure sets."""

    def __init__(self, path=""):
        """Initialise a Study object from a given directory. Find any images,
        dose fields, and structure sets within this directory."""

        # Intialise as an Archive object
        skrt.core.Archive.__init__(self, path, allow_dirs=True)

        # Load data
        self.load_images()

        # Load structure sets
        self.load_from_subdir(
            dtype=StructureSet,
            subdir="RTSTRUCT",
            attr_name="structure_sets",
            load=False
        )

        # Load dose maps
        self.load_from_subdir(
            dtype=Dose,
            subdir="RTDOSE",
            attr_name="doses",
            load=False
        )

        # Load treament plans
        self.load_from_subdir(
            dtype=Plan,
            subdir="RTPLAN",
            attr_name="plans",
            load=False
        )

        if hasattr(self, 'plan_types'):
            for plans in self.plan_types.values():
                for plan in plans:
                    self.link_plan_to_doses(plan)

        if 'ct' in self.image_types and 'cthd' in self.image_types:
            if (len(self.image_types['ct']) == 1
                and len(self.image_types['cthd']) == 1):
                self.cthd_images[0].structure_sets = []
                for ss in self.ct_images[0].structure_sets:
                    self.cthd_images[0].add_structure_set(ss)
                    ss.set_image(self.cthd_images[0])
        #  self.load_dose_or_plan()

    def load_images(self):
        """Load images of various types by attempting to load from all
        files in subdirectories not named RTDOSE, RTSTRUCT, or RTPLAN."""

        # Find image subdirs by looking at all subdirs and ignoring those
        # with special names
        special_dirs = ["RTPLAN", "RTSTRUCT", "RTDOSE"]
        self.image_types = {}
        for file in self.files:

            subdir = os.path.basename(file.path)
            if subdir in special_dirs:
                continue
            self.image_types[subdir.lower()] = {}

            # Get Image objects in this subdir
            attr = f"{subdir.lower()}_images"
            images = self.create_objects(dtype=Image, subdir=subdir, load=False)
            for im in images:
                im.image_type = subdir.lower()

            # Store the images
            setattr(self, attr, images)
            self.image_types[subdir.lower()] = images

    def load_from_subdir(self, dtype, subdir, attr_name, **kwargs):
        """Create objects of type <dtype> from each directory in <subdir> and 
        attempt to assign the corresponding image. The created objects will
        be assign to self.<IMAGE_TYPE>_<attr_name>, where <image_type> is 
        inferred from the subdirectory from which the objects were created.

        **Parameters**:

        dtype : type
            Data type to use when creating objects.

        subdir : str
            Subdirectory in which to search for objects.

        attr_name : str
            Name of attribute to which loaded objects should be assigned.

        `**`kwargs :
            Keyword args to pass to object initialisation.
        """

        obj_dir = os.path.join(self.path, subdir)
        if not os.path.exists(obj_dir):
            return

        # Initialise dictionary of all objects created.
        obj_types_name = f'{attr_name[:-1]}_types'
        setattr(self, obj_types_name, {})
        obj_types = getattr(self, obj_types_name)

        # Search subdirectories (each corresponds to an image type)
        for im_type_dir in os.listdir(obj_dir):

            im_type = im_type_dir.lower()
            if subdir in ['RTDOSE', 'RTPLAN'] \
                    and im_type not in self.image_types:
                im_type = 'ct'
            subpath = f"{subdir}/{im_type_dir}"

            all_objs = []

            # Create archive object for each subdir (each corresponds to 
            # an image)
            archives = self.create_objects(dtype=skrt.core.Archive,
                                           subdir=subpath)
            for archive in archives:
                if os.path.isfile(archive.path):
                    archive.path = os.path.dirname(archive.path)

                # Create objects within this archive
                objs = archive.create_objects(
                    dtype=dtype, timestamp_only=False, **kwargs)

                # Look for an image matching the timestamp of this archive
                if im_type in self.image_types:
                    image_types = self.image_types[im_type]
                    if im_type != im_type_dir.lower():
                        obj_to_match = objs[0]
                    else:
                        obj_to_match = archive
                    image, ss = find_matching_object(obj_to_match, image_types)
                    if image is not None:
                        for obj in objs:
                            if hasattr(obj, "set_image"):
                                obj.set_image(image)

                    if ss is not None:
                        for obj in objs:
                            if hasattr(obj, "set_structure_set"):
                                obj.set_structure_set(ss)

                # Add to list of all objects for this image type
                # The preceding code results in object duplication.
                # This should be debugged, but for now just avoid
                # storing duplicates.
                for obj in objs:
                    if obj not in all_objs:
                        all_objs.append(obj)
                #all_objs.extend(objs)

            # Create attribute for objects of this type
            if len(all_objs):
                key = f"{im_type_dir.lower().replace('-', '_')}"
                setattr(self, f"{key}_{attr_name}", all_objs)
                obj_types[key] = all_objs

    def add_image(self, im, image_type="ct"):
        '''Add a new image of a given image type.'''

        # Ensure we have a list of this type of image
        im_name = f"{image_type.lower()}_images"
        if not hasattr(self, im_name):
            setattr(self, im_name, [])

        # Add image to the list
        images = getattr(self, im_name)
        if isinstance(im, Image):
            im_to_add = im
        else:
            im_to_add = Image(im)
        images.append(im)

        # Also add to dict of images of each type
        if image_type not in self.image_types:
            self.image_types[image_type.lower()] = [im]
        else:
            self.image_types[image_type.lower()].append(im)

        # Ensure image has a timestamp
        if not im.timestamp:
            im.timestamp = skrt.core.generate_timestamp()

        # Ensure corresponding structure list exists
        struct_subdir = f"RTSTRUCT/{image_type.upper()}"
        struct_name = f"{image_type.lower()}_structure_sets"
        if not hasattr(self, struct_name):
            setattr(self, struct_name, [])

        # Add the image's structure sets to structure set list
        structure_sets = getattr(self, struct_name)
        for structure_set in im.get_structure_sets():
            structure_sets.append(structure_set)
            if not structure_set.timestamp:
                structure_set.timestamp = skrt.core.generate_timestamp()

    def correct_dose_image_position(self, doses=[]):
        """Correct for image positions from CheckTomo being offset by one slice
        relative to image positions."""

        for dose in doses:
            dx, dy, dz = dose.voxel_size
            x0, y0, z0 = dose.image_position
            dose.image_position = (x0, y0, z0 + dz)
        return doses

    def get_machine_sublist(self, dtype="", machine="", ignore_case=True):
        """Get list of doses or treatment plans corresponding to a specific
        machine."""

        sublist = []
        if dtype.lower() in ["plan", "rtplan"]:
            objs = self.plans
        elif dtype.lower() in ["dose", "rtdose"]:
            objs = self.doses
        else:
            objs = []

        if ignore_case:
            for obj in objs:
                if objs.machine.lower() == machine.lower():
                    sublist.append(obj)
        else:
            for obj in objs:
                if objs.machine == machine:
                    sublist.append(object)
        return sublist

    def get_mvct_selection(self, mvct_dict={}, min_delta_hours=0.0):
        """Get a selection of MVCT images which were taken at least
        <min_delta_hours> apart. <mvct_dict> is a dict where the keys are
        patient IDs, and the paths are directory paths from which to load images
        for that patient."""

        # Find images meeting the time separation requirement
        if min_delta_hours > 0:
            mvct_images = get_time_separated_objects(self.mvct_images, min_delta_hours)
        else:
            mvct_images = self.mvct_images

        # Find images matching the directory requirement
        selected = []
        patient_id = self.get_patient_id()
        if patient_id in mvct_dict:

            # Get all valid directories for this patient
            valid_dirs = [skrt.core.fullpath(path) for path in mvct_dict[patient_id]]

            # Check for images matching that directory requirement
            for mvct in mvct_images:
                mvct_dir = os.path.dirname(mvct.files[-1].path)
                if skrt.core.fullpath(mvct_dir) in valid_dirs:
                    selected.append(mvct)

        # Otherwise, just return all images for this patient
        else:
            selection = mvct_images

        return selection

    def get_patient_id(self):
        patient_id = os.path.basename(os.path.dirname(self.path))
        return patient_id

    def get_plan_data(self, dtype="RtPlan", subdir="RTPLAN", exclude=[], images=[]):
        """Get list of RT dose or plan objects specified by dtype='RtDose' or
        'RtPlan' <dtype>, respectively) by searching within a given directory,
        <subdir> (or within the top level directory of this Study, if
        <subdir> is not provided).

        Subdirectories with names in <exclude> will be ignored.

        Each dose-like object will be matched by timestamp to one of the images
        in <images> (which should be a list of DatedStores), if provided."""

        doses = []

        # Get initial path to search
        if subdir:
            path1 = os.path.join(self.path, subdir)
        else:
            path1 = self.path

        # Look for subdirs up to two levels deep from initial dir
        subdirs = []
        if os.path.isdir(path1):

            # Search top level of dir
            path1_subdirs = os.listdir(path1)
            for item1 in path1_subdirs:

                if item1 in exclude:
                    continue
                path2 = os.path.join(path1, item1)
                n_sub_subdirs = 0

                # Search any directories in the top level dir
                if os.path.isdir(path2):
                    path2_subdirs = os.listdir(path2)
                    for item2 in path2_subdirs:
                        path3 = os.path.join(path2, item2)

                        # Search another level (subdir/item1/item2/*)
                        if os.path.isdir(path3):
                            n_sub_subdirs += 1
                            if subdir:
                                subdirs.append(os.path.join(subdir, item1, item2))
                            else:
                                subdirs.append(item1, item2)

                if not n_sub_subdirs:
                    if subdir:
                        subdirs = [os.path.join(subdir, item1)]
                    else:
                        subdirs = [item1]

                for subdir_item in subdirs:
                    doses.extend(
                        self.create_objects(dtype=dtype, subdir=subdir_item)
                    )

        # Assign dose-specific properties
        if dtype == "RtDose":
            new_doses = []
            for dose in doses:

                # Search for images with matching timestamp
                timestamp = os.path.basename(os.path.dirname(dose.path))
                if images:
                    try:
                        dose.date, dose.time = timestamp.split("_")
                        image = create_objects(images, dose)
                        dose.machine = image.machine
                    except BaseException:
                        image = images[-1]
                        dose.date = image.date
                        dose.time = image.time

                    dose.timestamp = f"{dose.date}_{dose.time}"
                    dose.image = image

                dose.couch_translation, dose.couch_rotation = get_couch_shift(dose.path)
                # WARNING!
                #     Couch translation third component (y) inverted with
                #     respect to CT image
                # WARNING!
                new_doses.append(dose)
            doses = new_doses

        doses.sort()
        return doses

    def get_plan_dose(self):

        plan_dose = None
        dose_dict = {}

        # Group doses by summation type
        for dose in self.doses:
            if dose.summationType not in dose_dict:
                dose_dict[dose.summationType] = []
            dose_dict[dose.summationType].append(dose)
        for st in dose_dict:
            dose_dict[st].sort()

        # 'PLAN' summation type: just take the newest entry
        if "PLAN" in dose_dict:
            plan_dose = dose_dict["PLAN"][-1]
            plan_dose.imageStack = plan_dose.getImageStack()

        else:

            # Get fraction group and beam sequence
            if self.plans:
                n_frac_group = self.plans[-1].nFractionGroup
                n_beam_seq = self.plans[-1].nBeamSequence
            else:
                n_frac_group = None
                n_beam_seq = None

            # Sum over fractions
            if "FRACTION" in dose_dict:
                if len(dose_dict["FRACTION"]) == n_frac_group:

                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = dose_dict["FRACTION"][0]

                    # Sum fractions
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, "FRACTION")

            # Sum over beams
            elif "BEAM" in sum_type:
                if len(dose_dict["BEAM"]) == n_beam_seq:

                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = dose_dict["BEAM"][0]

                    # Sum beams
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, "BEAM")

        return plan_dose

    def get_description(self):
        """Load a study description."""

        # Find an object from which to extract description
        obj = None
        if self.studies:
            obj = getattr(self, f"{self.image_types[0].lower()}_images")[-1]
        description = ""
        if obj:
            if obj.files:
                image_path = obj.files[-1].path
                ds = pydicom.dcmread(fp=image_path, force=True)
                if hasattr(ds, "StudyDescription"):
                    description = ds.StudyDescription

        return description

    def sum_dose_plans(self, dose_dict={}, sum_type=""):
        """Sum over doses using a given summation type."""

        plan_dose = None
        if sum_type in dose_dict:
            dose = dose_dict[sum_type].pop()
            plan_dose = RtDose()
            plan_dose.machine = dose.machine
            plan_dose.path = dose.path
            plan_dose.subdir = dose.subdir
            plan_dose.date = dose.date
            plan_dose.time = dose.time
            plan_dose.timestamp = dose.timestamp
            plan_dose.summationType = "PLAN"
            plan_dose.imagePosition = dose.imagePosition
            plan_dose.reverse = dose.reverse
            plan_dose.voxelSize = dose.voxelSize
            plan_dose.transform_ijk_to_xyz = dose.transform_ijk_to_xyz
            plan_dose.imageStack = dose.getImageStack()
            for dose in dose_dict[sum_type]:
                plan_dose.imageStack += dose.getImageStack()

        return plan_dose

    def link_plan_to_doses(self, plan):
        '''
        Link plan to doses derived from it.

        **Parameter:**

        plan : skrt.dose.Plan
            Plan object for which dose associations are to be determined.
        '''

        plan_uid = pydicom.dcmread(plan.path, force=True).SOPInstanceUID
        doses = []
        if hasattr(self, 'dose_types'):
            for value in self.dose_types.values():
                doses.extend(value)
        for dose in doses:
            dose_ds = pydicom.dcmread(dose.path, force=True)
            if hasattr(dose_ds, 'ReferencedRTPlanSequence'):
                for referenced_plan in dose_ds.ReferencedRTPlanSequence:
                    if plan_uid == referenced_plan.ReferencedSOPInstanceUID:
                        dose.set_plan(plan)

    def save_images_as_nifti(self, outdir='.', image_types=None, times=None,
            verbose=True, image_size=None, voxel_size=None, fill_value=None,
            bands=None, require_structure_set=None):
        '''
        Save study's image data as nifti files.

        **Parameters:**

        outdir - str, default='.'
            Directory to which nifti files will be saved.

        image_types - list/str/None, default=None
            Images types to be saved: None to save all, otherwise a list
            of image types to save, or a string specifying a single image
            type to save.

        times : dict, default=None
            Dictionary where the keys are image types and the values are
            lists of timestamp indices for the images to be saved,
            0 being the earliest and -1 being the most recent.  If set to
            None, all images are saved.

        verbose - bool, default=True
           Flag indicating whether to report progress.

        image_size - tuple, default=None
            Image dimensions (dx, dy, dz) in voxels for image resizing
            prior to writing.  If None, original image size is kept.

        voxel_size - tuple, default=None
            Voxel dimensions (dx, dy, dz) in mm for image resizing
            prior to writing.  If None, original voxel size is kept.

        fill_value - int/float, default=None
            Value used when extrapolating image outside data area.
            If None, use mininum value inside data area.

        bands - dict, default=None
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

        require_structure_set - list, default=None
            List of image types for which data are to be written only
            if there is an associated structure set.
        '''

        # Set defaults.
        if bands is None:
            bands = {}
        if require_structure_set is None:
            require_structure_set = []

        # Obtain full path to output directory.
        outdir = skrt.core.fullpath(outdir)

        # Obtain set of image types to be saved.
        save_types = set(self.image_types)
        if image_types is not None:
            if isinstance(image_types, str):
                image_types = {image_types}
            save_types = save_types.intersection(set(image_types))
        if verbose:
            if save_types:
                print(f"Saving image data to '{outdir}':")
            else:
                print(f"No data to save for output directory '{outdir}'")

        # Loop over image types to be saved.
        for save_type in save_types:
            images = sorted(self.image_types[save_type])

            # Only consider images with requested indices.
            if times:
                if save_type in times and times[save_type]:
                    images = [images[idx] for idx in times[save_type]]

            # Loop over images.
            for idx, image in enumerate(images):

                # Skip image if associated structure set required but missing.
                if (save_type in require_structure_set and
                        not image.structure_sets):
                    continue

                # Identify imaging machine.
                suffix = (image.get_machine() if image.get_machine()
                        else 'Unknown')

                # Clone image object for writing.
                # If data manipulation is required, this will be
                # applied to the clone rather than to the original.
                out_image = Image(image)

                # Resize to required image size and voxel size.
                if image_size is not None or voxel_size is not None:
                    im_fill_value = (fill_value if fill_value is not None
                            else out_image.get_min())
                    out_image.resize(image_size=image_size,
                            voxel_size=voxel_size, fill_value=im_fill_value)

                # Apply banding.
                if save_type in bands:
                    if bands[save_type]:
                        out_image.apply_bandings(bands[save_type])

                # Define output path and write image.
                outname = (f'{outdir}/{save_type.upper()}'
                           f'_{image.timestamp}_{idx+1:03}'
                           f'_{suffix}.nii.gz')
                out_image.write(outname=outname, verbose=verbose)

    def save_structure_sets_as_nifti(self, outdir='.',
            image_types=None, times=None, files=None,
            verbose=True, image_size=None, voxel_size=None, roi_names=None,
            force_roi_nifti=False, bilateral_names=None):

        '''
        Save study's structure-set data as nifti files.

        **Parameters:**

        outdir - str, default='.'
            Directory to which nifti files will be saved.

        image_types - list/str/None, default=None
            Images types for which structure sets are to be saved: None to
            save for all, otherwise a list of image types, or a string
            specifying a single image type.

        times : dict, default=None
            Dictionary where the keys are image types and the values are
            lists of timestamp indices for the images for which structure
            sets are to be saved, 0 being the earliest and -1 being the
            most recent.  If set to None, structure sets are saved for
            all images.

        files : dict, default=None
            Dictionary where the keys are image types and the values are
            lists of file indices for structure sets to be saved for
            a given image, 0 being the earliest and -1 being the most recent.
            If set to None, all of an image's structure sets are saved.

        verbose - bool, default=True
           Flag indicating whether to report progress.

        image_size - tuple, default=None
            Image dimensions (dx, dy, dz) in voxels for image resizing
            prior to writing.  If None, original image size is kept.

        voxel_size - tuple, default=None
            Voxel dimensions (dx, dy, dz) in mm for image resizing
            prior to writing.  If None, original voxel size is kept.

        roi_names : dict, default=None
            Dictionary of names for renaming ROIs, where the keys are new 
            names and values are lists of possible names of ROIs that should
            be assigned the new name. These names can also contain wildcards
            with the '*' symbol.

        force_roi_nifti : bool, default=False
            If True, force writing of dummy NIfTI file for all named ROIs
            not found in structure set.  If False, named ROIs not found in
            structure set are disregarded.

        bilateral_names : list, default=None
            List of names of ROIs that should be split into
            left and right parts.
        '''

        # Obtain full path to output directory.
        outdir = skrt.core.fullpath(outdir)

        # Initialise list or names for bilateral ROIs
        if bilateral_names is None:
            bilateral_names = []

        # Obtain set of image types for which structure sets are to be saved.
        save_types = set(self.image_types)
        if image_types is not None:
            if isinstance(image_types, str):
                image_types = {image_types}
            save_types = save_types.intersection(set(image_types))
        if verbose:
            if save_types:
                print(f"Saving structure_set data to '{outdir}':")
            else:
                print(f"No data to save for output directory '{outdir}'")

        # Loop over image types for which structure sets are to be saved.
        for save_type in save_types:
            if save_type not in self.image_types:
                continue
            images = sorted(self.image_types[save_type])

            # Only consider images with requested indices.
            if times:
                if save_type in times and times[save_type]:
                    images = [images[idx] for idx in times[save_type]]

            # Loop over images.
            for image in images:
                structure_sets = sorted(image.structure_sets)

                # Only consider structure sets with requested indices.
                if files and structure_sets:
                    if save_type in files and files[save_type]:
                        structure_sets = get_indexed_objs(
                                structure_sets, files[save_type])

                # Loop over structure sets
                idx = 0
                for structure_set in structure_sets:
                    # Filter ROIs
                    ss = structure_set.filtered_copy(names=roi_names,
                            keep_renamed_only=True)

                    # Resize associated image if needed.
                    if ss.get_roi_names() or force_roi_nifti:
                        if image_size is not None or voxel_size is not None:
                            im = Image(ss.image)
                            im.resize(image_size=image_size,
                                    voxel_size=voxel_size)
                            ss.set_image(im)

                    # Create list of ROIs to be saved.
                    if roi_names:
                        save_names = list(roi_names)
                    else:
                        save_names = list(ss.get_roi_names())

                    # Split ROI into left and right parts if needed.
                    for roi_name in bilateral_names:
                        save_names.remove(roi_name)
                        save_names.append(f'{roi_name}_left')
                        save_names.append(f'{roi_name}_right')
                        if roi_name in ss.get_roi_names():
                            ss_tmp = ss.get_roi(roi_name).split_in_two()
                            for roi in ss_tmp.get_rois():
                                ss.add(roi)

                    # Loop over ROIs.
                    idx_add = 0
                    for roi_name in save_names:
                        # Define output path
                        outname = (f'RTSTRUCT_{save_type.upper()}'
                                   f'_{image.timestamp}_{idx+1:03}'
                                   f'_{roi_name}.nii.gz')
                        if roi_name in ss.get_roi_names():
                            # Save ROI mask.
                            roi = ss.get_roi(roi_name)
                            roi.write(outname=outname, outdir=outdir,
                                    ext='nii.gz', verbose=verbose)
                            idx_add = 1
                        elif force_roi_nifti:
                            # Save dummy mask for ROI not found
                            # in structure set.
                            im = Image(ss.image)
                            shape = im.get_data().shape
                            im.data = np.zeros(shape)
                            outname = f'{outdir}/{outname}'
                            ss.image.write(outname=outname, verbose=verbose)
                            idx_add = 1
                    idx += idx_add

    def copy_dicom(self, outdir=".",
            images_to_copy=None, structure_sets_to_copy=None,
            doses_to_copy=None, plans_to_copy=None, overwrite=True,
            sort=True):
        """
        Copy study dicom data.

        **Parameters:**

        overwrite - bool, default=True
            If False, skip images with pre-existing output directories.
            If True, delete pre-existing output directories.

        outdir - str, default='.'
            Top-level directory to which nifti files for InnerEye will
            be written for study.  Each output image will be in a separate
            sub-directory, along with a file per associated ROI.

        images_to_copy : list/str/dict, default=None
            String specifiying image type for which all images are
            to be copied; list of strings specifying image types
            for which all images are to be copied; dictionary where
            the keys are image types and the values are lists of
            timestamp indices for the images to be copied, 0 being
            the earliest and -1 being the most recent.  If set to
            None, all images are copied.

        structure_sets_to_copy : list/str/dict, default=None
            String specifiying structure-set type for which all
            structure sets are to be copied; list of strings
            specifying structure-set types for which all structure
            sets are to be copied; dictionary where the keys are
            structure-set types and the values are lists of
            timestamp indices for the structure sets to be copied,
            0 being the earliest and -1 being the most recent.  If set to
            None, all structure sets are copied.
        """
        # Ensure that study output directory exists.
        study_dir = skrt.core.make_dir(outdir, overwrite=overwrite)

        # Obtain dictionary associating indices to images.
        image_indices = get_data_indices(images_to_copy, self.image_types)

        # Identify non-image data.
        non_image_data = [
                ("RTSTRUCT", "structure_sets",
                    get_data_indices(structure_sets_to_copy,
                    getattr(self, "structure_set_types", []))),
                ("RTDOSE", "doses", get_data_indices(doses_to_copy,
                    getattr(self, "dose_types", []))),
                ("RTPLAN", "plans", get_data_indices(plans_to_copy,
                    getattr(self, "plan_types", []))),
                ]

        # Loop over image types.
        # Overwriting taken into account at study level,
        # so don't overwrite at sub-study level.
        for image_type in image_indices:

            # Loop over images of current type.
            for idx1, im in enumerate(self.image_types[image_type]):

                # Check that image can be loaded.
                # If not, give warning and skip.
                try:
                    im.load()
                except:
                    im.data = None

                if im.data is None:
                    print(f"Problems loading: {image_type}")
                    print(getattr(im, "dicom_paths", []))
                    continue
                
                # Copy image.
                im.copy_dicom_files(image_type, idx1, image_indices,
                        study_dir / image_type.upper()
                        / f"{im.timestamp}_{idx1+1:03}",
                        overwrite=False, sort=sort)

                # Copy non-image data that matches an image type.
                for modality, reference, indices in non_image_data:
                    if image_type in indices:
                        for idx2, obj in enumerate(getattr(im, reference)):
                            obj.copy_dicom_files(image_type, idx2, indices,
                                    study_dir / modality / image_type.upper()
                                    / f"{im.timestamp}_{idx1+1:03}",
                                    overwrite=False, sort=sort)

        # Copy non-image data not matching an image type.
        for modality, reference, indices in non_image_data:
            for data_type in indices:
                if data_type not in image_indices:
                    for idx3, obj in enumerate(
                            getattr(self, f"{data_type}_{reference}")):
                        obj.copy_dicom_files(data_type, idx3, indices,
                                study_dir / modality / data_type.upper(),
                                overwrite=False, sort=sort)


class Patient(skrt.core.PathData):
    """
    Object associated with a top-level directory whose name corresponds to
    a patient ID, and whose subdirectories contain studies.
    """

    def __init__(self, path=None, exclude=None, unsorted_dicom=None,
            id_mappings=None):
        '''
        Create instance of Patient class.

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

        unsorted_dicom : bool, default=False
            If True, don't assume that patient data are organised
            accoridng to the VoxTox model, and create data hierarchy
            based on information read from DICOM files.

        id_mappings : dict, default=None
            By default, the patient identifier is set to be the name
            of the directory that contains the patient data.  The
            id_mappings dictionary allows mapping from the default
            name (dictionary key) to a different name (associated value).
        '''

        # Initialise parameters.
        path = path or ""
        exclude = exclude or ["logfiles"]
        unsorted_dicom = unsorted_dicom or skrt.core.Defaults().unsorted_dicom
        id_mappings = id_mappings or {}

        # Record start time
        tic = timeit.default_timer()

        # Set path and patient ID
        if path is None:
            path = os.getcwd()
        self.path = skrt.core.fullpath(str(path))
        patient_id = os.path.basename(self.path)
        self.id = id_mappings.get(patient_id, patient_id)

        # Find studies
        if unsorted_dicom:
            self.sort_dicom()
        else:
            self.studies = self.create_objects(dtype=Study)
            if not self.studies:
                if os.path.isdir(self.path):
                    if os.access(self.path, os.R_OK):
                        subdirs = sorted(os.listdir(self.path))
                        for subdir in subdirs:
                            if subdir not in exclude:
                                self.studies.extend(
                                        self.create_objects(dtype=Study,
                                            subdir=subdir)
                                        )
        
        # Add to data objects a reference to the patient object.
        self.link_study_data_to_patient()

        # Initialise dose sum (outside of any study).
        self.dose_sum = None
        self._dose_sum_time = None

        # Record end time, then store initialisation time.
        toc = timeit.default_timer()
        self._init_time = (toc - tic)

    def __gt__(self, other):
        '''
        Define whether <self> is greater than <other>.

        The comparison is based on id.
        '''
        return self.id > other.id

    def __ge__(self, other):
        '''
        Define whether <self> is greater than, or equal to, <other>.

        The comparison is based on id.
        '''
        return (self > other) or (self == other)

    def sort_dicom(self):
        '''
        Create patient tree of data objects from unsorted DICOM files.

        This method creates the hierarchy patient -> studies -> modalities,
        sorting through all files in the directory pointed to be the
        patient object's path attribute, and in all sub-directories.
        '''

        # Mapping between non-imaging DICOM modalities and
        # terms used in Patient attributes.
        data_types = {
                "rtplan": "plan",
                "rtstruct": "structure_set",
                "rtdose": "dose",
                }

        # Mapping between terms used in Patient attributes and
        # Scikit-rt classes to which they refer.
        data_classes = {
                "dose": Dose,
                "image": Image,
                "plan": Plan,
                "structure_set": StructureSet,
                }

        # Modalities not currently handled.
        unhandled_modalities = ["pr", "rtrecord", "sr"]

        # Loop through files in the directory pointed to path attribute,
        # and in sub-directories.  Use each file to try to instantiate
        # a skrt.core.DicomFile object.  Use instantiated objects to
        # identify studies, then group objects by study, modality,
        # series.
        self.studies = []
        patient_path = Path(self.path)
        file_paths = sorted(patient_path.glob("**/*"))
        for file_path in file_paths:
            dcm = skrt.core.DicomFile(file_path)
            if (dcm.ds and (dcm.study_instance_uid is not None)
                and (dcm.modality not in unhandled_modalities)):
                # Add to patient's list of Study objects.
                matched_attributes = dcm.get_matched_attributes(self.studies,
                        "study_instance_uid")
                if matched_attributes:
                    study = matched_attributes[0]
                else:
                    study = dcm.get_object(Study)
                    self.studies.append(study)

                # Add to study objects dictionary of types for each modality.
                # All modalities not used as keys in the data_types dictionary
                # are mapped to modality "image".
                dstring = data_types.get(dcm.modality, "image")
                dtypes = getattr(study, f"{dstring}_types")
                if not dcm.modality in dtypes:
                    dtypes[dcm.modality] = []

                if "image" == dstring:
                    # For "image" modality, group DicomFile objects
                    # by series instance uid.
                    matched_attributes = dcm.get_matched_attributes(
                            dtypes[dcm.modality],
                            ["series_instance_uid"]
                            )
                    if matched_attributes:
                        matched_attributes[0].dicom_paths.append(dcm.path)
                    else:
                        dcm.dicom_paths = [dcm.path]
                        dtypes[dcm.modality].append(dcm)
                else:
                    # For modalities other than "image",
                    # add DicomFile objects to the relevant list
                    # of the types dictionary.
                    dtypes[dcm.modality].append(dcm)

        # First loop over studies: create Scikit-rt objects;
        # group objects by modality.
        for study in self.studies:
            for dstring, dclass in data_classes.items():
                dtypes = getattr(study, f"{dstring}_types")
                for modality in dtypes:
                    # Define datastore for each modality.
                    datastore = f"{modality}_{dstring}s"
                    setattr(study, datastore, [])
                    for idx, dcm in enumerate(dtypes[modality]):
                        # Instantiate Scikit-rt object.
                        obj = dcm.get_object(dclass)
                        if "image" == dstring:
                            # Transfer list of source files
                            # from DicomFile object to Image object.
                            obj.dicom_paths = sorted(list(dcm.dicom_paths),
                                    key=skrt.core.alphanumeric)
                            # Subsitute Image object for DicomFile object
                            # in dictionary of image types.
                            dtypes[modality][idx] = obj
                        getattr(study, datastore).append(obj)

        # Second loop over studies: link structure sets, plans, doses.
        for study in self.studies:

            # Link structure sets and plans.
            if study.plan_types and study.structure_set_types:
                for plan in study.rtplan_plans:
                    ss = skrt.core.get_referenced_object(
                            plan, study.rtstruct_structure_sets,
                            "referenced_structure_set_sop_instance_uid")
                    if ss:
                        plan.set_structure_set(ss)

            # Link doses and plans.
            if study.plan_types and study.dose_types:
                for dose in study.rtdose_doses:
                    plan = skrt.core.get_referenced_object(
                            dose, study.rtplan_plans,
                            "referenced_plan_sop_instance_uid")
                    if plan:
                        dose.set_plan(plan)

        # Third loop over studies: link images and non-imaging data;
        # group non-imaging data by modality of linked image.
        for study in self.studies:

            # Link images and non-imaging data.
            for dcm_modality, dstring in data_types.items():
                dtypes = getattr(study, f"{dstring}_types")
                dcm_datastore = f"{dcm_modality}_{dstring}s"
                for obj in getattr(study, dcm_datastore, []):
                    image = skrt.core.get_referenced_image(
                            obj, study.image_types)
                    obj.set_image(image)

            # Reinforce linking of images, plans and structure sets.
            if study.plan_types and study.structure_set_types:
                for plan in study.rtplan_plans:
                    if not plan.image:
                        # If plan and image not directly linked,
                        # try to link via structure set.
                        ss = skrt.core.get_referenced_object(
                                plan, study.rtstruct_structure_sets,
                                "referenced_structure_set_sop_instance_uid")

                        image = ss.image if ss is not None else None
                        plan.set_image(image)

                for ss in study.rtstruct_structure_sets:
                    if not ss.image:
                        # If structure set and image not directly linked,
                        # try to link via plan.
                        for plan in ss.plans:
                            if plan.image:
                                ss.set_image(plan.image)
                                break

            # Group non-imaging data by both their own modality and
            # the modality of linked image.  Where there is no
            # linked image, group non-imaging data by their own
            # modality only.
            for dcm_modality, dstring in data_types.items():
                dtypes = getattr(study, f"{dstring}_types")
                dcm_datastore = f"{dcm_modality}_{dstring}s"
                orphan_objs = []
                for obj in getattr(study, dcm_datastore, []):
                    if obj.image:
                        datastore = f"{obj.image.modality}_{dstring}s"
                        if not obj.image.modality in dtypes:
                            dtypes[obj.image.modality] = []
                            setattr(study, datastore, [])
                        dtypes[obj.image.modality].append(obj)
                        getattr(study, datastore).append(obj)
                    else:
                        orphan_objs.append(obj)
                if orphan_objs:
                    setattr(study, dcm_datastore, orphan_objs)
                    dtypes[dcm_modality] = orphan_objs
                else:
                    # Discard datastores not needed
                    # if all non-imaging data has linked image.
                    if hasattr(study, dcm_datastore):
                        delattr(study, dcm_datastore)
                    if dcm_modality in dtypes:
                        del dtypes[dcm_modality]

    def link_study_data_to_patient(self):
        '''Add to data objects a reference to the associated patient object.'''

        categories = set()

        for study in self.studies:
            for attribute in dir(study):
                if "types" in attribute:
                    categories.add(attribute)
            study.patient = self

        for category in categories:
            for obj in self.combined_objs(category):
                obj.patient = self

    def add_study(self, subdir='', timestamp=None, images=None, 
                  image_type="ct"):
        '''Add a new study.'''
    
        # Create empty Study object
        s = Study("")
        s.subdir = subdir
        s.timestamp = timestamp if timestamp else skrt.core.generate_timestamp()

        # Add images
        if images:
            for im in images:
                s.add_image(im, image_type.lower())

        # Add to studies list
        self.studies.append(s)

    def combined_files(self, attr, min_date=None, max_date=None):
        """Get list of all files of a given data type <attr> associated with
        this patient, within a given date range if specified."""

        files = []
        for study in self.studies:
            objs = getattr(study, attr)
            for obj in objs:
                for file in obj.files:
                    if file.in_date_interval(min_date, max_date):
                        files.append(file)
        files.sort()
        return files

    def combined_files_by_dir(self, attr, min_date=None, max_date=None):
        """Get dict of all files of a given data type <attr> associated with
        this patient, within a given date range if specified. The dict keys
        will be the directories that the files are in."""

        files = {}
        for study in self.studies:
            objs = getattr(study, attr)
            for object in objs:
                for file in object.files:
                    if file.in_date_interval(min_date, max_date):
                        folder = os.path.dirname(skrt.core.fullpath(file.path))
                        if folder not in files:
                            files[folder] = []
                        files[folder].append(file)

        for folder in files:
            files[folder].sort()

        return files

    def combined_objs(self, attr, subdir=None):
        '''
        Get list of objects across all studies associated with this patient.

        Optionally restrict to studies within a given sub-directory.

        **Parameters:**

        attr : str
            Attribute name, identifying list of objects associated with a study.
            If attr ends with '_types', then objects of all listed types
            are retrieved.

        subdir : str, default=None
            Subdirectory grouping studies.  If specified, only studies in this
            subdirectory are considered.
        '''
        all_objs = []
        studies = self.get_subdir_studies(subdir) if subdir else self.studies
        if attr.endswith('_types'):
            dtype = attr.replace('_types', 's')
            for study in studies:
                obj_types = getattr(study, attr, [])
                for obj_type in sorted(obj_types):
                    all_objs.extend(getattr(study, f'{obj_type}_{dtype}'))
        else:
            for study in studies:
                objs = getattr(study, attr, None)
                if objs:
                    all_objs.extend(objs)
        all_objs.sort()
        return all_objs

    def combined_types(self, cls, subdir=None):
        '''
        Get list of object types across all studies, for a given class.

        Optionally restrict to studies within a given sub-directory.

        **Parameters:**

        cls : str
            Class name (case insensitive), for which types of object are
            to be retrieved.

        subdir : str, default=None
            Subdirectory grouping studies.  If specified, only studies in this
            subdirectory are considered.
        '''
        all_types = []
        studies = self.get_subdir_studies(subdir) if subdir else self.studies
        for study in studies:
            obj_types = getattr(study, f'{cls.lower()}_types', [])
            for obj_type in sorted(obj_types):
                if not obj_type in all_types:
                    all_types.append(obj_type)

        all_types.sort()
        return all_types

    def load_demographics(self):
        """Load a patient's birth date, age, and sex."""

        info = {"BirthDate": None, "Age": None, "Sex": None}

        # Find an object from which to extract the info
        obj = None
        if self.studies:
            # Preferentially choose first CT scan.
            image_types = list(self.studies[0].image_types.keys())
            if image_types:
                image_type = 'ct' if 'ct' in image_types else image_types[0]
                obj = getattr(
                    self.studies[0],
                    f"{image_type.lower()}_images")[0]

        # Read demographic info from the object
        if obj and obj.files:
            ds = pydicom.dcmread(fp=obj.files[-1].path, force=True)
            for key in info:
                for prefix in ["Patient", "Patients"]:
                    attr = f"{prefix}{key[0].upper()}{key[1:]}"
                    if hasattr(ds, attr):
                        info[key] = getattr(ds, attr)
                        break

        # Ensure sex is uppercase and single character
        if info["Sex"]:
            info["Sex"] = info["Sex"][0].upper()

        # Obtain birth date as pandas Timestamp.
        if info["BirthDate"]:
            try:
                info["BirthDate"] = pd.Timestamp(info["BirthDate"])
            except ValueError:
                info["BirthDate"] = None

        # Obtain age as an integer in years.
        if info["Age"]:
            info["Age"] = int(info["Age"].strip('0').strip('Y'))

        # Store data
        self.age = info["Age"]
        self.sex = info["Sex"]
        self.birth_date = info["BirthDate"]

    def get_age(self):

        if not hasattr(self, 'age'):
            self.load_demographics()
        return self.age

    def get_sex(self):

        if not hasattr(self, 'sex'):
            self.load_demographics()
        return self.sex

    def get_birth_date(self):

        if not hasattr(self, 'birth_date'):
            self.load_demographics()
        return self.birth_date

    def get_groupings(self, collections=None):
        """
        Obtain dictionary of information on patient groupings.

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in patient data paths.  If None, an empty dictionary is used.
        """
        collections = collections or {}
        info = {}
        for collection, collection_types in collections.items():
            info[collection] = None
            for collection_type in collection_types:
                if collection_type in self.path:
                    info[collection] = collection_type

        return info

    def get_info(self, collections=None, data_labels=None, image_types=None,
            dose_types=None, plan_image_type=None, treatment_image_type=None,
            min_delta=4, unit='hour', df=False):
        '''
        Retrieve patient summary information.

        **Parameters:**

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in data paths.  If None, an empty dictionary is used.

        data_labels : list, default=None
            List of strings specifying labels of data associated with
            a study.  If None, the list used is:
            ['image', 'structure_set', 'dose', 'plan'].

        image_types : list, default=None
            List of strings indicating types of image for which information
            is to be retrieved, for example ['ct', 'mr'].  If None,
            information is retrieved for all image types in the patient
            dataset.  Ignored if 'image' isn't included in data_labels.

        plan_image_type : str, default=None
            String identifying type of image recorded for treatment planning.

        treatment_image_type : str, default=None
            String identifying type of image recorded at treatment time.

        min_delta : int/pandas.Timedelta, default=4
            Minimum time interval required between image objects.
            If an integer, the unit must be specified.

        unit : str, default='hour'
            Unit of min_delta if the latter is specified as an integer;
            ignored otherwise.  Valid units are any accepted by
            pandas.Timedelta:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html

        df : bool, default=False
            If False, return summary information as a dictionary.
            If True, return summary information as a pandas dataframe.
        '''
        # Ensure that dictionary is defined for patient groupings.
        collections = collections or {}

        # Ensure that list is defined for data labels.
        if data_labels is None:
            data_labels = ['image', 'structure_set', 'dose', 'plan']

        # Ensure that list is defined for image_types.
        if image_types is None:
            image_types = self.combined_types('image')

        # Ensure that list is defined for dose_types.
        if dose_types is None:
            dose_types = self.combined_types('dose')

        # Create set combining image_types and dose_types.
        target_types = set(image_types).union(set(dose_types))

        # Create dictionary for information to be returned.
        info = {}

        # Store basic patient information.
        info['id'] = self.id
        info['birth_date'] = self.get_birth_date()
        info['age'] = self.get_age()
        info['sex'] = self.get_sex()

        # Store information on patient groupings.
        info = {**info, **self.get_groupings(collections)}

        # Store plan information.
        # Information is taken from the earliest plan file,
        # in the earliest study containing a plan file.
        info['plan_name'] = None
        info['plan_description'] = None
        info['plan_prescription_description'] = None
        info['plan_fraction'] = None
        info['plan_target_dose'] = None
        info['plan_targets'] = None
        info['plan_organs_at_risk'] = None
        for study in self.studies:
            if hasattr(study, 'plan_types') and study.plan_types:
                plan_type = sorted(list(study.plan_types.keys()))[0]
                plan = study.plan_types[plan_type][0]
                plan.load()
                info['plan_name'] = plan.name
                info['plan_description'] = plan.description
                info['plan_prescription_description'] = (
                        plan.prescription_description)
                info['plan_fraction'] = plan.n_fraction
                info['plan_target_dose'] = plan.target_dose
                if skrt.core.is_list(plan.targets):
                    info['plan_targets'] = len(plan.targets)
                if skrt.core.is_list(plan.organs_at_risk):
                    info['plan_organs_at_risk'] = len(plan.organs_at_risk)
                break

        # Store information about all data types, across all studies.
        info['n_study'] = len(self.studies)
        for study in self.studies:
            for data_label in data_labels:
                type_label = f'{data_label}_types'
                data_types = getattr(study, type_label, None)
                if data_types:
                    # For each data type, store information on:
                    # - number of files;
                    # - combined size (in bytes) of all files;
                    # For image data, also store information on:
                    # - number of image objects.
                    # For dose data, also store information on:
                    # - maximum dose.
                    for data_type, objs in sorted(data_types.items()):
                        if data_type not in target_types:
                            continue
                        file_label = f'{data_label}_{data_type}_file'
                        size_label = f'{data_label}_{data_type}_size'
                        obj_label = f'{data_label}_{data_type}_obj'
                        dose_label = f'{data_label}_{data_type}_max'
                        if not file_label in info:
                            info[file_label] = 0
                        if not size_label in info:
                            info[size_label] = 0
                        if 'image' == data_label:
                            if obj_label not in info:
                                info[obj_label] = 0
                            info[obj_label] += len(objs)
                        if 'dose' == data_label:
                            if dose_label not in info:
                                info[dose_label] = 0
                            for obj in objs:
                               info[dose_label] = max(
                                       info[dose_label], obj.get_max())
                        for obj in objs:
                            info[file_label] += obj.get_n_file()
                            info[size_label] += obj.get_file_size()

        # Create lists of Image objects for planning and for treatment,
        # and list of StructureSet objects for planning.
        if plan_image_type:
            plan_images = self.combined_objs(f'{plan_image_type}_images')
            plan_structure_sets = self.combined_objs(
                    f'{plan_image_type}_structure_sets')
        else:
            plan_images = None
            plan_structure_sets = None
        if treatment_image_type:
            treatment_images = self.combined_objs(
                    f'{treatment_image_type}_images')
        else:
            treatment_images = None

        # Store number of ROIs outlined on planning scan.
        if plan_structure_sets:
            info['plan_image_rois'] = len(plan_structure_sets[0].get_rois())
        else:
            info['plan_image_rois'] = None

        # Store time of image used for plan creation.
        if plan_images:
            info['plan_image_time'] = plan_images[0].get_pandas_timestamp()
            info['plan_image_day'] = info['plan_image_time'].isoweekday()
        else:
            info['plan_image_time'] = None
            info['plan_image_day'] = None
        info['plan_image_year'] = skrt.core.year_fraction(
                info['plan_image_time']) 

        # Store time of plan creation.
        if info['plan_fraction'] is not None:
            info['plan_time'] = plan.get_pandas_timestamp()
            info['plan_day'] = info['plan_time'].isoweekday()
        else:
            info['plan_time'] = None
            info['plan_day'] = None
        info['plan_year'] = skrt.core.year_fraction(
                info['plan_time']) 

        # Store number of days from planning image to plan creation.
        info['days_plan_image_to_plan'] = (
                skrt.core.get_interval_in_days(
                info['plan_image_time'], info['plan_time']))
        info['whole_days_plan_image_to_plan'] = (
                skrt.core.get_interval_in_whole_days(
                info['plan_image_time'], info['plan_time']))

        # Store times of first and last treatment images.
        if treatment_images:
            info['treatment_start'] = treatment_images[0].get_pandas_timestamp()
            info['treatment_end'] = treatment_images[-1].get_pandas_timestamp()
            info['treatment_start_day'] = info['treatment_start'].isoweekday()
            info['treatment_end_day'] = info['treatment_end'].isoweekday()
        else:
            info['treatment_start'] = None
            info['treatment_end'] = None
            info['treatment_start_day'] = None
            info['treatment_end_day'] = None
        info['treatment_start_year'] = skrt.core.year_fraction(
                info['treatment_start']) 
        info['treatment_end_year'] = skrt.core.year_fraction(
                info['treatment_end']) 

        # Store treatment duration.
        info['days_treatment'] = skrt.core.get_interval_in_days(
                info['treatment_start'], info['treatment_end'])
        info['whole_days_treatment'] = skrt.core.get_interval_in_whole_days(
                info['treatment_start'], info['treatment_end'])

        # Store number of days from planning to treatment start.
        info['days_plan_to_treatment'] = (
                skrt.core.get_interval_in_days(
                info['plan_time'], info['treatment_start']))
        info['whole_days_plan_to_treatment'] = (
                skrt.core.get_interval_in_whole_days(
                info['plan_time'], info['treatment_start']))

        # Store number of days from planning image to treatment start.
        info['days_plan_image_to_treatment'] = (
                skrt.core.get_interval_in_days(
                info['plan_image_time'], info['treatment_start']))
        info['whole_days_plan_image_to_treatment'] = (
                skrt.core.get_interval_in_whole_days(
                info['plan_image_time'], info['treatment_start']))

        # Check that intervals are consistent.
        if (info['whole_days_plan_image_to_plan'] and
                info['whole_days_plan_to_treatment'] and
                info['whole_days_plan_image_to_treatment']):
            assert (info['whole_days_plan_image_to_treatment'] ==
                    info['whole_days_plan_image_to_plan'] +
                    info['whole_days_plan_to_treatment'])

        # Filter to have images separated by a minimum amount of time.
        time_separated_plan_images = skrt.core.get_time_separated_objects(
                plan_images, min_delta=min_delta, unit=unit)
        time_separated_treatment_images = skrt.core.get_time_separated_objects(
                treatment_images, min_delta=min_delta, unit=unit)

        # Store number of time-separated images.
        info['n_treatment'] = len(time_separated_treatment_images)

        return (pd.DataFrame([info]) if df else info)

    def get_image_info(self, collections=None, image_types=None,
            min_delta=4, unit='hour', df=False):
        '''
        Retrieve information about images.

        **Parameters:**

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in data paths.  If None, an empty dictionary is used.

        image_types : list, default=None
            List of strings indicating types of image for which information
            is to be retrieved, for example ['ct', 'mr'].  If None,
            information is retrieved for all image types in the patient
            dataset.

        min_delta : int/pandas.Timedelta, default=4
            Minimum time interval required between image objects.
            If an integer, the unit must be specified.

        unit : str, default='hour'
            Unit of min_delta if the latter is specified as an integer;
            ignored otherwise.  Valid units are any accepted by
            pandas.Timedelta:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html

        df : bool, default=False
            If False, return summary information as a dictionary.
            If True, return summary information as a pandas dataframe.
        '''
        # Ensure that dictionary is defined for patient groupings.
        collections = collections or {}

        # Ensure that list is defined for image_types.
        if image_types is None:
            image_types = self.combined_types('image')

        all_info = []
        for image_type in sorted(image_types):
            image_label = f'{image_type.lower()}_images'
            images = self.combined_objs(image_label)
            time_separated_images = skrt.core.get_time_separated_objects(
                    images, min_delta=min_delta, unit=unit)

            time_last = None
            time_zero = time_separated_images[0].get_pandas_timestamp()

            for image in time_separated_images:
                info = {}
                info['id'] = self.id
                info = {**info, **self.get_groupings(collections)}
                info['modality'] = image_type
                info['dx'], info['dy'], info['dz'] = image.get_size()
                info['nx'], info['ny'], info['nz'] = image.get_n_voxels()
                info['voxel_dx'], info['voxel_dy'], info['voxel_dz'] = (
                        image.get_voxel_size())
                ds = image.get_dicom_dataset()
                info['machine_manufacturer'] = ds.Manufacturer
                info['machine_model'] = ds.ManufacturerModelName
                info['station_name'] = getattr(ds, "StationName", None)
                info['timestamp'] = image.get_pandas_timestamp()
                info['day'] = None
                info['hour_in_day'] = None
                info['hour_in_week'] = None
                info['time_delta'] = None
                info['days_delta'] = None
                info['whole_days_delta'] = None
                info['days_total'] = None
                info['whole_days_total'] = None
                if info['timestamp']:
                    info['day'] = info['timestamp'].isoweekday()
                    info['hour_in_day'] = skrt.core.get_hour_in_day(
                            info['timestamp'])
                    info['hour_in_week'] = skrt.core.get_hour_in_week(
                            info['timestamp'])
                    info['day_in_week'] = skrt.core.get_day_in_week(
                            info['timestamp'])
                    if time_last is not None:
                        info['time_delta'] = info['timestamp'] - time_last
                    else:
                        info['time_delta'] = None
                    info['days_delta'] = skrt.core.get_interval_in_days(
                            time_last, info['timestamp'])
                    info['whole_days_delta'] = (
                            skrt.core.get_interval_in_whole_days(
                                time_last, info['timestamp']))
                    time_last = info['timestamp']
                    info['days_total'] = skrt.core.get_interval_in_days(
                            time_zero, info['timestamp'])
                    info['whole_days_total'] = (
                            skrt.core.get_interval_in_whole_days(
                                time_zero, info['timestamp']))
                all_info.append(info)

        return (pd.DataFrame(all_info) if df else all_info)

    def get_dose_info(self, collections=None, dose_types=None, df=False):
        '''
        Retrieve information about dose.

        **Parameters:**

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in data paths.  If None, an empty dictionary is used.

        dose_types : list, default=None
            List of strings indicating types of dose for which information
            is to be retrieved.  If None, information is retrieved for
            all dose types in the patient dataset.

        df : bool, default=False
            If False, return summary information as a dictionary.
            If True, return summary information as a pandas dataframe.
        '''
        # Ensure that dictionary is defined for patient groupings.
        collections = collections or {}

        # Ensure that list is defined for dose_types.
        if dose_types is None:
            dose_types = self.combined_types('dose')

        all_info = []
        for dose_type in sorted(dose_types):
            dose_label = f'{dose_type.lower()}_doses'
            doses = self.combined_objs(dose_label)
            for dose in sorted(doses):
                dose.load()
                info = {}
                info['id'] = self.id
                info = {**info, **self.get_groupings(collections)}
                info['timestamp'] = dose.get_pandas_timestamp()
                info['day'] = (info['timestamp'].isoweekday()
                        if info['timestamp'] else None)
                info['dose_summation_type'] = dose.get_dose_summation_type()
                info['dose_max'] = dose.get_max()
                info['dose_units'] = dose.get_dose_units()
                info['dose_type'] = dose.get_dose_type()
                info['linked_plan'] = hasattr(dose, 'plan')
                info['modality'] = dose_type
                if hasattr(dose, 'plan'):
                    info['linked_plan_status'] = dose.plan.get_approval_status()
                    info['linked_plan_fraction'] = dose.plan.get_n_fraction()
                else:
                    info['linked_plan_status'] = None
                    info['linked_plan_fraction'] = None
                all_info.append(info)

        return (pd.DataFrame(all_info) if df else all_info)

    def get_plan_info(self, collections=None, plan_types=None,
            df=False, plan_filter=None):
        '''
        Retrieve information about treatment plans.

        **Parameters:**

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in data paths.  If None, an empty dictionary is used.

        plan_types : list, default=None
            List of strings indicating types of plan for which information
            is to be retrieved.  If None, information is retrieved for
            all plan types in the patient dataset.

        df : bool, default=False
            If False, return summary information as a dictionary.
            If True, return summary information as a pandas dataframe.

        plan_filter : str, default=None
            String specifying if filtering is to be performed:
            - "first_only" : retrieve information only for first plan found;
            - "first_of_each_type" : retrieve information only for the
              first plan found of each type;
            - any other value : no filtering.
        '''
        # Ensure that dictionary is defined for patient groupings.
        collections = collections or {}

        # Ensure that list is defined for plan_types.
        if plan_types is None:
            plan_types = self.combined_types('plan')

        all_info = []
        for plan_type in sorted(plan_types):
            plan_label = f'{plan_type.lower()}_plans'
            plans = self.combined_objs(plan_label)
            for plan in sorted(plans):
                plan.load()
                info = {}
                info['id'] = self.id
                info = {**info, **self.get_groupings(collections)}
                info['timestamp'] = plan.get_pandas_timestamp()
                info['day'] = (info['timestamp'].isoweekday()
                        if info['timestamp'] else None)
                info['modality'] = plan_type
                info['plan_name'] = plan.get_name()
                info['plan_description'] = plan.get_description()
                info['plan_prescription_description'] = (
                        plan.get_prescription_description())
                info['plan_fraction'] = plan.get_n_fraction()
                info['plan_target_dose'] = plan.get_target_dose()
                info['plan_status'] = plan.get_approval_status()
                info['linked_doses'] = len(plan.doses)

                dose_summation_types = []
                max_doses = []
                for dose in plan.doses:
                    dose_summation_types.append(dose.get_dose_summation_type())
                    max_doses.append(dose.get_max())
                info['linked_dose_summation_type'] = (
                        '_'.join(dose_summation_types))
                if not max_doses:
                    info['linked_dose_max'] = None
                elif len(max_doses) == 1:
                    info['linked_dose_max'] = max_doses[0]
                else:
                    max_doses = [f'{max_dose:.4f}'
                            for max_dose in max_doses]
                    info['linked_dose_max'] = '_'.join(max_doses)

                all_info.append(info)

                if "first_of_each_type" == plan_filter:
                    break
            if "first_only" == plan_filter:
                break

        return (pd.DataFrame(all_info) if df else all_info)

    def get_structure_set_info(self, collections=None, roi_names=None,
            ss_types=None, df=False, ss_filter=None, origin=None):
        '''
        Retrieve information about structure sets.

        **Parameters:**

        collections : dict, default=None
            Dictionary where keys define types of patient grouping,
            and values are lists defining specific groupings of this type.  For
            example, {'cohort' : ['discovery', 'consolidation']} defines
            two patient cohorts - a discovery cohort and a consolidation
            cohort.  It's assumed that specific groupings are included
            in data paths.  If None, an empty dictionary is used.

        roi_names : dict, default=None
            Dictionary of names for renaming ROIs, where the keys are new 
            names and values are lists of possible names of ROIs that should
            be assigned the new name. These names can also contain wildcards
            with the '*' symbol.  Infomration is retrieved only relative to
            ROIs that, after renaming, have names included in the keys.
            If None, no renaming is performed, and information is
            retrieved relative to all ROIs.

        ss_types : list, default=None
            List of strings indicating types of structure set
            for which information is to be retrieved.  If None,
            information is retrieved for all structure-set types
            in the patient dataset.

        df : bool, default=False
            If False, return summary information as a dictionary.
            If True, return summary information as a pandas dataframe.

        ss_filter : str, default=None
            String specifying if filtering is to be performed:
            - "first_only" : retrieve information only for first structure
              set found;
            - "first_of_each_type" : retrieve information only for the
              first structure set found of each type;
            - "last_only" : retrieve information only for last structure
              set found;
            - "last_of_each_type" : retrieve information only for the
              last structure set found of each type;
            - any other value : no filtering.

        origin : tuple/str, default=None
            Tuple specifying the (x, y, z) coordinates of the point
            with respect to which structure-set extents are to be
            determined, or a string specifying how to calculate
            this point:

            - foreground_centroid: take point to be the foreground centroid
              for the associated image.

            If None, then (0, 0, 0) is used.
        '''
        # Ensure that dictionary is defined for patient groupings.
        collections = collections or {}

        # Ensure that list is defined for ss_types.
        if ss_types is None:
            ss_types = self.combined_types('structure_set')

        # Set sort order so that if information is to be retrieved
        # only for the first of last structure set then this
        # will be first in the list.
        reverse = (True if ss_filter in
                ["last_only", "last_of_each_type"] else False)

        labels = {0: "min", 1: "max"}
        all_info = []
        for ss_type in sorted(ss_types):
            ss_label = f'{ss_type.lower()}_structure_sets'
            structure_sets = self.combined_objs(ss_label)
            for ss in sorted(structure_sets, reverse=reverse):
                ss.load()
                info = {}
                info['id'] = self.id
                info = {**info, **self.get_groupings(collections)}
                info['timestamp'] = ss.get_pandas_timestamp()
                info['modality'] = ss_type

                # Optionally filter, to have only specified ROIs.
                if roi_names:
                    ss = ss.filtered_copy(roi_names, keep_renamed_only=True)
                else:
                    roi_names = sorted(ss.get_roi_names())

                info['n_roi'] = len(ss.get_roi_names())

                # Determine foreground centroid of associated image.
                if "foreground_centroid" == origin and ss.image:
                    origin = ss.image.get_foreground_bbox_centre_and_widths()[0]

                # Add information on structure-set extents.
                if not isinstance(origin, str):
                    for idx, extents in enumerate(
                            ss.get_extents(origin=origin)):
                        for jdx, label in labels.items():
                            info[f"{_axes[idx]}_{label}"] = extents[jdx]
                        info[f"d{_axes[idx]}"] = extents[1] - extents[0]

                # Add information on ROIs that are present.
                for roi_name in roi_names:
                    info[roi_name] = int(roi_name in ss.get_roi_names())

                all_info.append(info)

                if ss_filter in ["first_of_each_type", "last_of_each_type"]:
                    break
            if ss_filter in ["first_only", "last_only"]:
                break

        return (pd.DataFrame(all_info) if df else all_info)

    def get_dose_sum(self, strategies=[("FRACTION", "PLAN",), ("BEAM",)],
            set_image=True, force=False):
        """
        Get dose summed over dose objects.

        This method has been tested using datasets for patients from the IMPORT
        trials.  It may not be generally valid.

        **Parameters:**

        strategies : list, default=[("FRACTION", "PLAN",), ("BEAM",)]
            List of tuples of dose summation types to be used in a
            summation strategy.  Each strategy is considered in turn,
            until a non-zero dose sum is obtained.  The default strategy
            if first to try summing doses with summation type "FRACTION"
            or "PLAN", and if this gives a null result to try summing
            doses with summation type "BEAM".

        set_image : bool/skrt.image.Image, default=True
            If True or an Image object, associate an image to the
            summed dose, and resize the summed dose as needed to match the
            image geometry.  The image associated is the input Image object,
            of for boolean input is the image with most voxels associated
            with any of the doses in the sum.

        force : bool, default=False
            If False, return the result of any previous dose summation.  If
            True, sum doses independently of any previous result.
        """

        # If not forcing summation, return any previous result.
        if self.dose_sum is not None and not force:
            return self.dose_sum

        # Record start time
        tic = timeit.default_timer()

        # Obtain dose objects across studies, disregarding any duplicates.
        doses = remove_duplicate_doses(self.combined_objs("dose_types"))

        # Try to sum doses using defined strategies.
        for strategy in strategies:
            filtered_doses = [dose for dose in doses
                if dose.get_dose_summation_type() in strategy]

            if filtered_doses:
                self.dose_sum = sum_doses(filtered_doses)
                break

        # Define image to be associated with dose sum,
        # and/or to be used in alternative summation strategy.
        image = None
        if set_image or not self.dose_sum:
            if isinstance(set_image, Image):
                image = set_image
            else:
                image = skrt.core.get_associated_image(filtered_doses)

        # Try alternative summation strategy,
        # allowing for the dose arrays to be summed having different shapes.
        if not self.dose_sum and image:
            doses_to_sum = []
            for dose in filtered_doses:
                dose_clone = dose.clone()
                dose_clone.match_size(image)
                doses_to_sum.append(dose_clone)
            self.dose_sum = sum_doses(doses_to_sum)

        # Associate image with summed dose,
        # and resize summed dose as needed to match image size.
        if set_image and self.dose_sum and image:
            self.dose_sum.set_image(image)
            self.dose_sum.match_size(image)

        # Record end time, then store time for dose summation.
        toc = timeit.default_timer()
        self._dose_sum_time = (toc - tic)

        return self.dose_sum

    def get_subdir_studies(self, subdir=""):
        """Get list of studies within a given subdirectory."""

        subdir_studies = []
        for study in self.studies:
            if subdir == study.subdir:
                subdir_studies.append(study)

        subdir_studies.sort()

        return subdir_studies

    def last_in_interval(self, attr=None, min_date=None, max_date=None):
        """Get the last object of a given attribute <attr> in a given
        date interval."""

        files = self.combined_files(attr)
        last = None
        files.reverse()
        for file in files:
            if file.in_date_interval(min_date, max_date):
                last = file
                break
        return last

    def copy(self, outdir='.', to_keep='all', overwrite=True,
            structure_set='all'):

        '''
        Copy patient dataset, with optional filtering.

        **Parameters:**

        outdir : str/pathlib.Path, default = '.'
            Output directory to which patient dataset is to be copied.

        to_keep : str/list, default = 'all'
            Specification of data types to be copied.  If 'all', all data
            are copied.  If a list, only data from the listed study
            ubdirectories are copied.  For example,
            to_keep=['CT', 'RTSTRUCT/CT'] would result in copying of only
            CT images and associated structure sets.

        overwrite : bool, default=True
            If True, delete any existing patient directory, and its contents,
            in the output directory before copying.

        structure_set : str/int/list, default = 'all'
            Select structure set(s) to copy.  If 'all', all structure sets 
            are copied.  If an integer or list of integers, a sorted list
            of structure sets for each image set is created, and only
            structure sets at the indicated positions in this list are
            copied.  For example, structure_set=[0, -1] would result
            in the first and last structure sets being copied.
        '''
        # Ensure that structure_set is a list or 'all'.
        if isinstance(structure_set, int):
            structure_set = [structure_set]
        if not isinstance(structure_set, list) and 'all' != structure_set:
            structure_set = []

        # Define output patient directory, and ensure that it exists.
        outdir_path = Path(skrt.core.fullpath(str(outdir)))
        patient_dir = outdir_path / self.id
        if patient_dir.exists() and overwrite:
            shutil.rmtree(patient_dir)
        patient_dir.mkdir(parents=True, exist_ok=True)

        for study in self.studies:

            # Define output study directory, and ensure that it exists.
            study_dir = patient_dir / study.subdir / study.timestamp
            study_dir.mkdir(parents=True, exist_ok=True)

            # Easy case: copy everything in the study directory.
            if 'all' == to_keep:
                shutil.copytree(study.path, str(study_dir), dirs_exist_ok=True)

            else:
                # Loop over subdirectories with content to be copied.
                for subdir in to_keep:
                    indir = Path(study.path) / subdir
                    if not indir.exists():
                        continue
                    outdir = study_dir / subdir
                    outdir.mkdir(parents=True, exist_ok=True)
                    # Copy everything if this isn't an RTSTRUCT directory
                    # to which selection is to be applied.
                    if ((not str(subdir).startswith('RTSTRUCT'))
                            or 'all' == structure_set):
                        shutil.copytree(str(indir), str(outdir),
                                dirs_exist_ok=True)
                    else:
                        # Subdirectories from which RTSTRUCT data
                        # are to be copied may be specified as any of:
                        # 'RTSTRUCT', 'RTSTRUCT/<modality>',
                        # 'RTSTRUCT/<modality>/<timestamp>'.
                        # Code tries to allow for any of these.
                        elements = str(subdir).strip(os.path.sep).split(
                                os.path.sep)
                        rtstruct_indir = Path(study.path) / elements[0]
                        rtstruct_outdir = study_dir / elements[0]

                        # Identify and loop over modalities.
                        if len(elements) > 1:
                            modalities = [elements[1]]
                        else:
                            modalities = rtstruct_indir.iterdir()
                        for modality in modalities:
                            modality_indir = rtstruct_indir / modality
                            modality_outdir = rtstruct_outdir / modality

                            # Identify and loop over timestamps.
                            if len(elements) > 2:
                                timestamps = [elements[2]]
                            else:
                                timestamps = modality_indir.iterdir()
                            for timestamp in timestamps:
                                ss_indir = modality_indir / timestamp

                                # Identify structure sets for copying.
                                structure_sets = list(ss_indir.iterdir())
                                structure_sets.sort()
                                ss_to_copy = []
                                for i in structure_set:
                                    try:
                                        ss = structure_set[i]
                                    except IndexError:
                                        ss = None
                                    if ss and ss not in ss_to_copy:
                                        ss_co_topy.append(ss)

                                # Copy selected structure sets.
                                for ss in ss_to_copy:
                                    ss_outdir = modality_outdir / timestamp
                                    ss_outdir.mkdir(parents=True, exist_ok=True)
                                    ss_in = ss_indir / ss
                                    ss_out = ss_outdir / ss
                                    shutil.copy2(ss_in, ss_out)

    def write(
        self,
        outdir=".",
        ext=".dcm",
        to_ignore=None,
        overwrite=True,
        structure_set="all",
        dose='all',
        root_uid=None
    ):
        """Write files tree."""

        if not ext.startswith("."):
            ext = f".{ext}"

        patient_dir = os.path.join(os.path.expanduser(outdir), self.id)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        elif overwrite:
            shutil.rmtree(patient_dir)
            os.mkdir(patient_dir)

        if to_ignore is None:
            to_ignore = []

        for study in self.studies:

            # Make study directory
            study_dir = os.path.join(
                patient_dir, study.subdir, study.timestamp
            )
            if not os.path.exists(study_dir):
                os.makedirs(study_dir)

            # Loop through image types
            for image_type in study.image_types:

                if image_type in to_ignore:
                    continue

                image_type_dir = os.path.join(study_dir, image_type.upper())
                if not os.path.exists(image_type_dir):
                    os.mkdir(image_type_dir)

                # Write all images of this image type
                for im in getattr(study, f"{image_type.lower()}_images"):

                    # Make directory for this image
                    if im.path:
                        im_timestamp = os.path.basename(im.path)
                    else:
                        im_timestamp = im.timestamp
                    im_dir = os.path.join(image_type_dir, im_timestamp)

                    # Write image data to nifti
                    if ext == ".dcm":
                        outname = im_dir
                    else:
                        outname = f"{im_dir}{ext}"
                    if os.path.exists(outname) and not overwrite:
                        continue
                    Image.write(im, outname, patient_id=self.id,
                                modality=image_type.upper(), root_uid=root_uid)

                    # Write associated structure sets
                    self.write_non_image_data(im, image_type, im_timestamp,
                            'structure_sets', 'RTSTRUCT', structure_set,
                            study_dir, overwrite, ext)

                    # Write associated doses
                    self.write_non_image_data(im, image_type, im_timestamp,
                            'doses', 'RTDOSE', dose, study_dir, overwrite, ext)

    def write_non_image_data(self, im=None, image_type=None, im_timestamp=None,
            items=None, modality=None, selection=None, outdir='.',
            overwrite=True, ext='.dcm'):

        if not ext.startswith("."):
            ext = f".{ext}"

        # Find data to write
        im_get_items = getattr(im, f'get_{items}')
        if selection == "all":
            items_to_write = im_get_items()
        elif selection is None:
            items_to_write = []
        elif isinstance(selection, int):
            items_to_write = [im_get_items()[selection]]
        elif skrt.core.is_list(selection):
            items_to_write = [im_get_items()[i] for i in items]
        else:
            raise TypeError('Unrecognised {items} selection: {selection}')

        # Write structure sets for this image
        for item in items_to_write:

        # Find path to output structure directory
            if modality in str(item.path):
                item_subpath = item.path.split(modality, 1)[1].strip(
                        os.path.sep)
                item_dir = os.path.join(outdir, modality, item_subpath)
            else:
                item_dir = os.path.join(
                        outdir, modality, image_type.upper(), im_timestamp)
            filename = f'{modality}_{item.timestamp}'

            # Ensure it exists
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)

            # Write dicom structure set
            if ext == '.dcm':
                if not os.path.exists(item_dir) or overwrite:
                    if not filename.endswith('.dcm'):
                        filename = f'{filename}.dcm'
                    if modality == 'RTSTRUCT':
                        item.write(outname=filename, outdir=item_dir)
                    else:
                        item_path = os.path.join(item_dir, filename)
                        item.write(outname=item_path)
            # Write ROIs to individual files
            elif 'RTSTRUCT' == modality:
                item.write(outdir=item_dir, ext=ext)


    def copy_dicom(self, outdir=".", studies_to_copy=None, overwrite=True,
            **kwargs):
        """
        Copy patient dicom data.

        **Parameters:**
        outdir - str, default='.'
            Top-level output directory.  Within this, there will be a
            patient sub-directory, containing a sub-directory for each
            study, containing in turn a sub-directory for each data
            modality.

        studies_to_copy: list/dict, default=None
            List of indices of studies for which data are
            to be written, 0 being the earliest study and -1 being
            the most recent: or a dictionary where keys will be used
            as names of subdirectories grouping studies, and values
            are indices of studies to be grouped.  If set to
            None, data for all studies are written, keeping any
            existing grouping.

        overwrite : bool, default=True
            If True, delete and recreate patient sub_directory
            before copying files.  If False and the patient sub-directory
            exists already, copy files to the existing directory.

        **kwargs
            Keyword arguments passed on to
                skrt.patient.Study.copy_dicom().
            For details, see this method's documentation.
        """
        # Define patient output directory.
        patient_dir = skrt.core.make_dir(Path(fullpath(outdir)) / self.id,
                overwrite=overwrite)

        # If studies_to_copy is None, set to select all studies.
        if studies_to_copy is None:
            studies_to_copy = {}
            for idx, study in enumerate(self.studies):
                if not study.subdir in studies_to_copy:
                    studies_to_copy[study.subdir] = []
                studies_to_copy[study.subdir].append(idx)
        elif not isinstance(studies_to_copy, dict):
            studies_to_copy = {"": studies_to_copy}

        # Process selected studies.
        for group, indices in studies_to_copy.items():
            for study in get_indexed_objs(self.studies, indices):
                # Overwriting taken into account at patient level,
                # so don't overwrite at study level.
                study.copy_dicom(outdir=patient_dir / group / study.timestamp,
                        overwrite=False, **kwargs)

def find_matching_object(obj, possible_matches):
    """For a given object <obj> and a list of potential matching objects
    <possible_matches>, find an object that matches <obj>'s path,
    <obj>'s date and time, or SOP Instance UID."""

    # Try matching on path
    for match in possible_matches:
        if os.path.basename(obj.path) == os.path.basename(match.path):
            return (match, None)

    # If no path match, try matching on timestamp
    for match in possible_matches:
        if (match.date == obj.date) and (match.time == obj.time):
            return (match, None)

    # If no timestamp match, try matching on SOP Instance UID
    if issubclass(type(obj), Dose):
        # ds_obj = obj.get_dicom_dataset()
        ds_obj = pydicom.dcmread(obj.path, force=True)
        if hasattr(ds_obj, 'ReferencedImageSequence'):
            # Omit part of UID after final dot,
            # to be insenstive to slice/frame considered.
            referenced_sop_instance_uid = '.'.join(
                    ds_obj.ReferencedImageSequence[-1]
                    .ReferencedSOPInstanceUID.split('.')[:-1])
            for match in possible_matches:
                # ds_match = match.get_dicom_dataset()
                ds_match = pydicom.dcmread(match.files[0].path, force=True)
                if hasattr(ds_match, 'SOPInstanceUID'):
                    sop_instance_uid = '.'.join(
                            ds_match.SOPInstanceUID.split('.')[:-1])
                    if sop_instance_uid == referenced_sop_instance_uid:
                        return (match, None)

    elif issubclass(type(obj), Plan):
        # ds_obj = obj.get_dicom_dataset()
        ds_obj = pydicom.dcmread(obj.path, force=True)
        if hasattr(ds_obj, 'ReferencedStructureSetSequence'):
            referenced_sop_instance_uid = ds_obj.\
                    ReferencedStructureSetSequence[-1].ReferencedSOPInstanceUID
            for match in possible_matches:
                for structure_set in match.get_structure_sets():
                    # ds_match = structure_set.get_dicom_dataset()
                    ds_match = pydicom.dcmread(structure_set.path, force=True)
                    if hasattr(ds_match, 'SOPInstanceUID'):
                        sop_instance_uid = ds_match.SOPInstanceUID
                        if sop_instance_uid == referenced_sop_instance_uid:
                            return (match, structure_set)

    return (None, None)
