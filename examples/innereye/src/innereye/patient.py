"""
Patient and Study classes for use with InnerEye data.

These classes are simplified versions of skrt.Patient and skrt.Study,
designed for handling of NIfTI-format images and structure sets,
organised in a way compatible with use by InnerEye.
"""
from pathlib import Path

from skrt import Image, ROI, StructureSet
from skrt.core import fullpath, Archive, PathData

class Patient(PathData):
    """
    Top-level class for collecting patient data sorted for InnerEye.
    """

    def __init__(self, path):
        self.path = fullpath(path)
        self.id = Path(self.path).name
        self.studies = self.create_objects(dtype=Study)

    def get_catalogue_entries(self, id1=1, image_types=None,
            roi_names=None, nz_min=16):
        """
        Obtain patient entries for InnerEye dataset catalogue.

        **Parameters:**

        id1 - int, default=1
            Subject identifier to be assigned to first image and
            associated structure set.  The identifier will be increased
            by 1 for each subsequent image.  All images processed
            by a single InnerEye instance must have a different identifier.

        image_types - list, default=None
            List of types of image to be catalogued.  If None, all image
            types in the dataset will be catalogued.

        roi_names - list, default=None
            List of names of ROIs to be catalogued.  If None, all ROIs
            in the dataset will be catalogued.

        nz_min - int, default=16
            Minimum number of image slices for image to be catalogued.
        """
        lines = []
        id2 = int(id1)
        for study in self.studies:
            lines.extend(study.get_catalogue_entries(id2, image_types,
                roi_names, nz_min))
            if lines:
                id2 = 1 + int(lines[-1].split(",")[0])
        return lines

class Study(Archive):
    """
    Class for collecting study-level data sorted for InnerEye.

    This class collects data for images and structure sets.
    """

    def __init__(self, path):
        super().__init__(path)

        self.image_types = {}
        self.structure_set_types = {}
        study_path = Path(self.path)

        # Loop over image modalities.
        for modality_path in study_path.iterdir():
            modality = modality_path.name

            # Loop over timestamp for current modality.
            for timestamp_path in modality_path.iterdir():
                image_paths = sorted(list(
                    timestamp_path.glob(f"{modality}*.nii*")))
                if 1 != len(image_paths):
                    continue

                # Load image for current modality and timestamp.
                modality = modality.lower()
                images = f"{modality}_images"
                im = Image(image_paths[0], load=True)
                if not modality in self.image_types:
                    self.image_types[modality] = []
                    setattr(self, images, [])
                self.image_types[modality].append(im)
                getattr(self, images).append(im)
                
                # Obtain paths to ROI files for current modality and timestamp.
                roi_paths = {str(roi_path) :
                        "_".join(roi_path.name.split(".")[0].split("_")[5 :])
                        for roi_path in timestamp_path.glob(f"RTSTRUCT*.nii*")}
                if not roi_paths:
                    continue

                # Load structure set for current modality and timestamp.
                structure_sets = f"{modality}_structure_sets"
                rois = [ROI(source=roi_path, name=roi_name)
                        for roi_path, roi_name in roi_paths.items()]
                ss = StructureSet(path=rois, image=im, load=True)
                if not modality in self.structure_set_types:
                    self.structure_set_types[modality] = []
                    setattr(self, structure_sets, [])
                self.structure_set_types[modality].append(ss)
                getattr(self, structure_sets).append(ss)

    def get_catalogue_entries(self, id1=1, image_types=None,
            roi_names=None, nz_min=16):
        """
        Obtain study entries for InnerEye dataset catalogue.

        **Parameters:**

        id1 - int, default=1
            Subject identifier to be assigned to first image and
            associated structure set.  The identifier will be increased
            by 1 for each subsequent image.  All images processed
            by a single InnerEye instance must have a different identifier.

        image_types - list, default=None
            List of types of image to be catalogued.  If None, all image
            types in the dataset will be catalogued.

        roi_names - list, default=None
            List of names of ROIs to be catalogued.  If None, all ROIs
            in the dataset will be catalogued.

        nz_min - int, default=16
            Minimum number of image slices for image to be catalogued.
        """
        # Initialise variables.
        if image_types is None:
            image_types = sorted(list(self.image_types.keys()))
        subject = int(id1)
        series_id = ""
        institution_id = ""
        lines = []

        # Loop over image types.
        for image_type in image_types:
            im_channel = ("ct" if image_type.lower in ["ct", "mvct"]
                    else image_type.lower())
            # Loop over images of current type.
            for im in self.image_types[image_type]:

                # Check that image satisfies size criterion.
                nx, ny, nz = im.get_n_voxels()
                if nz < nz_min:
                    continue

                # Expect number of structure sets to be one or zero.
                ss_ok = (len(im.structure_sets) <= 1)
                if not ss_ok:
                    continue

                # If image has an associated structure set,
                # check that this contains any ROIs required.
                if im.structure_sets:
                    ss = im.structure_sets[0]
                if roi_names and ss:
                    missing_rois = [roi_name for roi_name in roi_names
                            if not roi_name in ss.get_roi_names()]
                    if missing_rois:
                        ss_ok = False
                        break

                # Create catalogue entry for current image and structure set.
                if ss_ok:
                    lines.append(f"{subject},{im.path},{im_channel}"
                            f"{series_id},{institution_id},{nx},{ny},{nz}")
                    if ss:
                        roi_names_to_catalogue = roi_names or ss.get_roi_names()
                        for roi_name in roi_names_to_catalogue:
                            lines.append(f"{subject},{ss[roi_name].path},"
                                    f"{roi_name},{series_id},{institution_id},"
                                    f"{nx},{ny},{nz}")
                    subject += 1

        return lines
