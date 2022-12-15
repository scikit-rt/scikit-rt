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
        for modality_path in study_path.iterdir():
            modality = modality_path.name
            for timestamp_path in modality_path.iterdir():
                image_paths = sorted(list(
                    timestamp_path.glob(f"{modality}*.nii*")))
                if 1 != len(image_paths):
                    continue

                modality = modality.lower()
                images = f"{modality}_images"
                im = Image(images[0], load=True)
                if not modality in self.image_types:
                    self.image_types[modality] = []
                    setattr(self, images, [])
                self.image_types[modality].append(im)
                getattr(self, images).append(im)
                
                roi_paths = {str(roi_path) :
                        "_".join(roi_path.name.split(".")[0].split("_")[5 :])
                        for roi_path in timestamp_path.glob(f"RTSTRUCT*.nii*")}
                if not roi_paths:
                    continue

                structure_sets = f"{modality}_structure_sets"
                rois = [ROI(source=roi_path, name=roi_name)
                        for roi_path, roi_name in roi_paths.items()]
                ss = StructureSet(path=rois, image=im, load=True)
                if not modality in self.structure_set_types:
                    self.structure_set_types[modality] = []
                    setattr(self, structure_sets, [])
                self.structure_set_types[modality].append(ss)
                getattr(self, structure_sets).append(ss)
