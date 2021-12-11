"""Classes related to doses and plans."""

import numpy as np
import os
import pydicom
import functools

from skrt.core import MachineData
from skrt.image import Image


class Dose(Image):

    def __init__(self, path, image=None, *args, **kwargs):
        Image.__init__(self, path, *args, **kwargs)
        self.image = image

    def plot(self, include_image=False, opacity=None, **kwargs):
        '''
        '''

        # Plot underlying image
        if opacity is None:
            opacity = 1 if not include_image else 0.5
        if include_image and self.image is not None:
            self.image.plot(show=False)
            kwargs["ax"] = self.image.ax

        # Set default plotting kwargs
        dose_kwargs = {
            "cmap": "jet",
            "vmin": 0,
            "vmax": self.max,
            "alpha": opacity
        }
        mpl_kwargs = kwargs.get("mpl_kwargs", {})
        mpl_kwargs.update(dose_kwargs)
        kwargs["mpl_kwargs"] = mpl_kwargs

        # Plot dose field
        Image.plot(self, **kwargs)

    def plot_DVH(self, roi):
        pass

    def get_mean_dose(self, roi):
        pass

    @functools.cached_property
    def max(self):
        return self.get_data().max()


class RtDose(MachineData):
    def __init__(self, path=""):

        MachineData.__init__(self, path)

        if not os.path.exists(path):
            return

        ds = pydicom.read_file(path, force=True)

        # Get dose summation type
        try:
            self.summation_type = ds.DoseSummationType
        except AttributeError:
            self.summation_type = None

        # Get slice thickness
        if ds.SliceThickness:
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = None

        # Get image position and voxel sizes
        if ds.GridFrameOffsetVector[-1] > ds.GridFrameOffsetVector[0]:
            self.reverse = False
            self.image_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[0]),
            )
        else:
            self.reverse = True
            self.image_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[-1]),
            )
        self.voxel_size = (
            float(ds.PixelSpacing[0]),
            float(ds.PixelSpacing[1]),
            slice_thickness,
        )
        self.transform_ijk_to_xyz = get_transform_ijk_to_xyz(self)
        self.image_stack = None

    def get_image_stack(self, rescale=True, renew=False):

        if self.image_stack is not None and not renew:
            return self.image_stack

        # Load dose array from dicom
        ds = pydicom.read_file(self.path, force=True)
        self.image_stack = np.transpose(ds.pixel_array, (1, 2, 0))

        # Rescale voxel values
        if rescale:
            try:
                rescale_intercept = ds.RescaleIntercept
            except AttributeError:
                rescale_intercept = 0
                self.image_stack = self.image_stack * float(ds.DoseGridScaling) + float(
                    rescale_intercept
                )

        if self.reverse:
            self.image_stack[:, :, :] = self.image_stack[:, :, ::-1]

        return self.image_stack


class RtPlan(MachineData):
    def __init__(self, path=""):

        MachineData.__init__(self, path)

        ds = pydicom.read_file(path, force=True)

        try:
            self.approval_status = ds.ApprovalStatus
        except AttributeError:
            self.approval_status = None

        try:
            self.n_fraction_group = len(ds.FractionGroupSequence)
        except AttributeError:
            self.n_fraction_group = None

        try:
            self.n_beam_seq = len(ds.BeamSequence)
        except AttributeError:
            self.n_beam_seq = None

        self.n_fraction = None
        self.target_dose = None
        if self.n_fraction_group is not None:
            self.n_fraction = 0
            for fraction in ds.FractionGroupSequence:
                self.n_fraction += fraction.NumberOfFractionsPlanned
                if hasattr(fraction, "ReferencedDoseReferenceSequence"):
                    if self.target_dose is None:
                        self.target_dose = 0.0
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        self.target_dose += dose.TargetPrescriptionDose
