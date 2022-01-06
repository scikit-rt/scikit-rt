"""Classes related to doses and plans."""

import numpy as np
import os
import pydicom
import functools

from skrt.core import MachineData
import skrt.image


class Dose(skrt.image.Image):
    """Class representing a dose map. The same as an Image but with overridden
    plotting behaviour and extra functionality relating to ROIs."""

    def __init__(self, path, load=True, image=None, *args, **kwargs):

        skrt.image.Image.__init__(self, path, *args, **kwargs)
        self.image = image

        # Default dose plotting settings
        self._default_cmap = "jet"
        self._default_colorbar_label = "Dose (Gy)"
        self._default_vmin = 0
        self._default_vmax = None

    def load(self, *args, **kwargs):
        """Load self and set default maximum plotting intensity from max of
        data array."""

        skrt.image.Image.load(self, *args, **kwargs)
        self._default_vmax = self.max

    def set_image(self, image):
        """Set associated image."""

        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)

        self.image = image

    def plot(
        self, 
        include_image=False, 
        opacity=None, 
        mpl_kwargs=None,
        colorbar=False,
        show=True,
        **kwargs
    ):
        """Plot this dose map, optionally overlaid on its associated image.

        **Parameters**:

        include_image : bool, default=False
            If True and this Dose has an associate image, the dose map will
            be plotted overlaid on the image.
        
        opacity : float, default=None
            If plotting on top of an image, this sets the opacity of the dose
            map (0 = fully transparent, 1 = fully opaque).

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        colorbar : bool, default=True
            If True, a colorbar of dose level will be drawn alongside the plot.

        show : bool, default = True
            If True, the plot will be displayed immediately.

        `**`kwargs:
            Keyword args to pass to skrt.image.Image.plot().
        """

        # Plot underlying image
        if include_image and self.image is not None:
            self.image.plot(show=False, **kwargs)
            kwargs["ax"] = self.image.ax

        # Add opacity to mpl_kwargs
        if include_image and self.image is not None:
            if opacity is None:
                opacity = 0.5
            if mpl_kwargs is None:
                mpl_kwargs = {}
            mpl_kwargs["alpha"] = opacity

        # Plot dose field
        skrt.image.Image.plot(
            self, 
            colorbar=colorbar,
            mpl_kwargs=mpl_kwargs, 
            show=show, 
            **kwargs
        )

    def plot_DVH(self, roi):
        pass

    def get_mean_dose(self, roi):
        pass

    @functools.cached_property
    def max(self):
        return self.get_data().max()


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
