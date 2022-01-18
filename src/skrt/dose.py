"""Classes related to doses and plans."""

import numpy as np
import os
import pydicom
import functools

import matplotlib

from skrt.core import MachineData
import skrt.image


class ImageOverlay(skrt.image.Image):
    """Image that can be associated with another Image, and whose plotting
    functionality includes the ability to plot overlaid on its associated 
    Image."""

    def __init__(self, path, load=True, image=None, *args, **kwargs):

        skrt.image.Image.__init__(self, path, *args, **kwargs)
        self.set_image(image)

        # Default dose plotting settings
        self._default_cmap = "jet"
        self._default_colorbar_label = "Intensity"
        self._default_vmin = 0
        if not hasattr(self, "_default_vmax"):
            self._default_vmax = None

    def set_image(self, image):
        """Set associated image, initialising it if needed."""

        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)
        self.image = image

    def plot(
        self, 
        view="x-y",
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        colorbar=False,
        include_image=True, 
        opacity=None, 
        mpl_kwargs=None,
        show=True,
        **kwargs
    ):
        """Plot this overlay, optionally overlaid on its associated Image.

        **Parameters**:

        view : str
            Orientation in which to compute the index. Can be "x-y", "y-z", or
            "x-z".

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot. If None, new axes will be created.

        gs : matplotlib.gridspec.GridSpec, default=None
            If not None and <ax> is None, new axes will be created on the
            current matplotlib figure with this gridspec.

        figsize : float, default=None
            Figure height in inches; only used if <ax> and <gs> are None.


        include_image : bool, default=False
            If True and this ImageOverlay has an associate image, it will be 
            plotted overlaid on the image.
        
        opacity : float, default=None
            If plotting on top of an image, this sets the opacity of the 
            overlay (0 = fully transparent, 1 = fully opaque).

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        colorbar : bool, default=True
            If True, a colorbar will be drawn alongside the plot.

        show : bool, default = True
            If True, the plot will be displayed immediately.

        `**`kwargs:
            Keyword args to pass to skrt.image.Image.plot().
        """

        # Set up axes
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        self.load()

        # Plot underlying image
        if include_image and self.image is not None:
            self.image.plot(view, ax=self.ax, show=False)

            # Use default transprency if plotting image and opacity is None
            if opacity is None:
                opacity = 0.5

        # Add opacity to mpl_kwargs
        if mpl_kwargs is None:
            mpl_kwargs = {}
        if opacity is not None:
            mpl_kwargs["alpha"] = opacity

        # Plot self
        skrt.image.Image.plot(
            self, 
            view=view,
            ax=self.ax,
            zoom=zoom,
            colorbar=colorbar,
            mpl_kwargs=mpl_kwargs, 
            show=show, 
            **kwargs
        )


    def view(self, include_image=True, kwarg_name=None, **kwargs):
        """View with BetterViewer, optionally overlaying on image.

        **Parameters**:

        include_image : bool, default=False
            If True, this ImageOverlay will be displayed overlaid on its
            underlying Image.

        kwarg_name : str, default=None
            Name of kwarg under which to provide self to the call to
            self.image.Image.view() if include_image=True. By default,
            self will be passed to the "dose" parameter.

        `**`kwargs :
            Keyword args to pass to BetterViewer initialisation.
        """

        from skrt.better_viewer import BetterViewer

        if include_image and self.image is not None:
            if kwarg_name is None:
                kwarg_name = "dose"
            kwargs[kwarg_name] = self
            return self.image.view(**kwargs)

        return skrt.image.Image.view(self, **kwargs)

    @functools.cached_property
    def max(self):
        return self.get_data().max()



class Dose(ImageOverlay):
    """Class representing a dose map. The same as an Image but with overridden
    plotting behaviour and extra functionality relating to ROIs."""

    def __init__(self, *args, **kwargs):

        ImageOverlay.__init__(self, *args, **kwargs)

        # Plot settings specific to dose map
        self._default_cmap = "jet"
        self._default_colorbar_label = "Dose (Gy)"

    def load(self, *args, **kwargs):
        """Load self and set default maximum plotting intensity from max of
        data array if not yet set."""

        skrt.image.Image.load(self, *args, **kwargs)
        self._default_vmax = self.max

    def set_image(self, image):
        """Set associated image. Image.add_dose(self) will also be called."""

        ImageOverlay.set_image(self, image)
        if image is not None:
            image.add_dose(self)

    def view(self, **kwargs):

        return ImageOverlay.view(self, kwarg_name="dose", **kwargs)

    def get_dose_in_roi(self, roi):
        """Return 1D numpy array containing all of the dose values for the 
        voxels inside an ROI. Fails if the ROI and dose arrays are not the 
        same size."""

        roi.create_mask()
        if not self.has_same_geometry(roi.mask):
            raise RuntimeError("Dose field and ROI mask must have same geometry")

        dose_in_roi = self.get_data(standardise=True) \
                * roi.get_mask(standardise=True)
        return dose_in_roi[dose_in_roi > 0]

    def get_max_dose_in_rois(self, rois=[]):
        '''
        Return maximum dose in a set of rois.

        **Parameter:**

        rois : list, default=[]
            List of ROI objects, for which maximum dose is to be determined.
        '''
        # Determine the maximum dose for the input roi(s).
        dose_max = 0
        for roi in rois:
            doses = list(self.get_dose_in_roi(roi))
            doses.append(dose_max)
            dose_max=max(doses)
        return dose_max

    def plot_DVH(self, rois=[], bins=50, dose_min=0, dose_max=None,
            figsize=(8, 4), lw=2, n_colour=10, cmap='turbo', grid=True,
            fname=None):
        '''
        Plot dose-volume histogram for specified ROI(s).

        **Parameters:**

        rois : ROI/StructureSet/list, default=[]
            ROI(s) for which dose-volume histogram is to be plotted.  This
            can be a single skrt.structures.ROI object, a single
            skrt.structures.StructureSet object, or a list containing any
            combination of ROI and StructureSet objects.

        bins : int, default=50
            Number of bins in the dose-volume histogram.

        dose_min : float, default=0
            Minimum dose value to be included.

        dose_max : float, default=None
            Maximum dose value to be included.  If None, this is set to
            the maximum dose for the ROI(s) considered.

        figsize : tuple of floats, default=(8, 5)
            Dimensions (inches) of the matplotlib figure

        lw : float, default=2
            Width (relative to matplotlib default) of line used in drawing
            histogram outlines.

        n_colour : int, default=10
            Number of colours over which to cycle when drawing histograms.

        cmap : str, default='turbo'
            Name of a matplotlib colour map from which to define set of
            colours over which to cycle when drawing histograms.

            For predfined names see:

            https://matplotlib.org/stable/gallery/color/colormap_reference.html

        grid : bool, default=True
            If True, overlay a grid on the dose-volume histogram.

        fname : str, default=None
            Name of file for saving output.
        '''

        # Create figure and define the list of colours.
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
        colours = matplotlib.cm.get_cmap(cmap)(np.linspace(0, 1,n_colour))
        ax.set_prop_cycle(color=colours)

        # Ensure that rois is a list.
        if issubclass(type(rois),
                (skrt.structures.ROI, skrt.structures.StructureSet)):
            rois = [rois]

        # Create a list containing all unique rois.
        all_rois = []
        for item in rois:
            if issubclass(type(item), skrt.structures.ROI):
                candidate_rois = [item]
            elif issubclass(type(item), skrt.structures.StructureSet):
                candidate_rois = item.get_rois()
            for roi in candidate_rois:
                if not roi in all_rois:
                    all_rois.append(roi)

        # Determine the maximum dose for the input roi(s).
        if dose_max is None:
            dose_max = self.get_max_dose_in_rois(all_rois)

        # Plot the dose-volume histograms, and extract information for legend.
        lines= []
        labels= []
        for roi in all_rois:
            doses = self.get_dose_in_roi(roi)
            n, bins, patches = ax.hist(doses, bins=bins,
                    range=(dose_min,dose_max), lw=lw, histtype='step',
                    density=True, cumulative=-1)
            colour = patches[0].get_edgecolor()
            lines.append(matplotlib.lines.Line2D([0], [0], color=colour, lw=lw))
            labels.append(roi.name)

        # Label the axes, add grid and legend, tighten layout.
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Volume fraction')
        ax.grid(grid)
        ax.legend(handles=lines, labels=labels, loc='center left',
                bbox_to_anchor=(1.01, 0.5))
        matplotlib.pyplot.tight_layout()

        # Show figure or save to file.
        if fname is not None:
            matplotlib.pyplot.savefig(fname)
        else:
            matplotlib.pyplot.show()

        return ax

    def get_mean_dose(self, roi):
        """Get mean dose inside an ROI."""

        return np.mean(self.get_dose_in_roi(roi))


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
