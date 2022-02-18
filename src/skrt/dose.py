"""Classes related to doses and plans."""

import numpy as np
import os
import pydicom
import functools

import matplotlib

from skrt.core import Archive, Data
import skrt.image
import skrt.structures


class ImageOverlay(skrt.image.Image):
    """Image that can be associated with another Image, and whose plotting
    functionality includes the ability to plot overlaid on its associated 
    Image."""

    def __init__(self, path="", load=True, image=None, *args, **kwargs):

        skrt.image.Image.__init__(self, path, load, *args, **kwargs)
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
        view=None,
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
            "x-z".  If None, the initial view is chosen to match
            the image orienation.

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
        if not view:
            view = self.get_orientation_view()
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

        # Ensure linking between image and dose
        self.set_image(self.image)

        # Plot settings specific to dose map
        self._default_cmap = "jet"
        self._default_colorbar_label = "Dose (Gy)"

        # Delete spurious attributes inherited from Image class
        for attribute in ['doses', 'plans']:
            if hasattr(self, attribute):
                delattr(self, attribute)

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

    def set_plan(self, plan):
        """Set associated plan, initialising it if needed."""

        # Convert to Plan object if needed
        if plan and not isinstance(plan, Plan):
            plan = Plan(plan)

        # Assign plan to self
        self.plan = plan

        # Assign self to the plan
        if self.plan is not None:
            self.plan.add_dose(self)

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
            fname=None, legend_bbox_to_anchor=(1.01, 0.5),
            legend_loc='center left'):
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
        ax.legend(handles=lines, labels=labels,
                bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)
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


class Plan(Archive):
    def __init__(self, path="", load=True):

        self.loaded = False
        self.path = None
        self.name = None
        self.approval_status = None
        self.n_fraction_group = None
        self.n_beam_seq = None
        self.n_fraction = None
        self.organs_at_risk = None
        self.target_dose = None
        self.targets = None
        self.image = None
        self.structure_set = None
        self.doses = []
        self.objectives = Data()
        for constraint_attribute in Constraint.get_weight_and_objectives():
            setattr(self.objectives, constraint_attribute, None)

        Archive.__init__(self, path)

        self.constraints_loaded = False
        if load:
            self.load()

    def load(self, force=False):
        '''
        Load plan data.
        '''

        if self.loaded and not force:
            return

        self.dicom_dataset = pydicom.dcmread(self.path, force=True)

        self.name = getattr(self.dicom_dataset, 'RTPlanName', None)

        try:
            self.approval_status = self.dicom_dataset.ApprovalStatus
        except AttributeError:
            self.approval_status = None

        try:
            self.n_fraction_group = len(
                    self.dicom_dataset.FractionGroupSequence)
        except AttributeError:
            self.n_fraction_group = None

        try:
            self.n_beam_seq = len(self.dicom_dataset.BeamSequence)
        except AttributeError:
            self.n_beam_seq = None

        self.n_fraction = None
        if self.n_fraction_group is not None:
            self.n_fraction = 0
            for fraction in self.dicom_dataset.FractionGroupSequence:
                self.n_fraction += fraction.NumberOfFractionsPlanned
                if hasattr(fraction, "ReferencedDoseReferenceSequence"):
                    if self.target_dose is None:
                        self.target_dose = 0.0
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        self.target_dose += dose.TargetPrescriptionDose

        self.loaded = True
        self.load_constraints()

    def load_constraints(self, force=False):
        '''
        Load dose constraints from plan.
        '''

        if ((self.constraints_loaded and not force) or not self.structure_set):
            return

        rois = {}
        for roi in self.structure_set.get_rois():
            rois[roi.number] = roi

        self.organs_at_risk = []
        self.targets = []
        
        ds = self.get_dicom_dataset()

        for item in ds.DoseReferenceSequence:
            roi = rois.get(item.ReferencedROINumber, None)
            if roi is None:
                continue
            roi.roi_type = item.DoseReferenceType
            roi.constraint = Constraint()
            roi.constraint.weight = float(item.ConstraintWeight)
            if 'ORGAN_AT_RISK' == item.DoseReferenceType:
                roi.constraint.maximum_dose = float(item.OrganAtRiskMaximumDose)
                roi.constraint.full_volume_dose = float(
                        item.OrganAtRiskFullVolumeDose)
                roi.constraint.overdose_volume_fraction = float(
                        item.OrganAtRiskOverdoseVolumeFraction)
                self.organs_at_risk.append(roi)
            elif 'TARGET' == item.DoseReferenceType:
                roi.constraint.minimum_dose = float(item.TargetMinimumDose)
                roi.constraint.prescription_dose = float(
                        item.TargetPrescriptionDose)
                roi.constraint.maximum_dose = float(item.TargetMaximumDose)
                roi.constraint.underdose_volume_fraction = float(
                        item.TargetUnderdoseVolumeFraction)
                self.targets.append(roi)

        self.constraints_loaded = True

    def get_dicom_dataset(self):
        '''
        Return pydicom.dataset.FileDataset object associated with this plan.
        '''

        self.load()
        return self.dicom_dataset

    def get_targets(self):
        '''
        Return list of ROIs identified in plan as targets.
        '''
        self.load()
        return self.targets

    def get_organs_at_risk(self):
        '''
        Return list of ROIs identified in plan as organs at risk.
        '''
        self.load()
        return self.organs_at_risk

    def get_dose_objective(self, objective='maximum_dose', idx_dose=0,
            dose=None):
        '''
        Obtain Dose object representing weight or objective of dose constraint.

        **Parameters:**
        objective : str, default='maximum_dose'
            Identifier of objective for which information is to be obtained.
            For a list of objectives, see skrt.dose.Constraint class.

        idx_dose : int, default=0
            Index of dose object in self.doses from which dose data are
            to be taken.

        dose : skrt.dose.Dose, default=None
            Dose object from which dose data are to be taken.  If specified,
            idx_dose is ignored.
        '''

        # Check that specified objective is known.
        if objective not in Constraint.get_weight_and_objectives():
            print(f'Unknown dose objective: \'{objective}\'')
            print(f'Known objectives: {Constraint.get_weight_and_objectives()}')
            return None


        # Return pre-existing result if available.
        dose_objective = getattr(self.objectives, objective)
        if dose_objective:
            return dose_objective

        # Check that data needed for defining dose objective are available.
        rois = self.get_targets()
        rois.extend(self.get_organs_at_risk())
        if not rois or not (self.doses or dose):
            return dose_objective

        # Initialise Dose object for objective data.
        if dose:
            dose_objective = Dose(dose)
        else:
            dose_objective = Dose(self.doses[idx_dose])

        dose_objective.load()
        dose_objective.data = np.zeros(dose_objective.data.shape)
        
        # Obtain objective information for each ROI.
        for roi in rois:
            if roi.constraint:
                roi_clone = skrt.structures.ROI(roi)
                roi_clone.set_image(dose_objective)
                mask = roi_clone.get_mask()
                dose_objective.data[mask > 0] = (
                        getattr(roi_clone.constraint, objective))

        setattr(self.objectives, objective, dose_objective)

        return dose_objective

    def get_approval_status(self):
        '''Return plan approval status.'''
        self.load()
        return self.approval_status

    def get_n_beam_seq(self):
        '''Return number of beam sequences for this plan.'''
        self.load()
        return self.n_beam_seq

    def get_n_fraction(self):
        '''Return number of fractions for this plan.'''
        self.load()
        return self.n_fraction

    def get_n_fraction_group(self):
        '''Return number of fraction groups for this plan.'''
        self.load()
        return self.n_fraction_group

    def get_name(self):
        '''Return plan name.'''
        self.load()
        return self.name

    def get_target_dose(self):
        '''Return dose to target (tumour) for this plan.'''
        self.load()
        return self.name

    def set_image(self, image):
        """Set associated image, initialising it if needed."""

        # Convert to Image object if needed
        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)

        # Assign image to self
        self.image = image

        # Assign self to the image
        if self.image is not None:
            self.image.add_plan(self)

    def set_structure_set(self, structure_set):
        """Set associated structure set, initialising it if needed."""

        # Convert to StructureSet object if needed
        if structure_set and not isinstance(
                structure_set, skrt.structures.StructureSet):
            structure_set = skrt.structures.StructureSet(structure_set)

        # Assign structure set to self
        self.structure_set = structure_set

        # Assign self to the structure set and its rois
        if self.structure_set is not None:
            self.structure_set.add_plan(self)

    def add_dose(self, dose):
        """Add a Dose object to be associated with this plan. This does not
        affect the plan associated with the Dose object.

        **Parameters:**

        dose : skrt.dose.Dose
            A Dose object to assign to this plan.
        """

        self.doses.append(dose)
        self.doses.sort()

    def clear_doses(self):
        """Clear all dose maps associated with this plan."""

        self.doses = []


class Constraint(Data):
    '''
    Container for data relating to a dose constraint.
    '''

    @classmethod
    def get_weight_and_objectives(cls):
        '''
        Return weight and objectives that may be associated with a constraint.
        '''
        return ['weight', 'minimum_dose', 'maximum_dose', 'full_volume_dose',
                'prescription_dose', 'underdose_volume_fraction',
                'overdose_volume_fraction']

    def __init__(self, opts={}, **kwargs):
        """
        Constructor of Container class.

        The following constraint attributes may be set via opts or
        **kwargs, but are otherwise initialised to None:

        - weight
        - minimum_dose
        - maximum_dose
        - full_volume_dose,
        - prescription_dose
        - underdose_volume_fraction
        - overdose_volume_fraction

        **Parameters:**

        opts: dict, default={}
            Dictionary to be used in setting instance attributes
            (dictionary keys) and their initial values.

        `**`kwargs
            Keyword-value pairs to be used in setting instance attributes
            and their initial values.
        """

        for attribute in Constraint.get_weight_and_objectives():
            setattr(self, attribute, None)

        super().__init__(opts, **kwargs)

