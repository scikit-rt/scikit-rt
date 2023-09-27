"""Classes related to doses and plans."""

import numbers
import functools

import matplotlib
import numpy as np
import pydicom

import skrt.core
import skrt.image
import skrt.structures


class ImageOverlay(skrt.image.Image):
    """Image that can be associated with another Image, and whose plotting
    functionality includes the ability to plot overlaid on its associated
    Image."""

    def __init__(self, path="", load=True, image=None, **kwargs):
        kwargs["default_intensity"] = kwargs.get("default_intensity", None)
        super().__init__(path=path, load=load, **kwargs)
        self.image = None
        self.set_image(image)

        # Default dose plotting settings
        self._default_cmap = "jet"
        self._default_colorbar_label = "Intensity"
        self._default_opacity = 0.5

    def set_image(self, image):
        """Set associated image, initialising it if needed."""

        if image and not isinstance(image, skrt.image.Image):
            image = skrt.image.Image(image)
        self.image = image

    def get_image(self):
        """Return associated image."""
        return self.image

    def plot(
        self,
        view=None,
        sl=None,
        idx=None,
        pos=None,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        colorbar=False,
        xlabel=None,
        ylabel=None,
        no_xlabel=False,
        no_ylabel=False,
        xticks=None,
        yticks=None,
        no_xticks=False,
        no_yticks=False,
        no_xtick_labels=False,
        no_ytick_labels=False,
        include_image=True,
        opacity=None,
        intensity=None,
        mpl_kwargs=None,
        show=True,
        mask=None,
        mask_threshold=0.5,
        masked=True,
        invert_mask=False,
        mask_color="black",
        **kwargs,
    ):
        """Plot this overlay, optionally overlaid on its associated Image.

        **Parameters**:

        view : str
            Orientation in which to compute the index. Can be "x-y", "x-z",
            "y-x", "y-z", "z-x", or
            "z-y".  If None, the initial view is chosen to match
            the image orienation.

        sl : int, default=None
            Slice number to plot. Takes precedence over <idx> and <pos> if not
            None. If all of <sl>, <idx>, and <pos> are None, the central
            slice will be plotted.

        idx : int, default=None
            Index of the slice in the array to plot. Takes precendence over
            <pos>.

        pos : float, default=None
            Position in mm of the slice to plot. Will be rounded to the nearest
            slice. Only used if <sl> and <idx> are both None.

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

        colorbar : int/bool, default=False
            Indicate whether to display colour bar(s):
            - 1 or True: colour bar for main image;
            - 2: colour bars for main image and for any associated image
            or overlay;
            - 0 or False: no colour bar.

        xlabel : str, default=None
            Custom label for the x axis.  If None, the label is set to
            be the coordinate axis and units, for example "x (mm)".
            If False or '', the x axis will not be labelled.

        ylabel : str, default=None
            Custom label for the y axis.  If None, the label is set to
            be the coordinate axis and units, for example "y (mm)".
            If False or '', the x axis will not be labelled.

        no_xlabel : bool, default=False
            If True, the x axis will not be labelled.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        xticks : tuple/dict, default=None
            Tick locations and custom tick labels for the x axis.

        yticks : tuple/dict, default=None
            Tick locations and custom tick labels for the y axis.

        no_xticks : bool, default=False
            If True, ticks (and their labels) on the x axis will not be shown.

        no_yticks : bool, default=False
            If True, ticks (and their labels) on the y axis will not be shown.

        no_xtick_labels : bool, default=False
            If True, ticks on the x axis will not be labelled.

        no_ytick_labels : bool, default=False
            If True, ticks on the y axis will not be labelled.

        show : bool, default = True
            If True, the plot will be displayed immediately.

        mask : Image/list/ROI/str/StructureSet, default=None
            Image object representing a mask or a source from which
            an Image object can be initialised.  In addition to the
            sources accepted by the Image constructor, the source
            may be an ROI, a list of ROIs or a StructureSet.  In the
            latter cases, the mask image is derived from the ROI mask(s).

        mask_threshold : float, default=0.5
            Threshold for mask data.  Values above and below this value are
            set to True and False respectively.  Taken into account only
            if the mask image has non-boolean data.

        masked : bool, default=True
            If True and a mask is specified, the image is masked.

        invert_mask : bool, default=False
            If True and a mask is applied, the mask will be inverted.

        mask_color : matplotlib color, default="black"
            color in which to plot masked areas.

        `**`kwargs:
            Keyword args to pass to skrt.image.Image.plot().
        """

        # Set up axes
        if not view:
            view = self.get_orientation_view()
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        self.load()

        # Plot underlying image, always without title.
        if include_image and self.image is not None:
            self.image.plot(
                view,
                sl=sl,
                idx=idx,
                pos=pos,
                ax=self.ax,
                show=False,
                title="",
                colorbar=max((colorbar - 1), 0),
                xlabel=xlabel,
                ylabel=ylabel,
                no_xlabel=no_xlabel,
                no_ylabel=no_ylabel,
                xticks=xticks,
                yticks=yticks,
                no_xticks=no_xticks,
                no_yticks=no_yticks,
                no_xtick_labels=no_xtick_labels,
                no_ytick_labels=no_ytick_labels,
                mask=mask,
                mask_threshold=mask_threshold,
                masked=masked,
                invert_mask=invert_mask,
                mask_color=mask_color,
            )

            # Use default transprency if plotting image and opacity is None
            if opacity is None:
                opacity = self._default_opacity

        # Add opacity to mpl_kwargs
        if mpl_kwargs is None:
            mpl_kwargs = {}
        if opacity is not None:
            mpl_kwargs["alpha"] = opacity

        # Add intensity to mpl_kwargs
        if mpl_kwargs is None:
            mpl_kwargs = {}
        if intensity is not None:
            mpl_kwargs["vmin"] = intensity[0]
            mpl_kwargs["vmax"] = intensity[1]

        # Plot self
        skrt.image.Image.plot(
            self,
            view=view,
            sl=sl,
            idx=idx,
            pos=pos,
            ax=self.ax,
            zoom=zoom,
            colorbar=colorbar,
            mpl_kwargs=mpl_kwargs,
            show=show,
            xlabel=xlabel,
            ylabel=ylabel,
            no_xlabel=no_xlabel,
            no_ylabel=no_ylabel,
            xticks=xticks,
            yticks=yticks,
            no_xticks=no_xticks,
            no_yticks=no_yticks,
            no_xtick_labels=no_xtick_labels,
            no_ytick_labels=no_ytick_labels,
            mask=mask,
            mask_threshold=mask_threshold,
            masked=masked,
            invert_mask=invert_mask,
            mask_color=mask_color,
            **kwargs,
        )

    @functools.cached_property
    def max(self):
        """Return maximum data value."""
        return self.get_data().max()


class Dose(ImageOverlay):
    """
    Class representing a radiotherapy dose field.  This is the
    same as the Image class, but with overridden
    plotting behaviour, and with extra functionality relating to ROIs.

    Attributes of a Dose object should usually be accessed via
    their getter methods, rather than directly, to ensure that
    attribute values have been loaded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure linking between image and dose
        self.set_image(self.image)

        # Plot settings specific to dose map
        self._default_cmap = "jet"
        self._default_colorbar_label = "Dose (Gy)"
        self._default_opacity = 0.5

        # Delete spurious attributes inherited from Image class
        for attribute in ["doses", "plans"]:
            if hasattr(self, attribute):
                delattr(self, attribute)

        # Initialise additional attributes
        self.dose_units = None
        self.dose_type = None
        self.dose_summation_type = None
        self.n_fraction = None
        self._equivalent_dose = None
        self.plan = None

    def load(self, *args, **kwargs):
        """Load self and set default maximum plotting intensity from max of
        data array if not yet set."""

        skrt.image.Image.load(self, *args, **kwargs)
        self._default_vmax = self.data.max()
        dset = self.dicom_dataset
        if dset:
            self.dose_units = getattr(dset, "DoseUnits", None)
            self.dose_type = getattr(dset, "DoseType", None)
            self.dose_summation_type = getattr(dset, "DoseSummationType", None)

    def copy_dicom(self, *args, **kwargs):
        """
        Copy (single) source dicom file.

        The present class (skrt.dose.Dose) inherits from skrt.image.Image,
        which in turn inherits from skrt.core.Dated.  This method
        forces calling of the copy_dicom() method of the latter.
        For details of parameters, see documentation for:
        skrt.core.Dated.copy_dicom().
        """
        skrt.core.Dated.copy_dicom(self, *args, **kwargs)

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

    def get_dose_units(self):
        """
        Get unit of dose.

        Possible values include:

        - "GY": Gray;
        - "RELATIVE": dose relative to a reference dose.

        For details, see:

        - https://dicom.innolitics.com/ciods/rt-dose/rt-dose/30040002
        """
        self.load()
        return self.dose_units

    def get_dose_type(self):
        """
        Get dose type.

        Possible values are:

        - "PHYSICAL": physical dose;
        - "EFFECTIVE": physical dose adjusted for a given biological effect;
        - "ERROR": difference between ideal and planned dose.

        For details, see:

        - https://dicom.innolitics.com/ciods/rt-dose/rt-dose/30040004
        """
        self.load()
        return self.dose_type

    def get_dose_summation_type(self):
        """
        Get type of dose summation.

        Possible values include:

        - "PLAN": dose summed over all fraction groups;
        - "MULTI_PLAN": dose summed over two or more plans;
        - "FRACTION": dose for a single fraction group;
        - "BEAM": dose for one or more beams.

        For details, see:

        - https://dicom.innolitics.com/ciods/rt-dose/rt-dose/3004000a
        """
        self.load()
        return self.dose_summation_type

    def get_n_fraction(self):
        """Return number of dose fractions."""
        if self.n_fraction is None and hasattr(self, "plan"):
            self.n_fraction = self.plan.get_n_fraction()
        return self.n_fraction

    def get_dose_in_roi_3d(self, roi, standardise=True):
        """Return copy of the dose array that has values retained
        inside an ROI, and set to zero elsewhere.  Fails if the
        ROI and dose arrays are not the same size."""

        roi.create_mask()
        if not self.has_same_geometry(roi.mask, standardise=standardise):
            roi = roi.clone()
            roi.set_image(self)

        dose_in_roi = self.get_data(standardise=standardise) * roi.get_mask(
            standardise=standardise
        )
        return dose_in_roi

    def get_dose_in_roi(self, roi, standardise=True):
        """Return 1D numpy array containing all of the dose values for the
        voxels inside an ROI. Fails if the ROI and dose arrays are not the
        same size."""
        dose_in_roi = self.get_dose_in_roi_3d(roi, standardise)
        return dose_in_roi[dose_in_roi > 0]

    def get_max_dose_in_rois(self, rois=None):
        """
        Return maximum dose in a set of rois.

        **Parameter:**

        rois : list, default=None
            List of ROI objects, for which maximum dose is to be determined.
        """
        # Determine the maximum dose for the input roi(s).
        dose_max = 0
        if rois:
            for roi in rois:
                doses = list(self.get_dose_in_roi(roi))
                doses.append(dose_max)
                dose_max = max(doses)
        return dose_max

    def plot_dvh(
        self,
        rois=None,
        bins=50,
        dose_min=0,
        dose_max=None,
        figsize=(8, 4),
        lw=2,
        n_colour=None,
        cmap="turbo",
        grid=True,
        fname=None,
        legend_bbox_to_anchor=(1.01, 0.5),
        legend_loc="center left",
    ):
        """
        Plot dose-volume histogram for specified ROI(s).

        **Parameters:**

        rois : ROI/StructureSet/list, default=None
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

        n_colour : int, default=None
            Number of colours over which to cycle when drawing histograms
            based on a colour map.  In None, the ROI color attribute
            is used to define histogram colour.

        cmap : str, default='turbo'
            Name of a matplotlib colour map from which to define set of
            colours over which to cycle when drawing histograms.

            For predfined names see:

            https://matplotlib.org/stable/gallery/color/colormap_reference.html

        grid : bool, default=True
            If True, overlay a grid on the dose-volume histogram.

        fname : str, default=None
            Name of file for saving output.
        """

        # Obtain a list containing all unique rois.
        all_rois = skrt.structures.get_all_rois(rois)

        # Determine the maximum dose for the input roi(s).
        if dose_max is None:
            dose_max = self.get_max_dose_in_rois(all_rois)

        # Create figure and define the list of colours.
        axes = matplotlib.pyplot.subplots(figsize=figsize)[1]
        if n_colour is None:
            colours = [roi.color for roi in all_rois]
        else:
            colours = matplotlib.cm.get_cmap(cmap)(np.linspace(0, 1, n_colour))
        axes.set_prop_cycle(color=colours)

        # Plot the dose-volume histograms, and extract information for legend.
        lines = []
        labels = []
        for roi in all_rois:
            doses = self.get_dose_in_roi(roi)
            bins, patches = axes.hist(
                doses,
                bins=bins,
                range=(dose_min, dose_max),
                lw=lw,
                histtype="step",
                density=True,
                cumulative=-1,
            )[1:]
            colour = patches[0].get_edgecolor()
            lines.append(matplotlib.lines.Line2D([0], [0], color=colour, lw=lw))
            labels.append(roi.name)

        # Label the axes, add grid and legend, tighten layout.
        axes.set_xlabel("Dose (Gy)")
        axes.set_ylabel("Volume fraction")
        axes.grid(grid)
        axes.legend(
            handles=lines,
            labels=labels,
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
        )
        matplotlib.pyplot.tight_layout()

        # Show figure or save to file.
        if fname is not None:
            matplotlib.pyplot.savefig(fname)
        else:
            matplotlib.pyplot.show()

        return axes

    def plot_DVH(self, *args, **kwargs):
        """Alias for plot_dvh()."""
        self.plot_dvh(*args, **kwargs)

    def get_mean_dose(self, roi):
        """Get mean dose inside an ROI."""

        doses = self.get_dose_in_roi(roi)
        if doses.size:
            return np.mean(doses)
        if roi.get_volume(method="mask"):
            return 0
        return None

    def get_dose_quantile(self, roi, quantile=0.5):
        """Get specified dose quantile inside an ROI."""

        doses = self.get_dose_in_roi(roi)
        if doses.size:
            return np.quantile(doses, quantile)
        return None

    def get_bed(self, *args, **kwargs):
        """
        Alias for skrt.dose.Dose method get_biologically_effective_dose().

        See aliased method for documentation.
        """
        self._equivalent_dose = "get_bed()"
        return self.get_biologically_effective_dose(*args, **kwargs)

    def get_biologically_effective_dose(
        self,
        rois=None,
        alpha_beta_ratios=None,
        n_fraction=None,
        fill=0,
        standardise=False,
        force=False,
    ):
        """
        Get a Dose object where array values are biologically effective dose.

        The biologically effective dose, BED, for a volume element
        characterised by linear and quadratic coefficients alpha and beta
        for the linear-quadratic equation, is related to the physical dose,
        D, delivered over n equal fractions as:
            BED = D * (1 + (D/n) / (alpha/beta)).

        **Parameters:**

        rois : ROI/StructureSet/list, default=None
            Object(s) specifying ROIs for which calculation is to be
            performed.  The object(s) can be a single
            skrt.structures.ROI object, a single skrt.structures.StructureSet
            object, or a list containing any combination of ROI
            and StructureSet objects.

        alpha_beta_ratios : dict, default=None
            By default, values of alpha over beta used are the values
            set for individual ROIs, for example using
            skrt.StructureSet.set_alpha_beta_ratios().  The default
            values can be overridden here by passing a dictionary
            where keys are ROI names and values are revised values
            for alpha over beta.

        n_fraction : float, default=None
            Number of equal-dose fractions over which physical total dose
            is delivered.  If None, the value returned by
            self.get_n_fraction() is used.

        fill : float/None, default=0
            Specification of dose value to set outside of ROIs, and for ROIs
            where alpha over beta has a null value.  If a float, this is
            used as the default dose value.  If None, physical-dose values
            are retained.

        standardise : bool, default=False
            If False, the data arrays for Doses and for ROI masks will be
            kept in the orientation in which they were loaded; otherwise,
            they will be converted to standard dicom-style orientation,
            such that [column, row, slice] corresponds to the [x, y, z] axes.

        force : bool, default=False
            Determine action if the current dose object has
            dose_type set to "EFFECTIVE".  If False, write an error message,
            and don't apply the formula for recalculating dose.  If
            True, apply the formula, even if the dose_type suggests that
            dose values have already been corrected for biological effect.
        """
        self._equivalent_dose = "get_biologically_effective_dose()"
        return self.get_equivalent_dose(
            dose_rate=0,
            rois=rois,
            alpha_beta_ratios=alpha_beta_ratios,
            n_fraction=n_fraction,
            fill=fill,
            standardise=standardise,
            force=force,
        )

    def get_eqd(self, *args, **kwargs):
        """
        Alias for skrt.dose.Dose method get_equivalent_dose().

        See aliased method for documentation.
        """
        self._equivalent_dose = "get_eqd()"
        return self.get_equivalent_dose(*args, **kwargs)

    def get_equivalent_dose(
        self,
        dose_rate=2,
        rois=None,
        alpha_beta_ratios=None,
        n_fraction=None,
        fill=0,
        standardise=False,
        force=False,
    ):
        """
        Get a Dose object where array values are equivalent dose
        for a specified dose per fraction.

        If a physical dose, D, is delivered over n equal fractions to a
        volume element characterised by linear and quadratic coefficients
        alpha and beta for the linear-quadratic equation, the total equivalent
        dose, EQDd, at a dose rate d per fraction, is:
            EQDd = D * ((D/n) + (alpha/beta)) / (d + (alpha/beta)).
        It approaches the biologically effective dose, BED,
        as d tends to zero:
            EQD0 = D * (1 + (D/n) / (alpha/beta)) = BED.

        **Parameters:**

        dose_rate : float, default=2
            Dose (Gy) per fraction for which calculation is to be determined.

        rois : ROI/StructureSet/list, default=None
            Object(s) specifying ROIs for which calculation is to be
            performed.  The object(s) can be a single
            skrt.structures.ROI object, a single skrt.structures.StructureSet
            object, or a list containing any combination of ROI
            and StructureSet objects.

        alpha_beta_ratios : dict, default=None
            By default, values of alpha over beta used are the values
            set for individual ROIs, for example using
            skrt.StructureSet.set_alpha_beta_ratios().  The default
            values can be overridden here by passing a dictionary
            where keys are ROI names and values are revised values
            for alpha over beta.

        n_fraction : float, default=None
            Number of equal-dose fractions over which physical total dose
            is delivered.  If None, the value returned by
            self.get_n_fraction() is used.

        fill : float/None, default=0
            Specification of dose value to set outside of ROIs, and for ROIs
            where alpha over beta has a null value.  If a float, this is
            used as the default dose value.  If None, physical-dose values
            are retained.

        standardise : bool, default=False
            If False, the data arrays for Doses and for ROI masks will be
            kept in the orientation in which they were loaded; otherwise,
            they will be converted to standard dicom-style orientation,
            such that [column, row, slice] corresponds to the [x, y, z] axes.

        force : bool, default=False
            Determine action if the current dose object has
            dose_type set to "EFFECTIVE".  If False, write an error message,
            and don't apply the formula for recalculating dose.  If
            True, apply the formula, even if the dose_type suggests that
            dose values have already been corrected for biological effect.
        """
        # Ensure that a method name is set.
        if self._equivalent_dose is None:
            self._equivalent_dose = "get_equivalent_dose()"

        # Check input values.
        if not isinstance(fill, numbers.Number) and fill is not None:
            self.logger.error(
                "Value of fill set to '%s' but must be a number or None.", fill
            )
            self._equivalent_dose = None
            return None

        self.load()
        if "EFFECTIVE" == self.dose_type and not force:
            self.logger.error(
                "Method '%s' called for dose with type '%s'.  "
                "If this is the intention, repeat call with 'force=True'",
                self._equivalent_dose,
                self.dose_type,
            )
            self._equivalent_dose = None
            return None

        rois = skrt.structures.get_all_rois(rois)
        if not rois:
            self.logger.error(
                "Method %s called without specifying any ROIs.",
                self._equivalent_dose,
            )
            self._equivalent_dose = None
            return None

        if not n_fraction:
            n_fraction = self.get_n_fraction()

        if not n_fraction:
            self.logger.error(
                "Method %s called without specifying number of fractions.",
                self._equivalent_dose,
            )
            self._equivalent_dose = None
            return None

        # Set default dose to be fill value.
        if isinstance(fill, numbers.Number):
            dose = Dose(
                path=(
                    fill * np.ones(self.get_data(standardise=standardise).shape)
                ),
                affine=self.get_affine(standardise=standardise),
            )
        # Set default dose to be initial dose.
        elif fill is None:
            dose = Dose(
                path=self.get_data(standardise=standardise),
                affine=self.get_affine(standardise=standardise),
            )

        # Calculate equivalent dose inside ROIs with alpha/beta defined.
        for roi in rois:
            # For current ROI, use alpha/beta given in input,
            # or use the value assigned to the ROI.
            alpha_over_beta = alpha_beta_ratios.get(
                roi.name, roi.alpha_over_beta
            )
            # If alpha/beta is defined,
            # calculate the equivalent dose.
            if alpha_over_beta is not None:
                dose_in_roi = self.get_dose_in_roi_3d(
                    roi, standardise=standardise
                )
                dose_in_roi *= (alpha_over_beta + dose_in_roi / n_fraction) / (
                    dose_rate + alpha_over_beta
                )
                dose.data[dose_in_roi > 0] = dose_in_roi[dose_in_roi > 0]

        # Reset the default maximum plotting intensity
        # from the maximum of the data array.
        setattr(dose, "_default_vmax", dose.data.max())

        # Set dose_type to indicate that biological effect
        # is taken into account.
        dose.dose_type = "EFFECTIVE"

        # Unset method name - it will be reset by the next caller.
        self._equivalent_dose = None
        return dose


class Plan(skrt.core.Archive):
    """
    Class representing a radiotherapy plan.

    Attributes of a Plan object should usually be accessed via
    their getter methods, rather than directly, to ensure that
    attribute values have been loaded.
    """

    def __init__(self, path="", load=True):
        """
        Instantiate Plan from a DICOM rtplan file.

        **Parameters:**

        path: str/pathlib.Path, default=""
            Path to DICOM rtplan file from which plan is to be instantiated.

        load: bool, default=True
            If True, load plan data immediately.  If False, plan data will
            be loaded automatically the first time that a Plan method
            requiring the data is called, or may be explicitly loaded
            by calling the load() method.
        """
        self.loaded = False
        self.path = None
        self.name = None
        self.description = None
        self.prescription_description = None
        self.approval_status = None
        self.n_fraction_group = None
        self.n_beam_seq = None
        self.n_fraction = None
        self.organs_at_risk = None
        self.machine = None
        self.station_name = None
        self.target_dose = None
        self.targets = None
        self.image = None
        self.structure_set = None
        self.doses = []
        self.objectives = skrt.core.Data()
        for constraint_attribute in Constraint.get_weight_and_objectives():
            setattr(self.objectives, constraint_attribute, None)

        super().__init__(path)

        self.constraints_loaded = False
        if load:
            self.load()

    def load(self, force=False):
        """
        Load plan data.

        **Parameter:**

        force: bool, default=False
            If True, plan data will be reloaded from file, even if
            loaded previously.
        """
        if self.loaded and not force:
            return

        self.dicom_dataset = pydicom.dcmread(self.path, force=True)

        self.name = getattr(self.dicom_dataset, "RTPlanName", None)
        self.description = getattr(
            self.dicom_dataset, "RTPlanDescription", None
        )
        self.prescription_description = getattr(
            self.dicom_dataset, "PrescriptionDescription", None
        )
        self.station_name = getattr(self.dicom_dataset, "StationName", None)

        try:
            self.approval_status = self.dicom_dataset.ApprovalStatus
        except AttributeError:
            self.approval_status = None

        try:
            self.n_fraction_group = len(
                self.dicom_dataset.FractionGroupSequence
            )
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
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        if hasattr(dose, "TargetPrescriptionDose"):
                            if self.target_dose is None:
                                self.target_dose = 0.0
                            self.target_dose += dose.TargetPrescriptionDose
        dose_reference_sequence = getattr(
            self.dicom_dataset, "DoseReferenceSequence", []
        )

        if self.target_dose is None:
            target_dose = 0.0
            for dose_reference in dose_reference_sequence:
                target_dose = max(
                    target_dose,
                    getattr(dose_reference, "TargetPrescriptionDose", 0),
                )
            if target_dose:
                self.target_dose = target_dose

        self.loaded = True
        self.load_constraints()

    def load_constraints(self, force=False):
        """
        Load dose constraints from plan.

        **Parameter:**
        force: bool, default=False
            If True, dose constraints will be reloaded from file, even if
            loaded previously.
        """
        if (self.constraints_loaded and not force) or not self.structure_set:
            return

        rois = {}
        for roi in self.structure_set.get_rois():
            rois[roi.number] = roi

        self.organs_at_risk = []
        self.targets = []

        dset = self.get_dicom_dataset()

        dose_reference_sequence = getattr(dset, "DoseReferenceSequence", [])

        for item in dose_reference_sequence:
            roi = rois.get(getattr(item, "ReferencedROINumber", None), None)
            if roi is None:
                continue
            roi.roi_type = item.DoseReferenceType
            roi.constraint = Constraint()
            roi.constraint.weight = skrt.core.get_float(
                item, "ConstraintWeight"
            )
            if item.DoseReferenceType in ["ORGAN_AT_RISK", "OAR"]:
                roi.constraint.maximum_dose = skrt.core.get_float(
                    item, "OrganAtRiskMaximumDose"
                )
                roi.constraint.full_volume_dose = skrt.core.get_float(
                    item, "OrganAtRiskFullVolumeDose"
                )
                roi.constraint.overdose_volume_fraction = skrt.core.get_float(
                    item, "OrganAtRiskOverdoseVolumeFraction"
                )
                self.organs_at_risk.append(roi)
            elif item.DoseReferenceType in ["TARGET"]:
                roi.constraint.minimum_dose = skrt.core.get_float(
                    item, "TargetMinimumDose"
                )
                roi.constraint.prescription_dose = skrt.core.get_float(
                    item, "TargetPrescriptionDose"
                )
                roi.constraint.maximum_dose = skrt.core.get_float(
                    item, "TargetMaximumDose"
                )
                roi.constraint.underdose_volume_fraction = skrt.core.get_float(
                    item, "TargetUnderdoseVolumeFraction"
                )
                self.targets.append(roi)

        self.constraints_loaded = True

    def get_dicom_dataset(self):
        """
        Return pydicom.dataset.FileDataset object associated with this plan.
        """
        self.load()
        return self.dicom_dataset

    def get_targets(self):
        """
        Return list of ROIs identified in plan as targets.
        """
        self.load()
        return self.targets

    def get_organs_at_risk(self):
        """
        Return list of ROIs identified in plan as organs at risk.
        """
        self.load()
        return self.organs_at_risk

    def get_dose_objective(
        self, objective="maximum_dose", idx_dose=0, dose=None
    ):
        """
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
        """
        # Check that specified objective is known.
        if objective not in Constraint.get_weight_and_objectives():
            print(f"Unknown dose objective: '{objective}'")
            print(f"Known objectives: {Constraint.get_weight_and_objectives()}")
            return None

        # Return pre-existing result if available.
        dose_objective = getattr(self.objectives, objective, None)
        if dose_objective:
            return dose_objective

        # Check that data needed for defining dose objective are available.
        rois = self.get_targets()
        if rois:
            rois.extend(self.get_organs_at_risk())
        else:
            rois = self.get_organs_at_risk()
        if not rois or not (self.doses or dose):
            return dose_objective

        # Initialise Dose object for objective data.
        if dose:
            dose_objective = Dose(dose)
        else:
            dose_objective = Dose(self.doses[idx_dose])

        dose_objective.load()
        dose_objective.data = (-1) * np.ones(dose_objective.data.shape)

        # Obtain objective information for each ROI.
        for roi in rois:
            if roi.constraint:
                roi_clone = skrt.structures.ROI(roi)
                roi_clone.set_image(dose_objective)
                mask = roi_clone.get_mask()
                value = getattr(roi_clone.constraint, objective)
                if value is None:
                    continue
                if "maximum_dose" == objective:
                    dose_objective.data[np.where(
                        (mask > 0) & ((dose_objective.data < 0) |
                                      (dose_objective.data > value)))] = value
                elif "minimum_dose" == objective:
                    dose_objective.data[np.where(
                        (mask > 0) & ((dose_objective.data < 0) |
                                      (dose_objective.data < value)))] = value
                else:
                    dose_objective.data[mask > 0] = value

            dose_objective.data = np.ma.masked_where(
                    dose_objective.data < 0, dose_objective.data)
        setattr(self.objectives, objective, dose_objective)

        return dose_objective

    def get_approval_status(self):
        """Return plan approval status."""
        self.load()
        return self.approval_status

    def get_description(self):
        """Return plan description."""
        self.load()
        return self.description

    def get_station_name(self):
        """Return station name."""
        self.load()
        return self.station_name

    def get_machine(self, machines=None):
        """
        Return machine name.

        **Parameter:**

        machines: dict, default=None
            Dictionary mapping between station names and machine names.
        """
        return skrt.image.Image.get_machine(self, machines)

    def get_prescription_description(self):
        """Return plan prescription description."""
        self.load()
        return self.prescription_description

    def get_n_beam_seq(self):
        """Return number of beam sequences for this plan."""
        self.load()
        return self.n_beam_seq

    def get_n_fraction(self):
        """Return number of fractions for this plan."""
        self.load()
        return self.n_fraction

    def get_n_fraction_group(self):
        """Return number of fraction groups for this plan."""
        self.load()
        return self.n_fraction_group

    def get_name(self):
        """Return plan name."""
        self.load()
        return self.name

    def get_target_dose(self):
        """Return dose to target (tumour) for this plan."""
        self.load()
        return self.target_dose

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

    def get_image(self):
        """Return Image object associated with this plan."""
        return self.image

    def set_structure_set(self, structure_set):
        """Set associated structure set, initialising it if needed."""

        # Convert to StructureSet object if needed
        if structure_set and not isinstance(
            structure_set, skrt.structures.StructureSet
        ):
            structure_set = skrt.structures.StructureSet(structure_set)

        # Assign structure set to self
        self.structure_set = structure_set

        # Assign self to the structure set and its rois
        if self.structure_set is not None:
            self.structure_set.add_plan(self)

    def get_structure_set(self):
        """Return StructureSet object associated with this plan."""
        return self.structure_set

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

    def get_doses(self):
        """Return Dose objects associated with this plan."""
        return self.doses


class Constraint(skrt.core.Data):
    """
    Container for data relating to a dose constraint.
    """

    @classmethod
    def get_weight_and_objectives(cls):
        """
        Return weight and objectives that may be associated with a constraint.
        """
        return [
            "weight",
            "minimum_dose",
            "maximum_dose",
            "full_volume_dose",
            "prescription_dose",
            "underdose_volume_fraction",
            "overdose_volume_fraction",
        ]

    def __init__(self, opts=None, **kwargs):
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

        opts: dict, default=None
            Dictionary to be used in setting instance attributes
            (dictionary keys) and their initial values.

        `**`kwargs
            Keyword-value pairs to be used in setting instance attributes
            and their initial values.
        """

        for attribute in Constraint.get_weight_and_objectives():
            setattr(self, attribute, None)

        super().__init__(opts or {}, **kwargs)


def remove_duplicate_doses(doses=None):
    """
    Remove duplicates from a list of dose objects.

    Dose instance dose1 is taken to be a duplicate of dose2 if
    dose1.has_same_data(dose2) is True.

    **Parameter:**
    doses: list, default=None
        List of dose objects, from which duplicates are to be removed.
    """
    return skrt.image.remove_duplicate_images(doses)


def sum_doses(doses=None):
    """
    Sum doses of the same geometry.

    If not all doses have the same same geometry (shape, origin,
    voxel size), None is returned.  Otherwise, a Dose object is
    returned that has the same geometry as the input doses, and
    where array values are the sums of the array values of
    the input doses.

    **Parameter:**
    doses: list, default=None
        List of Dose objects to be summed.
    """
    return skrt.image.sum_images(doses)
