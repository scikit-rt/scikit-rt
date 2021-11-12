"""New version of QuickViewer that uses the scikit-rt Image class."""

import ipywidgets as ipyw
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


from skrt.core import is_list
from skrt.image import (
    Image, 
    _axes, 
    _slice_axes, 
    _plot_axes,
    _default_figsize
)

# ipywidgets settings
_style = {'description_width': 'initial'}


class BetterViewer:
    '''Display multiple SingleViewers and/or comparison images.'''

    def __init__(
        self,
        nii=None,
        title=None,
        mask=None,
        dose=None,
        structs=None,
        multi_structs=None,
        jacobian=None,
        df=None,
        share_slider=True,
        orthog_view=False,
        plots_per_row=None,
        match_axes=None,
        scale_in_mm=True,
        show_cb=False,
        show_overlay=False,
        show_diff=False,
        comparison=None,
        comparison_only=False,
        cb_splits=2,
        overlay_opacity=0.5,
        overlay_legend=False,
        legend_loc='lower left',
        translation=False,
        translation_file_to_overwrite=None,
        show_mse=False,
        dta_tolerance=5,
        dta_crit=1,
        diff_crit=15,
        suptitle=None,
        show=True,
        **kwargs,
    ):
        '''
        Display one or more interactive images.

        Parameters
        ----------

        nii : string/nifti/array/list, default=None
            Source of image data for each plot. If multiple plots are to be
            shown, this must be a list. Image sources can be any of:
            (a) The path to a NIfTI or DICOM file;
            (b) A nibabel.nifti1.Nifti1Image object;
            (c) The path to a file containing a NumPy array;
            (d) A NumPy array.

        title : string or list of strings, default=None
            Custom title(s) to use for the image(s) to be displayed. If the
            number of titles given, n, is less than the number of images, only
            the first n figures will be given custom titles. If any titles are
            None, the name of the image file will be used as the title.

        mask : string/nifti/array/list, default=None
            Source(s) of array(s) to with which to mask each plot (see valid
            image sources for <nii>).

        dose : string/nifti/array/list, default=None
            Source(s) of dose field array(s) to overlay on each plot (see valid
            image sources for <nii>).

        structs : str/list/dict, default=None
            Locations of files from which to load structures masks. This
            argument can be any of:

            1) String:
                a) The path to a NIfTI file containing a structure mask;
                b) The path to a DICOM file containing structure contours;
                b) A wildcard matching one or more of the above file types;
                c) The path to a directory containing NIfTI or DICOM files;
                d) A wildcard matching one or more directories containing 
                   NIfTI or DICOM files.

                If the string is found to match a directory, the structures 
                from all NIfTI or DICOM files inside that directory will 
                be loaded.

                Structure names will be inferred from the filenames (NIfTI)
                or the structure names inside the file (DICOM) unless
                the user indicates otherwise in the <struct_names> parameter;
                e.g. a structure taken from a file called
                'right_parotid.nii.gz' would automatically be called
                'right parotid' in QuickViewer.

                If multiple loaded structures have the same names, QuickViewer
                will attempt to label each with a unique name in the UI:
                    - If two structures named 'heart' are loaded from different
                      directories dir1 and dir2, these will be labelled
                      'Heart (dir1)' and 'Heart (dir2) in the UI.
                    -  If two structures named 'heart' are loaded from
                       different files, file1.nii and file2.nii, these will be
                       labelled 'Heart (file1.nii)' and 'Heart (file2.nii)' in
                       the UI.
                However, if the <struct_legend> option is used, the structures
                will be labelled with the same name in the figure legend. See
                the labelling option below in part (3) or the <struct_names>
                option for more customisation.

            2) List:
                a) A list of any of the strings described above; all structure
                   files found will be loaded.
                b) A list of pairs of paths to files containing structures to
                   be compared to one another (see the <struct_comparison>
                   option).

            3) Dict:
                Structure filepaths can be nested inside a dictionary, where
                the keys are labels which the user wishes to use to refer to
                structures in those files, and the values are any of the
                options listed above (except 2b).

                The label will be displayed in parentheses next to the
                structure names in the QuickViewer UI and structure legend.

                The <struct_names> and <struct_options> arguments can also
                be nested inside a dictionary if the user wants to apply
                different name and color options to the structures associated
                with different labels.

            By default, each NIfTI file will be assumed to contain a single 
            structure. To load multiple label masks from a single NIfTI file,
            add the string 'multi:' before the filepath, e.g.:

                structs='multi:my_file.nii'

            or alternatively use the multi_structs parameter.

        multi_structs : str/list/dict, default=None

            Path(s) to file(s) from which to load multiple structure label 
            masks per file. 

            Same as the <structs> argument, except each file specified in the
            <multi_structs> argument will be checked for different labels 
            instead of being treated as a binary mask.

            This can be used in conjunction with <structs> to load single masks
            from some files and multiple masks from others.

        jacobian : string/nifti/array/list, default=None
            Source(s) of jacobian determinant array(s) to overlay on each plot
            (see valid image sources for <nii>).

        df : string/nifti/array/list, default=None
            Source(s) of deformation field(s) to overlay on each plot
            (see valid image sources for <nii>).

        share_slider : bool, default=True
            If True and all displayed images are in the same frame of
            reference, a single slice slider will be shared between all plots.
            If plots have different frames of reference, this option will be
            ignored.

        orthog_view : bool, default=False
            If True, an orthgonal view with an indicator line showing the
            current slice will be displayed alongside each plot.

        plots_per_row : int, default=None
            Number of plots to display before starting a new row. If None,
            all plots will be shown on a single row.

        match_axes : int/str, default=None
            Method for adjusting axis limits. Can either be:
            - An integer n, where 0 < n < number of plots, or n is -1. The axes
              of all plots will be adjusted to be the same as those of plot n.
            - 'all'/'both': axes for all plots will be adjusted to cover the
              maximum range of all plots.
            - 'overlap': axes for all plots will be adjusted to just show the
              overlapping region.
            - 'x': same as 'all', but only applied to the current x axis.
              whatever the x axis is in the current view.
            - 'y': same as 'all', but only applied to the current y axis.

        scale_in_mm : bool, default=True
            If True, the axis scales will be shown in mm instead of array
            indices.

        show_cb : bool, default=False
            If True, a chequerboard image will be displayed. This option will
            only be applied if the number of images in <nii> is 2.

        show_overlay : bool, default=False
            If True, a blue/red transparent overlaid image will be displayed.
            This option will only be applied if the number of images in
            <nii> is 2.

        show_diff : bool, default=False
            If True, a the difference between two images will be shown. This
            option will only be applied if the number of images in <nii>
            is 2.

        comparison : bool/str/list, default=None
            Indicator for which type(s) of comparison image(s) to show. If
            True or 'all', a comparison image control by a dropdown menu with all
            comparison options will be loaded. Can also be a single string or
            list containing any combination of 'all', 'chequerboard',
            'overlay', and 'difference' to plot those images in the desired order.

        comparison_only : bool, False
            If True, only comparison images (overlay/chequerboard/difference)
            will be shown. If no comparison options are selected, this
            parameter will be ignored.

        cb_splits : int, default=2
            Number of sections to show for chequerboard image. Minimum = 1
            (corresponding to no chequerboard). Can later be changed
            interactively.

        overlay_opacity : float, default=0.5
            Initial opacity of overlay. Can later be changed interactively.

        overlay_legend : bool default=False
            If True, a legend will be displayed on the overlay plot.

        legend_loc : str, default='lower left'
            Location for any legends being displayed. Must be a valid
            matplotlib legend location.

        translation : bool, default=False
            If True, widgets will be displayed allowing the user to apply a
            translation to the image and write this to an elastix transform
            file or plain text file.

        translation_file_to_overwrite : str, default=None
            If not None and the <translation> option is used, this parameter
            will be used to populate the 'Original' and 'Output' file fields in
            the translation user interface.

        show_mse : str, default=None
            Color for annotation of mean-squared-error if using comparison
            images. If None, no annotation will be added. Can be set to 'True'
            to use default colour (white).

        dta_tolerance : float, default=5
            Tolerance to use when computing distance-to-agreement.

        dta_crit : float, default=1
            Critical value of distance-to-agreement to use when computing
            gamma index.

        diff_crit : float, default=15
            Critical value of difference to use when computing gamma index.

        suptitle : string, default=None
            Global title for all subplots. If None, no suptitle will be added.

        show : bool, default=True
            If True, the plot will be displayed when the QuickViewer object is
            created. Otherwise, the plot can be displayed later via
            QuickViewer.show().

        Keyword arguments
        -----------------

        timeseries : str/list/dict, default=None
            A series of image files taken at difference dates. This can be:
            (a) The path to a directory containing multiple image files/
                multiple directories each containing one image file;
            (b) A list of paths to image files;
            (c) A dict of dates and image files.

            In cases (a) and (b), QuickViewer will attempt to infer the date
            first from the directory name, then from the filename if no valid
            date is found within the directory name. The date is taken from the
            first unbroken string of numbers that can be successfully parsed
            with dateutil.parser.

        init_view : string, default='x-y'
            Orientation at which to initially display the image(s).

        init_sl : integer, default=None
            Slice number in the initial orientation direction at which to
            display the first image (can be changed interactively later). If
            None, the central slice will be displayed.

        init_pos : float, default=None
            Position in mm of the first slice to display. This will be rounded
            to the nearest slice. If <init_pos> and <init_idx> are both given,
            <init_pos> will override <init_idx> only if <scale_in_mm> is True.

        hu : float/tuple, default=(-300, 200)
            HU central value or range thresholds at which to display the image.
            Can later be changed interactively. If a single value is given, the
            HU range will be centred at this value with width given by
            <hu_width>. If a tuple is given, the HU range will be set to the
            two values in the tuple.

        hu_width : float, default=500
            Initial width of the HU window. Only used if <hu> is a single
            value.

        hu_limits : tuple, default=None
            Full range to use for the HU slider. Can also set to 'auto' to
            detect min and max HU in the image. Defaults to (-2000, 2000).

        hu_step : float, default=None
            Step size to use for the HU slider. Defaults to 1 if the maximum
            HU is >= 10, otherwise 0.1.

        figsize : float/tuple, default=5
            Height of the displayed figure in inches; figure width will be 
            automatically generated from this. If None, the value in
            _default_figsize is used. Can also be a tuple containing 
            (width, height) in inches in order to set width and height 
            manually.

        xlim : tuple, default=None
            Custom limits for the x axis. If one of the values in the tuple is
            None, the default x limits will be used.

        ylim : tuple, default=None
            Custom limits for the y axis. If one of the values in the tuple is
            None, the default y limits will be used.

        zlim : tuple, default=None
            Custom limits for the z axis. If one of the values in the tuple is
            None, the default z limits will be used.

        cmap : str, default=None
            Matplotlib colormap to use for image plotting. Supercedes any
            cmap included in <mpl_kwargs>.

        colorbar : bool, default=False
            If True, colorbars will be displayed for HU, dose and Jacobian
            determinant.

        colorbar_label : str, default=None
            Label for the colorbar and range slider. If None, will default to
            either 'HU' if an image file is given, or 'Dose (Gy)' if a dose
            file is given without an image.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the main image.See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
            Some useful keywords are:
                - 'cmap': colormap (default='gray').
                - 'interpolation': interpolation method (default='antialiased')

        dose_opacity : float, default=0.5
            Initial opacity of the overlaid dose field. Can later be changed
            interactively.

        dose_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the dose image. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
            Some useful keywords are:
                - 'cmap': colormap (default='jet').
                - 'interpolation': interpolation method (default='antialiased')

        dose_range : list, default=None
            Min and max dose range to plot. This can also be set via the 'vmin'
            and 'xmax' keys in <dose_kwargs>; <dose_range> will take precedence
            if set.

        dose_cmap : str, default='jet'
            Matplotlib colormap to use for dose field plotting. Supercedes
            any cmap in <dose_kwargs>

        invert_mask : bool, default=False
            If True, any masks applied will be inverted.

        mask_color : matplotlib color, default='black'
            color in which to display masked areas.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).

        jacobian_opacity : float, default=0.5
            Initial opacity of the overlaid jacobian determinant. Can later
            be changed interactively.

        jacobian_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the jacobian determinant. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
            Some useful keywords are:
                - 'cmap': colormap (default='seismic').
                - 'interpolation': interpolation method (default='antialiased')

        df_plot_type : str, default='grid'
            Option for initial plotting of deformation field. Can be 'grid',
            'quiver', or 'none'. Can later be changed interactively.

        df_spacing : int/tuple, default=30
            Spacing between arrows on the quiver plot/gridlines on the grid
            plot of a deformation field. Can be a single value for spacing in
            all directions, or a tuple with values for (x, y, z). Dimensions
            are mm if <scale_in_mm> is True, or voxels if <scale_in_mm> is
            False.

        df_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib when plotting
            the deformation field.

            For grid plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            Some useful keywords are:
                - 'linewidth': default=2
                - 'color': default='green'

            For quiver plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html.

        struct_plot_type : str, default='contour'
            Option for initial plot of structures. Can be 'contour', 'mask',
            'filled', or 'none'. Can later be changed interactively.

        struct_opacity : float, default=None
            Initial opacity of structures when plotted as masks or filled
            contours. Can later be changed interactively. Default=1 for masks,
            0.3 for filled contours.

        struct_linewidth : float, default=2
            Initial linewidth of structures when plotted as contours. Can later
            be changed interactively.

        struct_info : bool, default=False
            If True, the lengths and volumes of each structure will be
            displayed below the plot.

        struct_info_dp : int, default=2
            Number of decimal places to show for floats in structure info and
            comparison tables.

        length_units : str, default=None
            Units in which to display the lengths of structures if
            <struct_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:
                (a) 'mm'
                (b) 'voxels'

        area_units : str, default=None
            Units in which to display the areas of structures if
            <struct_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:
                (a) 'mm'
                (b) 'voxels'

        vol_units : str, default=None
            Units in which to display the volumes of structures if
            <struct_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm^3 if <scale_in_mm> is True. Options:
                (a) 'mm' for mm^3
                (b) 'voxels' for voxels
                (c) 'ml' for ml

        struct_legend : bool, default=False
            If True, a legend will be displayed for any plot with structures.

        init_struct : str, default=None
            If set to a structure name, the first slice to be displayed will
            be the central slice of that structure. This supercedes <init_pos>
            and <init_sl>.

        struct_names : list/dict, default=None
            Custom names for structures.

            If only one structure is to be loaded per file, this should be a
            dictionary where the keys are the desired custom names. The values
            of the dictionary can be either:
                (a) A string containing the path of the file containing the 
                    structure to be renamed;
                (b) A string containing the name of the structure to be renamed 
                    within a DICOM file;
                (c) A string containing the automatically-generated name of the
                    structure to be renamed (e.g. if the structure came from a 
                    file right_parotid.nii, its automatically generated name
                    would be 'right parotid';
                (d) A wildcard matching any of the above;
                (e) A list of any of the above.
            The list functionality allows the user to list multiple names
            that should be replaced by a single custom name, e.g. to handle
            cases where the same structure may have different names in 
            different DICOM files.

            If multiple structures are to be loaded from files (i.e. the 
            <multi_structs> parameter is set, or paths in the <structs> 
            parameter are prefixed with 'multi:') the <struct_names> parameter
            can either be:
                (a) A list of names, where the Nth name in the list will be
                   applied to the structure with mask label N in the structure
                   array; or
                (b) A dictionary where the keys are integers such that the
                   name associated with key N will be applied to the structure
                   with mask label N in the structure array.

            Any of the options described above can also be nested into a
            dictionary where the keys are labels, if a label dictionary was
            used to load structures in the <structs> parameter. The nested
            options for each key will only be applied to structures whose
            label is that key.

        struct_colors : dict, default=None
            A dictionary mapping structure names to colors in which the
            structure will be displayed.

            The dictionary keys should be either structure names or wildcards
            matching structure name(s). Note that structure names are inferred
            from the structure's filename unless otherwise specified in the
            <struct_names> parameter.

            The values of the dictionary can be any matplotlib colorlike
            object.

            The color dictionary can also be nested into a dictionary where the
            keys are labels, if a label dictionary was used to load structures
            in the <structs> parameter. The nested options for each key will
            only be applied to structures whose label is that key.

        structs_as_mask : bool, default=True
            If True, any loaded structures will be used to mask the image and
            dose field.

        compare_structs : bool, default=False
            If True, slice-by-slice comparison metrics for pairs of structures
            will be displayed below the plot, and compared structures masks
            will be plotted with their overlapping region in a different
            colour.

            The structures to compare can be set in three different ways:

                a) The user can explicitly indicate which structures should be
                   compared by giving a list of lists for the <structs>
                   argument, where each sublist contains exactly 2 filepaths
                   corresponding to a pair of structures to be compared (see
                   option 2b for <structs>).

                b) If only two structures are found for the filepaths/wildcards
                   given in the <structs> option, these two structures will
                   be compared.

                c) Otherwise, QuickViewer will search for pairs of loaded
                   structures with the same name (either inferred from the
                   filenames or specified manually by the user in the
                   <struct_names> option). If no structures with matching names
                   are found, no comparisons will be performed.

        ignore_empty_structs : bool, default=False
            If True, any loaded structures array that only contains zeros will
            be ignored. If False, the names of empty structures will be
            displayed in the UI with '(empty)' next to them.

        ignore_unpaired_structs : bool, default=False
            If <structure_comparison> is True and structure pairs are
            automatically detected based on their names, this parameter
            determines whether any structures for which a matching name is not
            found should be displayed.

            If True, only the pairs of structures with matching names will be
            shown. If False, all loaded structures will be shown regardless of
            whether they have a comparison match.

        structs_to_keep : list, default=True
            List of structure names or wildcards matching structures that you
            wish to load. All other structures will be ignored.

        structs_to_ignore : list, default=True
            List of structure names or wildcards matching structures that you
            wish to ignore.

        autoload_structs : bool, default=True
            If True, structures will all be automatically loaded and plotted.
            If False, all structures will be initially turned off and will 
            only be loaded if the user turns them on via the structure checkbox
            UI.

        continuous_update : bool, default=False
            If True, sliders in the UI will continuously update the figure as
            they are adjusted. Can cause lag.

        annotate_slice : str, default=None
            Color for annotation of slice number. Can be set to 'True' to use
            default colour (white).  If None, no annotation will
            be added unless viewing outside jupyter, in which case the
            annotation will be white by default.

        save_as : str, default=None
            File to which the figure will be saved upon creation. If not None,
            a text input and button will be added to the UI, allowing the
            user to save the figure again at a later point.

        zoom : double/tuple, default=None
            Amount between by which to zoom in (e.g. zoom=2 would give a
            2x zoom). Can be a single value for all directions or a tuple of
            values for the (x, y, z) directions.

        zoom_centre : tuple, default=None
            Centrepoint of zooming in order (x, y, z). If any of the values are
            None, the midpoint of the image will be used in that direction.

        zoom_ui : bool, default=None
            If True, a UI for zooming will be displayed. Default is False
            unless structs are loaded, in which case default is True.

        downsample : int/tuple, default=None
            Factor by which to downsample an image. Can be a single value for
            all directions or a tuple of values in the (x, y, z) directions.
            For no downsampling, set values to None or 1, e.g. to downsample
            in the z direction only: downsample=(1, 1, 3).

        affine : 4x4 array, default=None
            Affine matrix to be used if image source(s) are NumPy array(s). If
            image sources are nifti file paths or nibabel objects, this
            parameter is ignored. If None, the arguments <voxel_sizes> and
            <origin> will be used to set the affine matrix.

        voxel_sizes : tuple, default=(1, 1, 1)
            Voxel sizes in mm, given in the order (y, x, z). Only used if
            image source is a numpy array and <affine> is None.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm, given in the order (y, x, z). Only used if
            image source is a numpy array and <affine> is None.

        major_ticks : float, default=None
            If not None, this value will be used as the interval between major
            tick marks. Otherwise, automatic matplotlib axis tick spacing will
            be used.

        minor_ticks : int, default=None
            If None, no minor ticks will be plotted. Otherwise, this value will
            be the number of minor tick divisions per major tick interval.

        ticks_all_sides : bool, default=False
            If True, major (and minor if using) tick marks will be shown above
            and to the right hand side of the plot as well as below and to the
            left. The top/right ticks will not be labelled.
        '''

        # Get image file inputs
        if not isinstance(nii, list) or isinstance(nii, tuple):
            self.nii = [nii]
        else:
            self.nii = nii
        self.n = len(self.nii)

        # Process other inputs
        self.title = self.get_input_list(title)
        self.dose = self.get_input_list(dose)
        self.mask = self.get_input_list(mask)
        self.structs = self.get_input_list(structs, allow_sublist=True)
        self.multi_structs = self.get_input_list(multi_structs, allow_sublist=True)
        self.jacobian = self.get_input_list(jacobian)
        self.df = self.get_input_list(df)

        # Make individual viewers
        self.scale_in_mm = scale_in_mm
        self.viewers = []
        viewer_type = SingleViewer if not orthog_view else OrthogViewer
        kwargs = {key.replace('colour', 'color'): val for key, val in kwargs.items()}
        for i in range(self.n):
            viewer = viewer_type(
                self.nii[i],
                title=self.title[i],
                dose=self.dose[i],
                mask=self.mask[i],
                structs=self.structs[i],
                multi_structs=self.multi_structs[i],
                jacobian=self.jacobian[i],
                df=self.df[i],
                standalone=False,
                scale_in_mm=scale_in_mm,
                legend_loc=legend_loc,
                **kwargs,
            )
            self.viewers.append(viewer)
        self.n = len(self.viewers)
        if not self.n:
            print('No valid images found.')
            return

        # Load comparison images
        self.cb_splits = cb_splits
        self.overlay_opacity = overlay_opacity
        self.overlay_legend = overlay_legend
        self.legend_loc = legend_loc
        self.comparison_only = comparison_only
        if (
            comparison_only
            and comparison is None
            and not any([show_cb, show_overlay, show_diff])
        ):
            comparison = True
        self.load_comparison(comparison, show_cb, show_overlay, show_diff)
        self.translation = translation
        self.tfile = translation_file_to_overwrite
        self.show_mse = show_mse
        self.dta_tolerance = dta_tolerance
        self.dta_crit = dta_crit
        self.diff_crit = diff_crit

        # Settings needed for plotting
        figsize = kwargs.get('figsize', _default_figsize)
        if is_list(figsize):
            self.figwidth = figsize[0]
            self.figsize = figsize[1]
        else:
            self.figsize = to_inches(figsize)
            self.figwidth = 'auto'
        self.colorbar = kwargs.get('colorbar', False)
        self.comp_colorbar = self.colorbar and self.comparison_only
        self.zoom = kwargs.get('zoom', None)
        self.plots_per_row = plots_per_row
        self.suptitle = suptitle
        self.match_axes(match_axes)
        self.in_notebook = in_notebook()
        self.saved = False
        self.plotting = False

        # Make UI
        #  if any([v.im.timeseries for v in self.viewers]):
            #  share_slider = False
        self.make_ui(share_slider)

        # Display
        self.show(show)

    def show(self, show=True):
        SingleViewer.show(self, show)

    def get_input_list(self, inp, allow_sublist=False):
        '''Convert an input to a list with one item per image to be
        displayed.'''

        if inp is None:
            return [None for i in range(self.n)]

        # Convert arg to a list
        input_list = []
        if isinstance(inp, list) or isinstance(inp, tuple):
            if self.n == 1 and allow_sublist:
                input_list = [inp]
            else:
                input_list = inp
        else:
            input_list = [inp]
        return input_list + [None for i in range(self.n - len(input_list))]

    def any_attr(self, attr):
        '''Check whether any of this object's viewers have a given attribute.'''

        return any([getattr(v.im, 'has_' + attr) for v in self.viewers])

    def load_comparison(self, comparison, show_cb, show_overlay, show_diff):
        '''Create any comparison images.'''

        # Work out which comparison images to show
        self.comparison = {}
        comp_opts = []
        self.has_multicomp = False
        if isinstance(comparison, str):
            comparison = [comparison]
        if isinstance(comparison, bool):
            self.has_multicomp = comparison
            if self.has_multicomp:
                comp_opts = ['all']
        elif is_list(comparison):
            self.has_multicomp = 'all' in comparison
            comp_opts = comparison
        for name, flag in zip(
            ['chequerboard', 'overlay', 'difference'],
            [show_cb, show_overlay, show_diff],
        ):
            if flag and name not in comp_opts:
                comp_opts.append(name)
            setattr(self, f'has_{name}', name in comp_opts)

        # Case with no comparison images
        if not self.has_multicomp and not len(comp_opts):
            return

        # Use first two images
        assert self.n > 1
        im1 = self.viewers[0].im
        im2 = self.viewers[1].im

        # Make individual comparisons
        for comp in comp_opts:
            name = 'multicomp' if comp == 'all' else comp
            plot_type = None if comp == 'all' else comp
            comp_im = ImageComparison(
                im1, im2, plot_type=plot_type, scale_in_mm=self.scale_in_mm
            )
            setattr(self, name, comp_im)
            self.comparison[name] = comp_im

    def match_axes(self, match_axes):
        '''Adjust axes of plots to match if the match_axes option is set.'''

        if match_axes is None:
            return

        # Match axes in all orientations
        for view in _slider_axes:

            # Calculate limits using all plots
            ax_lims = []
            if match_axes in ['all', 'both', 'x', 'y', 'overlap']:
                for i in range(2):
                    min_lims = [v.im.ax_lims[view][i][0] for v in self.viewers]
                    max_lims = [v.im.ax_lims[view][i][1] for v in self.viewers]
                    f1, f2 = min, max
                    if match_axes == 'overlap':
                        f1, f2 = f2, f1
                    if min_lims[0] > max_lims[0]:
                        f1, f2 = f2, f1
                    ax_lims.append([f1(min_lims), f2(max_lims)])

            # Match axes to one plot
            else:
                try:
                    im = self.viewers[match_axes].im
                    ax_lims = im.ax_lims[view]

                except TypeError:
                    raise TypeError('Unrecognised option for <match_axes>', match_axes)

            # Set these limits for all plots
            all_ims = [v.im for v in self.viewers] + [
                c for c in self.comparison.values()
            ]
            for im in all_ims:
                if match_axes != 'y':
                    im.ax_lims[view][0] = ax_lims[0]
                if match_axes != 'x':
                    im.ax_lims[view][1] = ax_lims[1]

    def make_ui(self, share_slider):

        # Only allow share_slider if images have same frame of reference
        if share_slider:
            share_slider *= all(
                [v.im.has_same_geometry(self.viewers[0].im) for v in self.viewers]
            )

        # Make UI for first image
        v0 = self.viewers[0]
        v0.make_ui()

        # Store needed UIs
        self.ui_view = v0.ui_view
        self.view = self.ui_view.value
        #  self.ui_struct_plot_type = v0.ui_struct_plot_type
        #  self.ui_struct_plot_type2 = v0.ui_struct_plot_type2
        #  self.ui_struct_comp_type = v0.ui_struct_comp_type
        #  self.struct_plot_type = self.ui_struct_plot_type.value
        #  self.struct_plot_type2 = self.ui_struct_plot_type2.value

        # Make main upper UI list (= view radio + single HU/slice slider)
        many_sliders = not share_slider and self.n > 1
        if not many_sliders:
            self.main_ui = v0.main_ui
            if v0.zoom_ui:
                v0.ui_zoom_reset.on_click(self.make_reset_zoom(v0))
        else:
            self.main_ui = [self.ui_view]

        # Make UI for other images
        for v in self.viewers[1:]:
            v.make_ui(vimage=v0, share_slider=share_slider)
            #  v0.structs_for_jump.update(v.structs_for_jump)
        #  v0.ui_struct_jump.options = list(v0.structs_for_jump.keys())

        # Make UI for each image (= unique HU/slice sliders and struct jump)
        self.per_image_ui = []
        if many_sliders:
            for v in self.viewers:

                # Structure jumping
                sliders = []
                #  if v.im.has_structs:
                    #  sliders.append(v.ui_struct_jump)
                #  else:
                    #  if self.any_attr('structs'):
                        #  sliders.append(ipyw.Label())

                # HU sliders
                if v.hu_from_width:
                    sliders.extend([v.ui_hu_centre, v.ui_hu_width])
                else:
                    sliders.append(v.ui_hu)

                # Zoom sliders
                if v.zoom_ui:
                    sliders.extend(v.all_zoom_ui)
                    v.ui_zoom_reset.on_click(self.make_reset_zoom(v))

                # Slice slider
                sliders.append(v.ui_slice)
                self.per_image_ui.append(sliders)

        # Make extra UI list
        #  self.extra_ui = []
        #  for attr in ['mask', 'dose', 'df']:
            #  if self.any_attr(attr):
                #  self.extra_ui.append(getattr(v0, 'ui_' + attr))
        #  if self.any_attr('jacobian'):
            #  self.extra_ui.extend([v0.ui_jac_opacity, v0.ui_jac_range])
        #  if self.any_attr('structs'):
            #  to_add = [
                #  v0.ui_struct_plot_type,
                #  v0.ui_struct_linewidth,
                #  v0.ui_struct_opacity,
            #  ]
            #  if any([v.im.comp_type == 'others' for v in self.viewers]):
                #  to_add.insert(1, v0.ui_struct_plot_type2)
                #  to_add.insert(2, v0.ui_struct_comp_type)
            #  self.extra_ui.extend(to_add)

        # Make extra UI elements
        #  self.make_lower_ui()
        #  self.make_comparison_ui()
        #  self.make_translation_ui()

        # Assemble UI boxes
        main_and_extra_box = ipyw.HBox(
            [
                ipyw.VBox(self.main_ui),
                #  ipyw.VBox(self.extra_ui),
                #  ipyw.VBox(self.trans_ui),
                #  ipyw.VBox(self.comp_ui),
            ]
        )
        self.slider_boxes = [ipyw.VBox(ui) for ui in self.per_image_ui]
        self.set_slider_widths()
        self.upper_ui = [main_and_extra_box, ipyw.HBox(self.slider_boxes)]
        self.upper_ui_box = ipyw.VBox(self.upper_ui)
        #  self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)
        self.all_ui = (
            self.main_ui
            #  + self.extra_ui
            + list(itertools.chain.from_iterable(self.per_image_ui))
            #  + self.comp_ui
            #  + self.trans_ui
            #  + self.ui_struct_checkboxes
            + [self.trigger]
        )

    def make_lower_ui(self):
        '''Make lower UI for structure checkboxes.'''

        # Saving UI
        self.lower_ui = []

        # Structure UI
        many_with_structs = sum([v.im.has_structs for v in self.viewers]) > 1
        self.ui_struct_checkboxes = []
        for i, v in enumerate(self.viewers):

            # Add plot title to structure UI
            if many_with_structs and v.im.has_structs:
                if not hasattr(v.im, 'title') or not v.im.title:
                    title = f'<b>Image {i + 1}</b>'
                else:
                    title = f'<b>{v.im.title + ":"}</b>'
                self.lower_ui.append(ipyw.HTML(value=title))

            # Add to overall lower UI
            self.lower_ui.extend(v.lower_ui)
            self.ui_struct_checkboxes.extend(v.ui_struct_checkboxes)

        self.lower_ui.extend([self.viewers[0].save_name, self.viewers[0].save_button])

    def make_comparison_ui(self):

        self.comp_ui = []

        # Multicomparison dropdown
        comp_opts = [
            'overlay',
            'chequerboard',
        ]
        if all([v.im.has_same_geometry(self.viewers[0].im) for v in self.viewers]):
            comp_opts.extend([
                'difference',
                'absolute difference',
                'distance to agreement',
                'gamma index',
            ])
        if self.comparison_only:
            comp_opts.extend(['image 1', 'image 2'])
        self.ui_multicomp = ipyw.Dropdown(options=comp_opts, description='Comparison')
        if self.has_multicomp:
            self.comp_ui.append(self.ui_multicomp)

        # Chequerboard slider
        max_splits = max([15, self.cb_splits])
        self.ui_cb = ipyw.IntSlider(
            min=2,
            max=max_splits,
            value=self.cb_splits,
            step=1,
            continuous_update=self.viewers[0].continuous_update,
            description='Chequerboard splits',
            style=_style,
        )
        if self.has_chequerboard or self.has_multicomp:
            self.comp_ui.append(self.ui_cb)

        # Overlay slider
        self.ui_overlay = ipyw.FloatSlider(
            value=self.overlay_opacity,
            min=0,
            max=1,
            step=0.1,
            description='Overlay opacity',
            continuous_update=self.viewers[0].continuous_update,
            readout_format='.1f',
            style=_style,
        )
        if self.has_overlay or self.has_multicomp:
            self.comp_ui.append(self.ui_overlay)

        # Inversion checkbox
        self.ui_invert = ipyw.Checkbox(value=False, description='Invert comparison')
        if len(self.comparison):
            self.comp_ui.append(self.ui_invert)

    def make_translation_ui(self):

        self.trans_ui = []
        if not self.translation:
            return

        self.trans_viewer = self.viewers[int(self.n > 1)]
        self.trans_ui.append(ipyw.HTML(value='<b>Translation:</b>'))

        # Get input/output filenames
        if self.tfile is None:
            tfile = self.find_translation_file(self.trans_viewer.im)
            self.has_translation_input = tfile is not None
            if self.has_translation_input:
                tfile_out = re.sub('.0.txt', '_custom.txt', tfile)
            else:
                tfile_out = 'translation.txt'
        else:
            tfile = self.tfile
            tfile_out = self.tfile
            self.has_translation_input = True

        # Make translation file UI
        if self.has_translation_input:
            self.translation_input = ipyw.Text(description='Original:', value=tfile)
            self.trans_ui.append(self.translation_input)
        self.translation_output = ipyw.Text(description='Save as:', value=tfile_out)
        self.tbutton = ipyw.Button(description='Write translation')
        self.tbutton.on_click(self.write_translation_to_file)
        self.trans_ui.extend([self.translation_output, self.tbutton])

        # Make translation sliders
        self.tsliders = {}
        for ax in _axes:
            n = self.trans_viewer.im.n_voxels[ax]
            self.tsliders[ax] = ipyw.IntSlider(
                min=-n,
                max=n,
                value=0,
                description=f'{ax} (0 mm)',
                continuous_update=False,
                #  style=_style
            )
            self.trans_ui.append(self.tsliders[ax])
        self.current_trans = {ax: slider.value for ax, slider in self.tsliders.items()}

    def find_translation_file(self, image):
        '''Find an elastix translation file inside the directory of an image.'''

        if not hasattr(image, 'path'):
            return
        indir = os.path.dirname(image.path)
        tfile = indir + '/TransformParameters.0.txt'
        if os.path.isfile(tfile):
            return tfile

    def write_translation_to_file(self, _):
        '''Write current translation to file.'''

        input_file = (
            self.translation_input.value if self.has_translation_input else None
        )
        if input_file == '':
            input_file = None
        output_file = self.translation_output.value
        translations = {
            f'd{ax}': -self.tsliders[ax].value
            * abs(self.trans_viewer.im.voxel_sizes[ax])
            for ax in self.tsliders
        }
        write_translation_to_file(output_file, input_file=input_file, **translations)

    def apply_translation(self):
        '''Update the description of translation sliders to show translation
        in mm if the translation is changed.'''

        new_trans = {ax: slider.value for ax, slider in self.tsliders.items()}
        if new_trans == self.current_trans:
            return

        # Set shift for image
        self.current_trans = new_trans
        self.trans_viewer.im.set_shift(
            self.current_trans['x'], self.current_trans['y'], self.current_trans['z']
        )

        # Adjust descriptions
        for ax, slider in self.tsliders.items():
            slider.description = '{} ({:.0f} mm)'.format(
                ax, self.trans_viewer.im.shift_mm[ax]
            )

    def set_slider_widths(self):
        '''Adjust widths of slider UI.'''

        if self.plots_per_row is not None and self.plots_per_row < self.n:
            return
        for i, slider in enumerate(self.slider_boxes[:-1]):
            width = (
                self.figsize
                * self.viewers[i].im.get_plot_aspect_ratio(
                    self.view, self.zoom, self.colorbar
                )
                * mpl.rcParams['figure.dpi']
            )
            slider.layout = ipyw.Layout(width=f'{width}px')

    def make_reset_zoom(self, viewer):
        '''Make a reset zoom function for a given viewer that updates zoom
        without plotting, then plots afterwards.'''

        def reset_zoom(_):
            self.plotting = True
            viewer.reset_zoom(_)
            self.plotting = False
            self.trigger.value = not self.trigger.value

        return reset_zoom

    def make_fig(self):

        # Get relative width of each subplot
        width_ratios = [
            v.im.get_plot_aspect_ratio(self.view, self.zoom, self.colorbar)
            for v in self.viewers
        ]
        width_ratios.extend(
            [
                c.get_plot_aspect_ratio(self.view, self.comp_colorbar, self.figsize)
                for c in self.comparison.values()
            ]
        )

        # Get rows and columns
        n_plots = (not self.comparison_only) * self.n + len(self.comparison)
        if self.comparison_only:
            width_ratios = width_ratios[self.n :]
        if self.plots_per_row is not None:
            n_cols = min([self.plots_per_row, n_plots])
            n_rows = int(np.ceil(n_plots / n_cols))
            width_ratios_padded = width_ratios + [
                0 for i in range((n_rows * n_cols) - n_plots)
            ]
            ratios_per_row = np.array(width_ratios_padded).reshape(n_rows, n_cols)
            width_ratios = np.amax(ratios_per_row, axis=0)
        else:
            n_cols = n_plots
            n_rows = 1

        # Calculate height and width
        if self.figwidth == 'auto':
            height = self.figsize * n_rows
            width = self.figsize * sum(width_ratios)
        else:
            height = self.figsize
            width = self.figwidth

        # Outside notebook, just resize figure
        if not self.in_notebook and hasattr(self, 'fig'):
            self.fig.set_size_inches(width, height)
            return

        # Make new figure
        self.fig = plt.figure(figsize=(width, height))

        # Make gridspec
        gs = self.fig.add_gridspec(n_rows, n_cols, width_ratios=width_ratios)
        i = 0
        if not self.comparison_only:
            for v in self.viewers:
                v.gs = gs[i]
                i += 1
        for c in self.comparison.values():
            c.gs = gs[i]
            i += 1

        # Assign callbacks to figure
        if not self.in_notebook:
            self.set_callbacks()

    def set_callbacks(self):
        '''Set callbacks for scrolls and keypresses.'''

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_key(self, event):
        '''Callbacks for key presses.'''

        # Apply all callbacks to first viewer
        self.plotting = True
        self.viewers[0].on_key(event)

        # Extra callbacks for scrolling each plot
        if len(self.per_image_ui):
            for v in self.viewers[1:]:

                # Scrolling
                n_small = 1
                n_big = 5
                if event.key == 'left':
                    v.decrease_slice(n_small)
                elif event.key == 'right':
                    v.increase_slice(n_small)
                elif event.key == 'down':
                    v.decrease_slice(n_big)
                elif event.key == 'up':
                    v.increase_slice(n_big)

        # Press i to invert comparisons
        elif event.key == 'i':
            if len(self.comparison):
                self.ui_invert.value = not self.ui_invert.value

        # Press o to change overlay opacity
        elif event.key == 'o':
            if self.has_overlay:
                ops = [0.2, 0.35, 0.5, 0.65, 0.8]
                next_op = {
                    ops[i]: ops[i + 1] if i + 1 < len(ops) else ops[0]
                    for i in range(len(ops))
                }
                diffs = [abs(op - self.ui_overlay.value) for op in ops]
                current = ops[diffs.index(min(diffs))]
                self.ui_overlay.value = next_op[current]

        # Remake plot
        self.plotting = False
        self.plot(tight_layout=(event.key == 'v'))

    def on_scroll(self, event):
        '''Callbacks for scroll events.'''

        # First viewers
        self.plotting = True
        self.viewers[0].on_scroll(event)

        # Extra callbacks for scrolling each plot
        if len(self.per_image_ui):
            for v in self.viewers[1:]:
                if event.button == 'up':
                    self.increase_slice()
                elif event.button == 'down':
                    self.decrease_slice()

        # Remake plot
        self.plotting = False
        self.plot(tight_layout=False)

    def plot(self, tight_layout=True, **kwargs):
        '''Plot all images.'''

        if self.plotting:
            return
        self.plotting = True

        # Deal with view change
        if self.ui_view.value != self.view:
            self.view = self.ui_view.value
            for v in self.viewers:
                v.view = self.ui_view.value
                v.on_view_change()
            self.set_slider_widths()

        # Deal with structure plot type change
        #  if self.struct_plot_type != self.ui_struct_plot_type.value:
            #  self.struct_plot_type = self.ui_struct_plot_type.value
            #  self.viewers[0].update_struct_sliders()

        # Deal with structure jumps
        #  for v in self.viewers:
            #  if v.ui_struct_jump != '':
                #  v.jump_to_struct()

        # Apply any translations
        #  if self.translation:
            #  self.apply_translation()

        # Reset figure
        self.make_fig()

        # Plot all images
        for v in self.viewers:
            if self.comparison_only:
                v.set_slice_and_view()
                v.im.set_slice(self.view, v.slice[self.view])
            else:
                v.plot()

        # Adjust comparison UI
        #  multicomp_plot_type = self.ui_multicomp.value
        #  if self.has_multicomp and len(self.comparison) == 1:
            #  self.ui_cb.disabled = not multicomp_plot_type == 'chequerboard'
            #  self.ui_overlay.disabled = not multicomp_plot_type == 'overlay'

        # Deal with comparisons
        #  if len(self.comparison):

            #  # Get settings
            #  invert = self.ui_invert.value
            #  plot_kwargs = self.viewers[0].v_min_max
            #  if self.viewers[0].mpl_kwargs is not None:
                #  plot_kwargs.update(self.viewers[0].mpl_kwargs)

            #  # Plot all comparisons
            #  for name, comp in self.comparison.items():
                #  plot_type = None if name != 'multicomp' else multicomp_plot_type
                #  SingleViewer.plot_image(
                    #  self,
                    #  comp,
                    #  invert=invert,
                    #  plot_type=plot_type,
                    #  cb_splits=self.ui_cb.value,
                    #  overlay_opacity=self.ui_overlay.value,
                    #  overlay_legend=self.overlay_legend,
                    #  overlay_legend_loc=self.legend_loc,
                    #  zoom=self.viewers[0].zoom,
                    #  zoom_centre=self.viewers[0].zoom_centre,
                    #  mpl_kwargs=self.viewers[0].v_min_max,
                    #  colorbar=self.comp_colorbar,
                    #  colorbar_label=self.viewers[0].colorbar_label,
                    #  show_mse=self.show_mse,
                    #  dta_tolerance=self.dta_tolerance,
                    #  dta_crit=self.dta_crit,
                    #  diff_crit=self.diff_crit,
                #  )

        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)

        if tight_layout:
            plt.tight_layout()
        self.plotting = False

        # Automatic saving on first plot
        if self.viewers[0].save_as is not None and not self.saved:
            self.viewers[0].save_fig()
            self.saved = True

        # Update figure
        if not self.in_notebook:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


class SingleViewer:
    """Class for displaying an Image with interactive elements."""

    def __init__(
        self,
        im=None,
        init_view="x-y",
        init_sl=None,
        init_pos=None,
        hu=[-300, 200],
        hu_width=500,
        hu_limits=None,
        hu_step=None,
        figsize=_default_figsize,
        xlim=None,
        ylim=None,
        zlim=None,
        zoom=None,
        zoom_centre=None,
        zoom_ui=None,
        cmap=None,
        colorbar=False,
        colorbar_label=None,
        mpl_kwargs=None,
        dose_opacity=0.5,
        dose_kwargs=None,
        dose_range=None,
        dose_cmap=None,
        invert_mask=False,
        mask_color="black",
        jacobian_opacity=0.5,
        jacobian_kwargs=None,
        df_plot_type="grid",
        df_spacing=30,
        df_kwargs=None,
        struct_plot_type="contour",
        struct_opacity=None,
        struct_linewidth=2,
        struct_info=False,
        struct_info_dp=2,
        length_units=None,
        area_units=None,
        vol_units=None,
        struct_legend=False,
        legend_loc="lower left",
        init_struct=None,
        standalone=True,
        continuous_update=False,
        annotate_slice=None,
        save_as=None,
        show=True,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
        no_axis_labels=False,
        scale_in_mm=True,
        **kwargs,
    ):

        # Make Image
        self.im = self.make_image(im, **kwargs)
        self.gs = None  # Gridspec in which to place plot axes
        self.scale_in_mm = scale_in_mm

        # Set initial orientation
        view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
        if init_view in view_map:
            self.view = view_map[init_view]
        else:
            self.view = init_view

        # Set initial slice number for each orientation
        self.slice = {
            view: np.ceil(self.im.n_voxels[z] / 2) 
            for view, z in _slice_axes.items()
        }
        if init_pos is not None and self.scale_in_mm:
            self.set_slice_from_pos(init_view, init_pos)
        else:
            self.set_slice(init_view, init_sl)

        # Assign plot settings
        # HU range settings
        self.hu = hu
        self.hu_width = hu_width
        self.hu_limits = hu_limits
        self.hu_step = hu_step
        self.hu_from_width = isinstance(hu, float) or isinstance(hu, int)
        if hu_limits is None:
            #  if self.im.dose_as_im:
                #  if dose_range:
                    #  self.hu_limits = dose_range
                #  else:
                    #  self.hu_limits = (self.im.data.min(), self.im.data.max())
            #  else:
            self.hu_limits = [-2000, 2000]
        elif hu_limits == "auto":
            self.hu_limits = (self.im.data.min(), self.im.data.max())
        if hu_step is None:
            self.hu_step = (
                1 if abs(self.hu_limits[1] - self.hu_limits[0]) >= 10 else 0.1
            )
        if self.hu[0] < self.hu_limits[0]:
            self.hu_limits[0] = self.hu[0]
        if self.hu[1] > self.hu_limits[1]:
            self.hu_limits[1] = self.hu[1]

        # Other settings
        self.in_notebook = in_notebook()
        self.mpl_kwargs = mpl_kwargs if mpl_kwargs else {}
        self.figsize = to_inches(figsize)
        self.continuous_update = continuous_update
        self.colorbar = colorbar
        self.colorbar_drawn = False
        self.colorbar_label = colorbar_label if colorbar_label is not None else "HU"
        self.annotate_slice = annotate_slice
        if self.annotate_slice is None and not self.in_notebook:
            self.annotate_slice = True
        self.save_as = save_as
        self.plotting = False
        self.callbacks_set = False
        self.standalone = standalone
        self.custom_ax_lims = xlim, ylim, xlim
        self.zoom = zoom
        self.zoom_centre = zoom_centre
        self.zoom_ui = zoom_ui
        #  if zoom_ui is None:
            #  self.zoom_ui = self.im.has_structs
        self.major_ticks = major_ticks
        self.minor_ticks = minor_ticks
        self.ticks_all_sides = ticks_all_sides
        self.no_axis_labels = no_axis_labels
        self.legend_loc = legend_loc

        # Colormap
        if cmap:
            self.mpl_kwargs["cmap"] = cmap
        #  elif self.im.dose_as_im:
            #  self.mpl_kwargs["cmap"] = dose_cmap if dose_cmap else "jet"

        # Display plot
        if standalone:
            self.make_ui()
            self.show(show)

    def make_image(self, im, *args, **kwargs):
        '''Set up image object.'''

        if isinstance(im, Image):
            return im
        return Image(*args, **kwargs)

    def set_slice(self, view, sl):
        """Set the current slice number in a specific orientation."""

        if sl is None:
            return
        max_slice = self.im.n_voxels[_slice_axes[view]]
        min_slice = 1
        if self.slice[view] < min_slice:
            self.slice[view] = min_slice
        elif self.slice[view] > max_slice:
            self.slice[view] = max_slice
        else:
            self.slice[view] = sl

    def set_slice_from_pos(self, view, pos):
        """Set the current slice number from a position in mm."""

        ax = _slice_axes[view]
        sl = self.im.pos_to_slice(pos, ax)
        self.set_slice(view, sl)

    def make_ui(self, other_viewer=None, share_slider=True):
        '''Make Jupyter notebook UI. If other_viewer contains another SingleViewer
        instance, the UI will be taken from that image. If share_slider is
        False, independent HU and slice sliders will be created.'''

        shared_ui = isinstance(other_viewer, SingleViewer)
        self.main_ui = []

        # View radio buttons
        if not shared_ui:
            self.ui_view = ipyw.RadioButtons(
                options=['x-y', 'y-z', 'x-z'],
                value=self.view,
                description='Slice plane selection:',
                disabled=False,
                style=_style,
            )
            #  if not self.im.dim2:
            self.main_ui.append(self.ui_view)
        else:
            self.ui_view = other_viewer.ui_view
            self.view = self.ui_view.value

        # HU and slice sliders
        if not share_slider or not shared_ui:

            # Make HU slider
            if self.hu_limits == 'auto':
                hu_limits = (self.im.data.min(), self.im.data.max())
            else:
                hu_limits = self.hu_limits

            # Single range slider
            if not self.hu_from_width:
                vmin = max([self.hu[0], hu_limits[0]])
                vmax = min([self.hu[1], hu_limits[1]])
                ui_hu_kwargs = {
                    'min': hu_limits[0],
                    'max': hu_limits[1],
                    'value': (vmin, vmax),
                    'description': self.colorbar_label,
                    'continuous_update': False,
                    'style': _style,
                    'step': self.hu_step,
                }
                slider_kind = (
                    ipyw.FloatRangeSlider if self.hu_step < 1 else ipyw.IntRangeSlider
                )
                self.ui_hu = slider_kind(**ui_hu_kwargs)
                self.main_ui.append(self.ui_hu)

            # Centre and window sliders
            else:
                self.ui_hu_centre = ipyw.IntSlider(
                    min=hu_limits[0],
                    max=hu_limits[1],
                    value=self.hu,
                    description='HU centre',
                    continuous_update=False,
                    style=_style,
                )
                self.ui_hu_width = ipyw.IntSlider(
                    min=0,
                    max=abs(hu_limits[1] - hu_limits[0]),
                    value=self.hu_width,
                    description='HU width',
                    continuous_update=False,
                    style=_style,
                )
                self.ui_hu_list = [self.ui_hu_centre, self.ui_hu_width]
                self.main_ui.extend(self.ui_hu_list)
                self.ui_hu = ipyw.VBox(self.ui_hu_list)

            # Make zoom UI
            if self.zoom_ui:

                # Get initial zoom level for each view
                if self.zoom is None:
                    init_zoom = 1
                elif is_list(self.zoom):
                    init_zoom = max(self.zoom)
                else:
                    init_zoom = self.zoom

                # Zoom level slider
                self.ui_zoom = ipyw.FloatSlider(
                    min=1,
                    max=8,
                    step=0.1,
                    value=init_zoom,
                    description='Zoom',
                    readout_format='.1f',
                    continuous_update=False,
                    style=_style,
                )

                # Get initial zoom centres
                zoom_centre = self.zoom_centre if is_list(self.zoom_centre) \
                        else [self.zoom_centre for i in range(3)]
                im_centre = list(self.im.get_image_centre())
                self.default_centre = {
                    view: [im_centre[ax[0]], im_centre[ax[1]]]
                    for view, ax in _plot_axes.items()
                }
                self.current_centre = {}
                for view in _plot_axes:
                    self.current_centre[view] = []
                    for i, ax in enumerate(_plot_axes[view]):
                        if zoom_centre is None or zoom_centre[ax] is None:
                            self.current_centre[view].append(
                                self.default_centre[view][i]
                            )
                        else:
                            self.current_centre[view].append(zoom_centre[ax])

                # Make zoom centre sliders
                self.ui_zoom_centre_x = ipyw.FloatSlider(
                    continuous_update=self.continuous_update,
                    readout_format='.1f',
                    step=1,
                )
                self.ui_zoom_centre_y = ipyw.FloatSlider(
                    continuous_update=self.continuous_update,
                    readout_format='.1f',
                    step=1,
                )
                self.update_zoom_sliders()

                # Zoom reset button
                self.ui_zoom_reset = ipyw.Button(description='Reset zoom')
                if self.standalone:
                    self.ui_zoom_reset.on_click(self.reset_zoom)
                self.all_zoom_ui = [
                    self.ui_zoom,
                    self.ui_zoom_centre_x,
                    self.ui_zoom_centre_y,
                    self.ui_zoom_reset,
                ]
                self.main_ui.extend(self.all_zoom_ui)

            # Make slice slider
            readout = '.1f' if self.scale_in_mm else '.0f'
            self.ui_slice = ipyw.FloatSlider(
                continuous_update=self.continuous_update,
                style=_style,
                readout_format=readout,
            )
            self.own_ui_slice = True
            self.update_slice_slider()
            #  if not self.im.dim2:
            self.main_ui.append(self.ui_slice)

        else:
            if self.hu_from_width:
                self.ui_hu_width = other_viewer.ui_hu_width
                self.ui_hu_centre = other_viewer.ui_hu_centre
            else:
                self.ui_hu = other_viewer.ui_hu
            self.ui_slice = other_viewer.ui_slice
            self.slice[self.view] = self.ui_slice.value
            if self.zoom_ui:
                self.ui_zoom = other_viewer.ui_zoom
                self.ui_zoom_centre_x = other_viewer.ui_zoom_centre_x
                self.ui_zoom_centre_y = other_viewer.ui_zoom_centre_y
                self.ui_zoom_reset = other_viewer.ui_zoom_reset
                self.current_centre = other_viewer.current_centre
            self.own_ui_slice = False

        # Make lower
        #  self.make_lower_ui()

        # Combine UI elements
        self.upper_ui = [ipyw.VBox(self.main_ui)] # ipyw.VBox(self.extra_ui)]
        self.upper_ui_box = ipyw.HBox(self.upper_ui)
        #  self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)
        self.all_ui = (
            self.main_ui #+ self.extra_ui + self.ui_struct_checkboxes 
            + [self.trigger]
        )

    def show(self, show=True):
        '''Display plot and UI.'''

        if self.in_notebook:
            SingleViewer.show_in_notebook(self, show)
        else:
            self.plot()
            if show:
                plt.show()

    def show_in_notebook(self, show):
        '''Display interactive output in a jupyter notebook.'''

        from IPython.display import display, HTML

        # Create keywords linked to all changeable UI elements
        ui_kw = {
            str(np.random.rand()): ui for ui in self.all_ui if hasattr(ui, 'value')
        }

        # Link the output plot to self.plot() and keywords
        self.out = ipyw.interactive_output(self.plot, ui_kw)

        # Display the UI and the output
        to_display = [self.upper_ui_box, self.out]
        #  if len(self.lower_ui):
            #  to_display.append(self.lower_ui_box)
        if show:
            display(*to_display)

    def plot(self, **kwargs):
        '''Plot a slice with current settings.'''

        if self.plotting:
            return
        self.plotting = True

        # Set slice and view
        self.set_slice_and_view()

        # Get main image settings
        mpl_kwargs = self.v_min_max
        if self.mpl_kwargs is not None:
            mpl_kwargs.update(self.mpl_kwargs)

        # Check whether colorbar already drawn
        colorbar = self.colorbar
        if not self.in_notebook and self.colorbar_drawn:
            colorbar = False

        # Get date argument
        #  n_date = 1
        #  if self.im.timeseries:
            #  n_date = self.ui_time.value

        # Make plot
        self.plot_image(
            self.im,
            view=self.view,
            sl=self.slice[self.view],
            gs=self.gs,
            mpl_kwargs=mpl_kwargs,
            figsize=self.figsize,
            zoom=self.zoom,
            zoom_centre=self.zoom_centre,
            colorbar=colorbar,
            colorbar_label=self.colorbar_label,
            legend_loc=self.legend_loc,
            annotate_slice=self.annotate_slice,
            major_ticks=self.major_ticks,
            minor_ticks=self.minor_ticks,
            ticks_all_sides=self.ticks_all_sides,
            no_axis_labels=self.no_axis_labels,
            show=False,
            xlim=self.custom_ax_lims[_plot_axes[self.view][0]],
            ylim=self.custom_ax_lims[_plot_axes[self.view][1]],
        )
        self.plotting = False
        self.colorbar_drawn = True

        # Ensure callbacks are set if outside jupyter
        if not self.in_notebook:
            self.set_callbacks()

        # Update figure
        if not self.in_notebook:
            self.im.fig.canvas.draw_idle()
            self.im.fig.canvas.flush_events()

    def plot_image(self, im, **kwargs):
        '''Plot an Image, reusing existing axes if outside a Jupyter
        notebook.'''

        # Get axes
        ax = None
        if not self.in_notebook and hasattr(im, 'ax'):
            ax = getattr(im, 'ax')
            ax.clear()

        # Plot image
        im.plot(ax=ax, **kwargs)

    def on_view_change(self):
        '''Deal with a view change.'''

        self.update_slice_slider()
        if self.zoom_ui:
            self.update_zoom_sliders()

    def set_slice_and_view(self):
        '''Get the current slice and view to plot from the UI.'''

        # Get view
        view = self.ui_view.value
        if self.view != view:
            self.view = view
            self.on_view_change()

        # Get slice
        self.jump_to_struct()
        self.slice[self.view] = self.slider_to_slice(self.ui_slice.value)
        if not self.scale_in_mm:
            self.update_slice_slider_desc()

        # Get HU range
        self.v_min_max = self.get_hu_range()

        # Get zoom settings
        if self.zoom_ui:
            self.zoom = self.ui_zoom.value
            self.zoom_centre = [1, 1, 1]
            for i, ui in enumerate([self.ui_zoom_centre_x, self.ui_zoom_centre_y]):
                self.zoom_centre[_plot_axes[self.view][i]] = ui.value
                self.current_centre[self.view][i] = ui.value

    def jump_to_struct(self):
        '''Jump to the mid slice of a structure.'''

        return

        if self.ui_struct_jump.value == '':
            return

        self.current_struct = self.ui_struct_jump.value
        struct = self.structs_for_jump[self.current_struct]
        if not struct.empty:
            if not struct.on_slice(self.view, self.slice[self.view]):
                mid_slice = int(np.mean(list(struct.contours[self.view].keys())))
                self.ui_slice.value = self.slice_to_slider(
                    mid_slice, _slice_axes[self.view]
                )
                self.slice[self.view] = mid_slice
            self.centre_at_struct(struct)
        self.ui_struct_jump.value = ''


    def get_hu_range(self):
        '''Get vmin and vmax from HU sliders.'''

        if self.hu_from_width:
            w = self.ui_hu_width.value / 2
            centre = self.ui_hu_centre.value
            return {'vmin': centre - w, 'vmax': centre + w}
        else:
            return {'vmin': self.ui_hu.value[0], 'vmax': self.ui_hu.value[1]}

    def update_slice_slider(self):
        '''Update the slice slider to show the axis corresponding to the
        current view, with value set to the last value used on that axis.'''

        if not self.own_ui_slice:
            return

        # Get new min and max
        ax = _slice_axes[self.view]
        if self.scale_in_mm:
            new_min = min(self.im.lims[ax])
            new_max = max(self.im.lims[ax])
        else:
            new_min = 1
            new_max = self.im.n_voxels[ax]

        # Set slider values
        val = self.slice_to_slider(self.slice[self.view])
        self.update_slider(self.ui_slice, new_min, new_max, val)

        # Set step and description
        self.ui_slice.step = abs(self.im.voxel_size[ax]) if self.scale_in_mm else 1
        self.update_slice_slider_desc()

    def update_slider(self, slider, new_min, new_max, val):
        '''Update slider min, max, and value without causing errors due to
        values outside range.'''

        # Set to widest range
        if new_min < slider.min:
            slider.min = new_min
        if new_max > slider.max:
            slider.max = new_max

        # Set new value
        slider.value = val

        # Set final limits
        if slider.min != new_min:
            slider.min = new_min
        if slider.max != new_max:
            slider.max = new_max

    def update_slice_slider_desc(self):
        '''Update slice slider description to reflect current axis and
        position.'''

        if not self.own_ui_slice:
            return
        ax = _slice_axes[self.view]
        if self.scale_in_mm:
            self.ui_slice.description = f'{_axes[ax]} (mm)'
        else:
            pos = self.im.slice_to_pos(self.slider_to_slice(self.ui_slice.value), ax)
            self.ui_slice.description = f'{_axes[ax]} ({pos:.1f} mm)'

    def update_zoom_sliders(self):
        '''Update zoom sliders to reflect the current view.'''

        if not self.zoom_ui:
            return

        units = ' (mm)' if self.scale_in_mm else ''
        for i, ui in enumerate([self.ui_zoom_centre_x, self.ui_zoom_centre_y]):

            # Set min, max, and value
            ax = _plot_axes[self.view][i]
            new_min = min(self.im.lims[ax])
            new_max = max(self.im.lims[ax])
            self.update_slider(ui, new_min, new_max, self.current_centre[self.view][i])

            # Update description
            ui.description = '{} centre {}'.format(
                _axes[_plot_axes[self.view][i]], 
                units
            )

    def reset_zoom(self, _):
        '''Reset zoom values to 1 and zoom centres to defaults.'''

        self.ui_zoom.value = 1
        self.current_centre[self.view] = self.default_centre[self.view]
        self.plotting = True
        self.update_zoom_sliders()
        self.plotting = False
        if self.standalone:
            self.trigger.value = not self.trigger.value

    def slice_to_slider(self, sl, ax=None):
        '''Convert a slice number to a slider value.'''

        if ax is None:
            ax = _slice_axes[self.view]

        if self.scale_in_mm:
            return self.im.slice_to_pos(sl, ax)
        else:
            return sl

    def slider_to_slice(self, val, ax=None):
        '''Convert a slider value to a slice number.'''

        if ax is None:
            ax = _slice_axes[self.view]

        if self.scale_in_mm:
            return self.im.pos_to_slice(val, ax)
        else:
            return int(val)


def in_notebook():
    """Check whether current code is being run within a Jupyter notebook."""

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        else:
            return False
    except NameError:
        return False


def to_inches(size):
    """Convert a size string to a size in inches. If a float is given, it will
    be returned. If a string is given, the last two characters will be used to
    determine the units:
        - "in": inches
        - "cm": cm
        - "mm": mm
        - "px": pixels
    """

    if not isinstance(size, str):
        return size

    val = float(size[:-2])
    units = size[-2:]
    inches_per_cm = 0.394
    if units == "in":
        return val
    elif units == "cm":
        return inches_per_cm * val
    elif units == "mm":
        return inches_per_cm * val / 10
    elif units == "px":
        return val / mpl.rcParams["figure.dpi"]

