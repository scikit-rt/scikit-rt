"""New version of QuickViewer that uses the scikit-rt Image class."""

import ipywidgets as ipyw
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from skrt.core import is_list
from skrt.image import (
    get_mask,
    Image, 
    ImageComparison,
    _axes, 
    _slice_axes, 
    _plot_axes,
    _default_figsize,
)
from skrt.dose import Dose
from skrt.registration import Grid, DeformationField, Jacobian
from skrt.structures import (
    StructureSet, 
    ROI, 
    get_colored_roi_string, 
    df_to_html,
    compare_roi_pairs
)

# ipywidgets settings
_style = {'description_width': 'initial'}


class BetterViewer:
    '''Display multiple SingleViewers and/or comparison images.'''

    def __init__(
        self,
        images=None,
        title=None,
        mask=None,
        dose=None,
        rois=None,
        #  multi_rois=None,
        grid=None,
        jacobian=None,
        df=None,
        share_slider=True,
        orthog_view=False,
        plots_per_row=None,
        match_axes=None,
        scale_in_mm=True,
        comparison=None,
        comparison_only=False,
        cb_splits=8,
        overlay_opacity=0.5,
        overlay_legend=False,
        legend_bbox_to_anchor=None,
        legend_loc='lower left',
        translation=False,
        translation_write_style=None,
        show_mse=False,
        dta_tolerance=5,
        dta_crit=1,
        diff_crit=15,
        suptitle=None,
        show=True,
        include_image=False,
        no_ui=False,
        ylabel_first_only=True,
        yticks_first_only=False,
        ytick_labels_first_only=True,
        colorbar_last_only=True,
        **kwargs,
    ):
        '''
        Display one or more interactive images.

        **Parameters:**

        images : string/Image/np.ndarray/list, default=None
            Source(s) of image data for each plot. If multiple plots are to be
            shown, this must be a list. Image sources can be Image objects, or 
            any of the valid sources for creating an Image object, including:

            (a) The path to a NIfTI or DICOM file;
            (b) The path to a file containing a NumPy array;
            (c) A NumPy array;
            (d) An Image object.

        title : string or list of strings, default=None
            Custom title(s) to use for the image(s) to be displayed. If the
            number of titles given, n, is less than the number of images, only
            the first n figures will be given custom titles. If any titles are
            None, the name of the image file will be used as the title.

        mask : Image/list/ROI/str/StructureSet, default=None
            Image object representing a mask or a source from which
            an Image object can be initialised.  In addition to the
            sources accepted by the Image constructor, the source
            may be an ROI, a list of ROIs or a StructureSet.  In the
            latter cases, the mask image is derived from the ROI mask(s).

        dose : string/nifti/array/list, default=None
            Source(s) of dose field array(s) to overlay on each plot (see valid
            image sources for <images>).

        rois : str/list/dict, default=None
            Locations of files from which to load ROIs. This
            argument can be any of:

            1) String:

               a) The path to a NIfTI file containing an ROI mask;
               b) The path to a DICOM file containing ROI contours;
               c) A wildcard matching one or more of the above file types;
               d) The path to a directory containing NIfTI or DICOM files;
               e) A wildcard matching one or more directories containing
                  NIfTI or DICOM files.

               If the string is found to match a directory, the ROIs 
               from all NIfTI or DICOM files inside that directory will 
               be loaded.

               ROI names will be inferred from the filenames (NIfTI)
               or the ROI names inside the file (DICOM) unless
               the user indicates otherwise in the <roi_names> parameter;
               e.g. an ROI taken from a file called
               'right_parotid.nii.gz' would automatically be called
               'right parotid' in QuickViewer.

               If multiple loaded ROIs have the same names, QuickViewer
               will attempt to label each with a unique name in the UI:

                   - If two ROIs named 'heart' are loaded from different
                     directories dir1 and dir2, these will be labelled
                     'Heart (dir1)' and 'Heart (dir2) in the UI.
                   -  If two ROIs named 'heart' are loaded from
                      different files, file1.nii and file2.nii, these will be
                      labelled 'Heart (file1.nii)' and 'Heart (file2.nii)' in
                      the UI.

               However, if the <legend> option is used, the ROIs
               will be labelled with the same name in the figure legend. See
               the labelling option below in part (3) or the <roi_names>
               option for more customisation.

            2) List:

               a) A list of any of the strings described above; all ROI
                  files found will be loaded.
               b) A list of pairs of paths to files containing ROIs to
                  be compared to one another (see the <roi_comparison>
                  option).

            3) Dict:

               ROI filepaths can be nested inside a dictionary, where
               the keys are labels which the user wishes to use to refer to
               ROIs in those files, and the values are any of the
               options listed above (except 2b).

               The label will be displayed in parentheses next to the
               ROIs names in the QuickViewer UI and ROI legend.

               The <roi_names> and <roi_options> arguments can also
               be nested inside a dictionary if the user wants to apply
               different name and color options to the ROIs associated
               with different labels.

            By default, each NIfTI file will be assumed to contain a single 
            ROI. To load multiple label masks from a single NIfTI file,
            add the string 'multi:' before the filepath, e.g.:

            - rois='multi:my_file.nii'

            or alternatively use the multi_rois parameter.

        roi_names : dict, default=None

        rois_to_keep : list, default=None

        rois_to_remove : list, default=None

        roi_info : bool/list, default=False

            If True, a table containg the volumes and centroids of each plotted
            ROI will be displayed below the plot. Can also be set to a list
            of ROI metrics - see skrt.structures.ROI.get_geometry() documentation
            for list of available metrics. If roi_info=True, metrics will be
            set to ["volume", "centroid", "area"].

        compare_rois : bool/list, default=False

            If True, a table containg the dice scores and centroid distances
            between ROIs will be displayed below the plot. Can also be set to a list
            of ROI comparison metrics - see skrt.structures.ROI.get_comparison() 
            documentation for list of available metrics. 

        show_compared_rois_only : bool, default=True
            If compare_rois is True, only ROIs that are being compared will be
            displayed. E.g. if two StructureSets are given in <rois> that 
            have some ROIs with matching names, only those with matching 
            names will be shown.

        multi_rois : str/list/dict, default=None

            Path(s) to file(s) from which to load multiple ROI label 
            masks per file. 

            Same as the <rois> argument, except each file specified in the
            <multi_rois> argument will be checked for different labels 
            instead of being treated as a binary mask.

            This can be used in conjunction with <rois> to load single masks
            from some files and multiple masks from others.

        roi_consensus : bool, default=False
            If True, add the option to plot the consensus of ROIs rather than
            plotting individually. Only works if a single StructureSet is
            provided for each image.

        grid : string/nifti/array/list, default=None
            Source(s) of grid array(s) to overlay on each plot
            (see valid image sources for <images>).

        jacobian : string/nifti/array/list, default=None
            Source(s) of jacobian determinant array(s) to overlay on each plot
            (see valid image sources for <images>).

        df : string/nifti/array/list, default=None
            Source(s) of deformation field(s) to overlay on each plot
            (see valid image sources for <images>).

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
            only be applied if the number of images in <images> is 2.

        show_overlay : bool, default=False
            If True, a blue/red transparent overlaid image will be displayed.
            This option will only be applied if the number of images in
            <images> is 2.

        show_diff : bool, default=False
            If True, a the difference between two images will be shown. This
            option will only be applied if the number of images in <images>
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

        cb_splits : int, default=8
            Number of sections to show for chequerboard image. Minimum = 1
            (corresponding to no chequerboard). Can later be changed
            interactively.

        overlay_opacity : float, default=0.5
            Initial opacity of overlay. Can later be changed interactively.

        overlay_legend : bool default=False
            If True, a legend will be displayed on the overlay plot.

        legend_bbox_to_anchor : str, default=None
            Bounding box relative to which any legends are to be positioned.
            Must be a valid matplotlib legend bbox_to_anchor.
            See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        legend_loc : str, default='lower left'
            Location for any legends being displayed. Must be a valid
            matplotlib legend location.

        translation : bool, default=False
            If True, widgets will be displayed allowing the user to apply a
            translation to the image. Note that this will not change the Image
            object itself, only the image displayed in the viewer.

        translation_write_style : bool, default="normal"
            Method for writing translations to a file. Can either be "normal"
            (will write x, y, z translations) or "shift" (will shift 
            translations in an elastix transform file).

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

        **Keyword arguments:**

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

        init_view : string, default=None
            Orientation ('x-y', 'y-z', 'x-z' at which to initially display
            the image(s).  If None, the initial view is chosen to match
            the image orienation.

        init_slice : integer, default=None
            Slice number in the initial orientation direction at which to
            display the first image (can be changed interactively later). If
            None, the central slice will be displayed.  Takes precedence over
            <init_idx>.

        init_idx : integer, default=None
            Index in the initial orienation direction of the slice
            in the array to plot.

        init_pos : float, default=None
            Position in mm of the first slice to display. This will be rounded
            to the nearest slice. If <init_pos> and <init_idx> are both given,
            <init_pos> will override <init_idx> only if <scale_in_mm> is True.

        intensity : float/tuple/str, default=None
            Intensity central value or range thresholds at which to display the image.
            Can later be changed interactively. If a single value is given, the
            Intensity range will be centred at this value with width given by
            <intensity_width>. If a tuple is given, the intensity range will be set to the
            two values in the tuple. If None, default intensity will be taken
            from the image itself.  If 'auto', lower and upper bounds are
            set to the minimum and maximum intensity in the image.

        intensity_width : float, default=500
            Initial width of the intensity window. Only used if <intensity> is a single
            value.

        intensity_limits : tuple, default=None
            Full range to use for the intensity slider. Can also set to 'auto'
            to detect min and max intensity in the image. Defaults to
            (-2000, 2000) for Image objects and to 'auto' otherwise (for
            example, for Dose objects).

        intensity_step : float, default=None
            Step size to use for the intensity slider. Defaults to 1 if the maximum
            intensity is >= 10, otherwise 0.1.

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

        colorbar : int/bool, default=False
            Indicate whether to display colour bar(s):
            - 1 or True: colour bar for main image;
            - 2: colour bars for main image and for any associated image
            or overlay;
            - 0 or False: no colour bar.

        colorbar_last_only : bool, default=True
            If True, and multiple plots are to be drawn, colorbar is
            set to False for all plots except the last.

        colorbar_label : str, default=None
            Label for the colorbar and range slider. If None, will default to
            either 'intensity' if an image file is given, or 'Dose (Gy)' if a dose
            file is given without an image.

        clb_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to pyplot.colorbar().
            For information on available keyword arguments, see:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html#matplotlib.pyplot.colorbar

        clb_label_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to colorbar.set_label().
            For information on available keyword arguments, see:
            https://matplotlib.org/stable/api/colorbar_api.html#matplotlib.colorbar.Colorbar.set_label

        no_xlabel : bool, default=False
            If True, the x axis will not be labelled.

        no_xticks : bool, default=False
            If True, ticks and tick labels on the x axis will not
            be shown.

        no_xtick_labels : bool, default=False
            If True ticks on the x axis will not be labelled.
            Disregarded if <no_xticks> is True.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        no_yticks : bool, default=False
            If True, ticks and tick labels on the y axis will not
            be shown.

        no_ytick_labels : bool, default=False
            If True ticks on the y axis will not be labelled.
            Disregarded if <no_yticks> is True.

        ylabel_first_only : bool, default=True
            If True, and multiple plots are to be displayed,
            the y axis will be labelled only for the first plot only.
            If <no_ylabel> is True, the y axis won't be labelled
            for any plot.

        yticks_first_only : bool, default=False
            If True, and multiple plots are to be displayed,
            ticks on the y axis will be shown for the first plot only.
            If <no_yticks> is True, ticks on the y axis won't be
            shown for any plot.

        ytick_labels_first_only : bool, default=True
            If True, and multiple plots are to be displayed, ticks
            on the y axis will be labelled only for the first plot only.
            If <no_ytick_labels> is True, ticks on the y axis won't be
            labelled for any plot.

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

        masked : bool, default=True
            If True and a mask is specified, the image is masked.

        invert_mask : bool, default=False
            If True, any masks applied will be inverted.

        mask_color : matplotlib color, default='black'
            color in which to display masked areas.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).

        grid_opacity : float, default=1.0
            Initial opacity of the overlaid grid. Can later
            be changed interactively.

        grid_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the grid. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.

        jacobian_opacity : float, default=0.8
            Initial opacity of the overlaid jacobian determinant. Can later
            be changed interactively.

        jacobian_range : list, default=None
            Min and max jacobian value to plot. This can also be set via the
            'vmin' and 'xmax' keys in <jacobian_kwargs>; <jacobian_range>
            will take precedence if set.  If None, the values of
            jacobian._default_vmin and jacobian._default_vmax are used.

        jacobian_cmap : str, default=None
            Matplotlib colormap to use for jacobian plotting. Supercedes
            any cmap in <jacobian_kwargs>.  If None, the value of
            jacobian._default_cmap is used.

        jacobian_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the jacobian determinant. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
            Some useful keywords are:

            - 'cmap': colormap (default='seismic').
            - 'interpolation': interpolation method (default='antialiased')

        df_plot_type : str, default='quiver'
            Option for initial plotting of deformation field. Can be 'quiver',
            'grid', 'x-displacement', 'y-displacement',
            'z-displacement', '3d-displacement', or 'none'.
            Can later be changed interactively.
            All quantities relate to the mapping of points from
            fixed image to moving image in image registration.

        df_spacing : int/tuple, default=30
            Spacing between arrows on the quiver plot/gridlines on the grid
            plot of a deformation field. Can be a single value for spacing in
            all directions, or a tuple with values for (x, y, z). Dimensions
            are mm if <scale_in_mm> is True, or voxels if <scale_in_mm> is
            False.

        df_opacity : float, default=0.5
            Initial opacity of the overlaid deformation field. Can later
            be changed interactively.


        df_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib when plotting
            the deformation field.  Note that different keyword arguments
            are accepted for different plot types.

            For grid plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            Some useful keywords are:

            - 'linewidth': default=2
            - 'color': default='green'

            For quiver plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html.

        roi_plot_type : str, default='contour'
            Option for initial plot of ROIs. Can be 'contour', 'mask',
            'filled', or 'none'. Can later be changed interactively.

        roi_opacity : float, default=None
            Initial opacity of ROIs when plotted as masks or filled
            contours. Can later be changed interactively. Default=1 for masks,
            0.3 for filled contours.

        roi_linewidth : float, default=2
            Initial linewidth of ROIs when plotted as contours. Can later
            be changed interactively.

        roi_info : bool, default=False
            If True, the lengths and volumes of each ROI will be
            displayed below the plot.

        roi_info_dp : int, default=2
            Number of decimal places to show for floats in ROI info and
            comparison tables.

        length_units : str, default=None
            Units in which to display the lengths of ROIs if
            <roi_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:

            (a) 'mm'
            (b) 'voxels'

        area_units : str, default=None
            Units in which to display the areas of ROIs if
            <roi_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:

            (a) 'mm'
            (b) 'voxels'

        vol_units : str, default=None
            Units in which to display the volumes of ROIs if
            <roi_info> if True. If None, units will be voxels if
            <scale_in_mm> is False, or mm^3 if <scale_in_mm> is True. Options:

            (a) 'mm' for mm^3
            (b) 'voxels' for voxels
            (c) 'ml' for ml

        legend : bool, default=False
            If True, a legend will be displayed for any plot with ROIs.

        init_roi : str, default=None
            If set to an ROI name, the first slice to be displayed will
            be the central slice of that ROI. This supercedes <init_pos>
            and <init_slice>.

        roi_names : list/dict, default=None
            Custom names for ROIs.

            If only one ROI is to be loaded per file, this should be a
            dictionary where the keys are the desired custom names. The values
            of the dictionary can be either:

            (a) A string containing the path of the file containing the 
                ROI to be renamed;
            (b) A string containing the name of the ROI to be renamed 
                within a DICOM file;
            (c) A string containing the automatically-generated name of the
                ROI to be renamed (e.g. if the ROI came from a 
                file right_parotid.nii, its automatically generated name
                would be 'right parotid';
            (d) A wildcard matching any of the above;
            (e) A list of any of the above.

            The list functionality allows the user to list multiple names
            that should be replaced by a single custom name, e.g. to handle
            cases where the same ROI may have different names in 
            different DICOM files.

            If multiple ROIs are to be loaded from files (i.e. the 
            <multi_rois> parameter is set, or paths in the <rois> 
            parameter are prefixed with 'multi:') the <roi_names> parameter
            can either be:

            (a) A list of names, where the Nth name in the list will be
                applied to the ROI with mask label N in the ROI
                array; or
            (b) A dictionary where the keys are integers such that the
                name associated with key N will be applied to the ROI
                with mask label N in the ROI array.

            Any of the options described above can also be nested into a
            dictionary where the keys are labels, if a label dictionary was
            used to load ROIs in the <rois> parameter. The nested
            options for each key will only be applied to ROIs whose
            label is that key.

        roi_colors : dict, default=None
            A dictionary mapping ROI names to colors in which the
            ROI will be displayed.

            The dictionary keys should be either ROI names or wildcards
            matching ROI name(s). Note that ROI names are inferred
            from the ROI's filename unless otherwise specified in the
            <roi_names> parameter.

            The values of the dictionary can be any matplotlib colorlike
            object.

            The color dictionary can also be nested into a dictionary where the
            keys are labels, if a label dictionary was used to load ROIs
            in the <rois> parameter. The nested options for each key will
            only be applied to ROIs whose label is that key.

        rois_as_mask : bool, default=True
            If True, any loaded ROIs will be used to mask the image and
            dose field.

        compare_rois : bool, default=False
            If True, slice-by-slice comparison metrics for pairs of ROIs
            will be displayed below the plot, and compared ROI masks
            will be plotted with their overlapping region in a different
            colour.

            The ROIs to compare can be set in three different ways:

            a) The user can explicitly indicate which ROIs should be
               compared by giving a list of lists for the <rois>
               argument, where each sublist contains exactly 2 filepaths
               corresponding to a pair of ROIs to be compared (see
               option 2b for <rois>).

            b) If only two ROIs are found for the filepaths/wildcards
               given in the <rois> option, these two ROIs will
               be compared.

            c) Otherwise, QuickViewer will search for pairs of loaded
               ROIs with the same name (either inferred from the
               filenames or specified manually by the user in the
               <roi_names> option). If no ROIs with matching names
               are found, no comparisons will be performed.

        ignore_empty_rois : bool, default=False
            If True, any loaded ROIs array that only contains zeros will
            be ignored. If False, the names of empty ROIs will be
            displayed in the UI with '(empty)' next to them.

        ignore_unpaired_rois : bool, default=False
            If <roi_comparison> is True and ROI pairs are
            automatically detected based on their names, this parameter
            determines whether any ROIs for which a matching name is not
            found should be displayed.

            If True, only the pairs of ROIs with matching names will be
            shown. If False, all loaded ROIs will be shown regardless of
            whether they have a comparison match.

        rois_to_keep : list, default=True
            List of ROI names or wildcards matching ROIs that you
            wish to load. All other ROIs will be ignored.

        rois_to_ignore : list, default=True
            List of ROI names or wildcards matching ROIs that you
            wish to ignore.

        autoload_rois : bool, default=True
            If True, ROIs will all be automatically loaded and plotted.
            If False, all ROIs will be initially turned off and will 
            only be loaded if the user turns them on via the ROI checkbox
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
            unless rois are loaded, in which case default is True.

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

        include_image : bool, default=False
            If True, and image has associated image, overlay former on the
            latter.

        no_ui : bool, default=False
            If True, omit user-interface elements (Jupyter widgets)
            for interaction with viewed image(s).  This can be useful
            for including graphics in stored notebooks.  On GitHub, for
            example, static plots in notebooks are rendered correctly,
            but Jupyter widgets can cause problems.
        '''

        # Get image file inputs
        if not isinstance(images, list) or isinstance(images, tuple):
            self.images = [images]
        else:
            self.images = images
        self.n = len(self.images)

        # Process other inputs
        self.title = self.get_input_list(title)
        self.dose = self.get_input_list(dose)
        self.mask = self.get_input_list(mask)
        self.rois = self.get_input_list(rois, allow_sublist=True)
        #  self.multi_rois = self.get_input_list(multi_rois, allow_sublist=True)
        self.grid = self.get_input_list(grid)
        self.jacobian = self.get_input_list(jacobian)
        self.df = self.get_input_list(df)

        # Define whether to omit user-interface elements.
        self.no_ui = no_ui

        self.colorbar = kwargs.pop('colorbar', False)

        # Set options for deformation field.
        self.df_kwargs = kwargs.get("df_kwargs", {})
        self.df_plot_type = kwargs.get("df_plot_type", "quiver")
        self.df_spacing = kwargs.get("df_spacing", 30)

        # Set options for Jacobian determinant.
        self.jacobian_kwargs = kwargs.get("jacobian_kwargs", {})
        jacobian_cmap = kwargs.get("cmap", None)
        if jacobian_cmap is not None:
            self.jacobian_kwargs["cmap"] = jacobian_cmap
        elif jacobian:
            for item in self.jacobian:
                if item is not None:
                    if isinstance(item, Jacobian):
                        jac = item
                    else:
                        jac = Jacobian(item)
                    self.jacobian_kwargs["cmap"] = self.jacobian_kwargs.get(
                            "cmap", jac._default_cmap)
                    break

        # Define default handling for y-axis label, y-axis tick labels,
        # and colour bar.
        no_ylabel_default = kwargs.pop("no_ylabel", False)
        no_yticks_default = kwargs.pop("no_yticks", False)
        no_ytick_labels_default = kwargs.pop("no_ytick_labels", False)
        colorbar_default = self.colorbar

        # Make individual viewers
        self.scale_in_mm = scale_in_mm
        self.viewers = []
        viewer_type = SingleViewer if not orthog_view else OrthogViewer
        kwargs = {key.replace('colour', 'color'): val for key, val in kwargs.items()}
        mask_threshold = kwargs.get("mask_threshold", 0.5)

        for i in range(self.n):
            
            no_ylabel = (no_ylabel_default if
                    (i == 0 or not ylabel_first_only) else True)
            no_yticks = (no_yticks_default if
                    (i == 0 or not yticks_first_only) else True)
            no_ytick_labels = (no_ytick_labels_default if
                    (i == 0 or not ytick_labels_first_only) else True)
            colorbar = (colorbar_default if
                    (i + 1 == self.n or not colorbar_last_only) else False)

            viewer = viewer_type(
                self.images[i],
                title=self.title[i],
                dose=self.dose[i],
                mask=get_mask(self.mask[i], mask_threshold, self.images[i]),
                rois=self.rois[i],
                #  multi_rois=self.multi_rois[i],
                grid=self.grid[i],
                jacobian=self.jacobian[i],
                df=self.df[i],
                standalone=False,
                scale_in_mm=scale_in_mm,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_loc=legend_loc,
                include_image=include_image,
                colorbar=colorbar,
                no_ylabel=no_ylabel,
                no_yticks=no_yticks,
                no_ytick_labels=no_ytick_labels,
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
        self.legend_bbox_to_anchor = legend_bbox_to_anchor
        self.legend_loc = legend_loc
        self.comparison_only = comparison_only
        if (
            comparison_only
            and comparison is None
            and not any([show_cb, show_overlay, show_diff])
        ):
            comparison = True
        self.load_comparison(comparison)
        self.translation = translation
        self.translation_write_style = translation_write_style
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
        self.comp_colorbar = self.colorbar and self.comparison_only
        self.zoom = kwargs.get('zoom', None)
        self.plots_per_row = plots_per_row
        self.suptitle = suptitle
        self.custom_ax_lims = {view: [None, None] for view in _plot_axes}
        self.match_axes(match_axes)
        self.in_notebook = in_notebook()
        self.saved = False
        self.plotting = False
        self.share_slider = share_slider

        # Make UI
        #  if any([v.image.timeseries for v in self.viewers]):
            #  share_slider = False
        self.make_ui()

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

        return any([getattr(v, 'has_' + attr) for v in self.viewers])

    def load_comparison(self, comparison):
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
        for name in ['chequerboard', 'overlay', 'difference']:
            setattr(self, f'has_{name}', name in comp_opts)

        # Case with no comparison images
        if not self.has_multicomp and not len(comp_opts):
            return

        # Use first two images
        assert self.n > 1
        im1 = self.viewers[0].image
        im2 = self.viewers[1].image

        # Make individual comparisons
        for comp in comp_opts:
            name = 'multicomp' if comp == 'all' else comp
            plot_type = None if comp == 'all' else comp
            comp_im = ImageComparison(
                im1, im2, plot_type=plot_type 
            )
            setattr(self, name, comp_im)
            self.comparison[name] = comp_im

    def match_axes(self, match_axes):
        '''Adjust axes of plots to match if the match_axes option is set.'''

        if match_axes is None:
            return

        # Match axes in all orientations
        all_ax_lims = {}
        for view in _slice_axes:

            # Calculate limits using all plots
            ax_lims = []
            if match_axes in ['all', 'both', 'x', 'y', 'overlap']:
                for i_ax in _plot_axes[view]:
                    min_lims = [v.image.image_extent[i_ax][0]
                                for v in self.viewers]
                    max_lims = [v.image.image_extent[i_ax][1]
                                for v in self.viewers]
                    f1, f2 = min, max
                    if match_axes == 'overlap':
                        f1, f2 = f2, f1
                    if min_lims[0] > max_lims[0]:
                        f1, f2 = f2, f1
                    ax_lims.append([f1(min_lims), f2(max_lims)])

            # Match axes to one plot
            else:
                try:
                    im = self.viewers[match_axes].image
                    for i_ax in _plot_axes[view]:
                        ax_lims.append(im.plot_extent[i_ax])

                except TypeError:
                    raise TypeError('Unrecognised option for <match_axes>', 
                                    match_axes)

            all_ax_lims[view] = ax_lims

        # Set these limits for all viewers and self
        for v in [self] + self.viewers:
            for view in all_ax_lims:
                if match_axes != 'y':
                    v.custom_ax_lims[view][0] = all_ax_lims[view][0]
                if match_axes != 'x':
                    v.custom_ax_lims[view][1] = all_ax_lims[view][1]

    def make_ui(self, no_roi=False, no_intensity=False):

        # Only allow share_slider if images have same frame of reference
        if self.share_slider:
            self.share_slider *= all(
                [v.image.has_same_geometry(self.viewers[0].image) for v in self.viewers]
            )

        # Make UI for first image
        v0 = self.viewers[0]
        v0.make_ui(no_roi=no_roi, no_intensity=no_intensity)

        # Store needed UIs
        self.ui_view = v0.ui_view
        self.view = self.ui_view.value
        self.ui_roi_plot_type = v0.ui_roi_plot_type
        self.ui_roi_consensus_switch = v0.ui_roi_consensus_switch
        self.ui_roi_consensus_type = v0.ui_roi_consensus_type
        self.roi_plot_type = self.ui_roi_plot_type.value

        # Make main upper UI list (= view radio + single intensity/slice slider)
        many_sliders = not self.share_slider and self.n > 1
        if not many_sliders:
            self.main_ui = v0.main_ui
            if v0.zoom_ui:
                v0.ui_zoom_reset.on_click(self.make_reset_zoom(v0))
        else:
            self.main_ui = [self.ui_view]

        # Make UI for other images
        for v in self.viewers[1:]:
            v.make_ui(other_viewer=v0, share_slider=self.share_slider,
                      no_roi=no_roi, no_intensity=no_intensity)
            v0.rois_for_jump.update(v.rois_for_jump)
        v0.ui_roi_jump.options = list(v0.rois_for_jump.keys())

        # Make UI for each image (= unique intensity/slice sliders and ROI jump)
        self.per_image_ui = []
        if many_sliders:
            for v in self.viewers:

                # ROI jumping
                sliders = []
                if v.has_rois:
                    sliders.append(v.ui_roi_jump)
                else:
                    if self.any_attr('rois'):
                        sliders.append(ipyw.Label())

                # Intensity sliders
                if not no_intensity:
                    if v.intensity_from_width:
                        sliders.extend([v.ui_intensity_centre, v.ui_intensity_width])
                    else:
                        sliders.append(v.ui_intensity)

                # Zoom sliders
                if v.zoom_ui:
                    sliders.extend(v.all_zoom_ui)
                    v.ui_zoom_reset.on_click(self.make_reset_zoom(v))

                # Slice slider
                sliders.append(v.ui_slice)
                self.per_image_ui.append(sliders)

        # Make extra UI list
        self.extra_ui = []
        for attr in ["mask", "dose", "df"]:
            if self.any_attr(attr):
                self.extra_ui.append(getattr(v0, 'ui_' + attr))
                if "df" == attr:
                    self.extra_ui.extend([v0.ui_df_spacing, v0.ui_df_opacity])
                if "mask" == attr:
                    self.extra_ui.append(v0.ui_mask_invert)
        if self.any_attr('jacobian'):
            cmap_name = getattr(self.jacobian_kwargs["cmap"], "name", None)
            if cmap_name == "jacobian":
                self.extra_ui.extend([v0.ui_jac_opacity])
            else:
                self.extra_ui.extend([v0.ui_jac_opacity, v0.ui_jac_range])
        if self.any_attr('grid'):
            self.extra_ui.extend([v0.ui_grid_opacity])
        if self.any_attr('rois'):
            to_add = [
                v0.ui_roi_plot_type,
                v0.ui_roi_linewidth,
                v0.ui_roi_opacity,
                 v0.ui_roi_select_all,
                 v0.ui_roi_deselect_all,
            ]
            if any([v.roi_consensus for v in self.viewers]):
                to_add.append(v0.ui_roi_consensus_switch)
                to_add.append(v0.ui_roi_consensus_type)
            self.extra_ui.extend(to_add)

        # Make extra UI elements
        self.make_lower_ui(no_roi=no_roi)
        self.make_comparison_ui()
        self.make_translation_ui()

        # Assemble UI boxes
        main_and_extra_box = ipyw.HBox(
            [
                ipyw.VBox(self.main_ui),
                ipyw.VBox(self.extra_ui),
                ipyw.VBox(self.translation_ui),
                ipyw.VBox(self.comp_ui),
            ]
        )
        self.slider_boxes = [ipyw.VBox(ui) for ui in self.per_image_ui]
        self.set_slider_widths()
        self.upper_ui = [main_and_extra_box, ipyw.HBox(self.slider_boxes)]
        self.upper_ui_box = ipyw.VBox(self.upper_ui)
        self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)
        self.all_ui = (
            self.main_ui
            + self.extra_ui
            + list(itertools.chain.from_iterable(self.per_image_ui))
            + self.comp_ui
            + self.translation_ui
            + self.ui_roi_checkboxes
            + [self.trigger]
        )

    def make_lower_ui(self, no_roi=False):
        '''Make lower UI for ROI checkboxes.'''

        # Saving UI
        self.lower_ui = []

        # ROI UI
        many_with_rois = sum([v.has_rois for v in self.viewers]) > 1
        self.ui_roi_checkboxes = []
        for i, v in enumerate(self.viewers):

            # Add plot title to ROI UI
            if many_with_rois and v.has_rois:
                if not hasattr(v.image, 'title') or not v.image.title:
                    title = f'<b>Image {i + 1}</b>'
                else:
                    title = f'<b>{v.image.title + ":"}</b>'
                self.lower_ui.append(ipyw.HTML(value=title))

            # Add to overall lower UI
            if not no_roi or v.roi_info or v.compare_rois:
                self.lower_ui.extend(v.lower_ui)
                self.ui_roi_checkboxes.extend(v.ui_roi_checkboxes)

        self.lower_ui.append(self.viewers[0].ui_save_plot)

    def make_comparison_ui(self):

        self.comp_ui = []

        # Multicomparison dropdown
        comp_opts = [
            'overlay',
            'chequerboard',
        ]
        if all([v.image.has_same_geometry(self.viewers[0].image) for v in self.viewers]):
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

        self.translation_ui = []
        if not self.translation:
            return

        self.translation_viewer = self.viewers[int(self.n > 1)]
        self.translation_ui.append(ipyw.HTML(value='<b>Translation:</b>'))

        # Make translation file UI
        self.translation_output = ipyw.Text(description='Save as:',
                                            value="translation.txt")
        self.tbutton = ipyw.Button(description='Write translation')
        self.tbutton.on_click(self.write_translation_to_file)
        self.translation_ui.extend([self.translation_output, self.tbutton])

        # Make translation sliders
        self.tsliders = {}
        for i, ax in enumerate(_axes):
            vx = abs(self.translation_viewer.image.voxel_size[i])
            n = self.translation_viewer.image.n_voxels[i]
            self.tsliders[ax] = ipyw.FloatSlider(
                min=-n * vx,
                max=n * vx,
                value=0,
                description=f'{ax} (mm)',
                continuous_update=False,
                step=vx,
                style=_style
            )
            self.translation_ui.append(self.tsliders[ax])
        self.translation_viewer.shift = [self.tsliders[ax].value for ax in _axes]

    def write_translation_to_file(self, _):
        '''Write current translation to file.'''

        output_file = self.translation_output.value
        if self.translation_write_style == "normal":
            out_text = ''.join(f'{ax} {self.tsliders[ax].value}\n' 
                               for ax in _axes)
            outfile = open(output_file, 'w')
            outfile.write(out_text)
            outfile.close()
        elif self.translation_write_style == "shift": 
            from skrt.registration import shift_translation_parameters
            shift_translation_parameters(
                output_file, 
                *[self.tsliders[ax].value for ax in _axes]
            )
        else:
            print("Unrecognised translation writing option: " +
                  self.translation_write_style)
            return
        print('Wrote translation to file:', output_file)

        # Reapply transformation if using a Registration object
        if self.translation_write_style == "shift" and \
           hasattr(self, "_registration"):
            self._registration.transform_moving_image(
                step=self._registration_step)

    def apply_translation(self):
        '''Update the description of translation sliders to show translation
        in mm if the translation is changed.'''

        new_shift = [self.tsliders[ax].value for ax in _axes]
        if new_shift == self.translation_viewer.shift:
            return
        self.translation_viewer.shift = new_shift

    def set_slider_widths(self):
        '''Adjust widths of slider UI.'''

        if self.plots_per_row is not None and self.plots_per_row < self.n:
            return
        for i, slider in enumerate(self.slider_boxes[:-1]):
            width = (
                self.figsize
                * self.viewers[i].image.get_plot_aspect_ratio(
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

    def make_fig(self, view_changed):

        # Get relative width of each subplot
        width_ratios = [v.image.get_plot_aspect_ratio(
            self.view, self.zoom, abs(self.colorbar))
            for v in self.viewers
        ]

        width_ratios.extend(
            [
                c.get_plot_aspect_ratio(self.view, self.zoom, 
                                        self.comp_colorbar, self.figsize)
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

        # Outside notebook, just resize figure if orientation has changed
        if not self.in_notebook and hasattr(self, 'fig'):
            if view_changed:
                self.fig.set_size_inches(width, height)
            return

        # Make new figure
        # !!!!
        # Not clear why, but the following call seems to change
        # the value of mpl.rcParams["font.size"]
        # !!!!
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

        # Deal with hiding/showing all ROIs
        for v in self.viewers:
            # Hide all ROIs
            if v.ui_roi_deselect_all.value:
                for roi in v.rois:
                    roi.checkbox.value = False
                v.ui_roi_deselect_all.value = False

            # Show all ROIs
            if v.ui_roi_select_all.value:
                for roi in v.rois:
                    roi.checkbox.value = True
                v.ui_roi_select_all.value = False

        # Deal with view change
        view_changed = self.ui_view.value != self.view
        if view_changed:
            self.view = self.ui_view.value
            for v in self.viewers:
                v.view = self.ui_view.value
                v.on_view_change()
            self.set_slider_widths()

        # Deal with ROI plot type change
        if self.roi_plot_type != self.ui_roi_plot_type.value:
            self.roi_plot_type = self.ui_roi_plot_type.value
            self.viewers[0].update_roi_sliders()

        # Deal with ROI jumps
        for v in self.viewers:
            if v.ui_roi_jump != '':
                v.jump_to_roi()

        # Apply any translations
        if self.translation:
            self.apply_translation()

        # Reset figure
        self.make_fig(view_changed)

        # Plot all images
        for v in self.viewers:
            if self.comparison_only:
                v.set_slice_and_view()
                #  v.image.set_slice(self.view, v.slice[self.view])
            else:
                v.plot()

        # Adjust comparison UI
        multicomp_plot_type = self.ui_multicomp.value
        if self.has_multicomp and len(self.comparison) == 1:
            self.ui_cb.disabled = not multicomp_plot_type == 'chequerboard'
            self.ui_overlay.disabled = not multicomp_plot_type == 'overlay'

        # Deal with comparisons
        if len(self.comparison):

            # Get settings
            invert = self.ui_invert.value
            plot_kwargs = self.viewers[0].v_min_max
            if self.viewers[0].mpl_kwargs is not None:
                plot_kwargs.update(self.viewers[0].mpl_kwargs)

            # Plot all comparisons
            for name, comp in self.comparison.items():
                plot_type = None if name != 'multicomp' else multicomp_plot_type
                SingleViewer.plot_image(
                    self,
                    comp,
                    view=self.viewers[0].view,
                    sl=[self.viewers[0].slice[self.viewers[0].view],
                        self.viewers[1].slice[self.viewers[1].view]],
                    invert=invert,
                    plot_type=plot_type,
                    cb_splits=self.ui_cb.value,
                    overlay_opacity=self.ui_overlay.value,
                    overlay_legend=self.overlay_legend,
                    overlay_legend_bbox_to_anchor=self.legend_bbox_to_anchor,
                    overlay_legend_loc=self.legend_loc,
                    zoom=self.viewers[0].zoom,
                    zoom_centre=self.viewers[0].zoom_centre,
                    mpl_kwargs=self.viewers[0].v_min_max,
                    colorbar=self.comp_colorbar,
                    colorbar_label=self.viewers[0].colorbar_label,
                    clb_kwargs=self.viewers[-1].clb_kwargs,
                    clb_label_kwargs=self.viewers[-1].clb_label_kwargs,
                    major_ticks=self.viewers[-1].major_ticks,
                    minor_ticks=self.viewers[-1].minor_ticks,
                    ticks_all_sides=self.viewers[-1].ticks_all_sides,
                    no_xlabel=self.viewers[-1].no_xlabel,
                    no_ylabel=self.viewers[-1].no_ylabel,
                    no_xticks=self.viewers[-1].no_xticks,
                    no_yticks=self.viewers[-1].no_yticks,
                    no_xtick_labels=self.viewers[-1].no_xtick_labels,
                    no_ytick_labels=self.viewers[-1].no_ytick_labels,
                    show_mse=self.show_mse,
                    dta_tolerance=self.dta_tolerance,
                    dta_crit=self.dta_crit,
                    diff_crit=self.diff_crit,
                    show=False,
                    use_cached_slices=(not self.comparison_only),
                    xlim=self.custom_ax_lims[self.viewers[0].view][0],
                    ylim=self.custom_ax_lims[self.viewers[0].view][1],
                    scale_in_mm=self.scale_in_mm
                )

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
        init_view=None,
        init_slice=None,
        init_idx=None,
        init_pos=None,
        intensity=None,
        intensity_width=500,
        intensity_limits=None,
        intensity_step=None,
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
        clb_kwargs=None,
        clb_label_kwargs=None,
        no_xlabel=False,
        no_ylabel=False,
        no_xticks=False,
        no_yticks=False,
        no_xtick_labels=False,
        no_ytick_labels=False,
        mpl_kwargs=None,
        dose=None,
        dose_opacity=0.5,
        dose_kwargs=None,
        dose_range=None,
        dose_cmap=None,
        masked=True,
        invert_mask=False,
        mask=None,
        mask_color="black",
        grid=None,
        grid_opacity=1.0,
        grid_kwargs=None,
        jacobian=None,
        jacobian_opacity=0.8,
        jacobian_range=None,
        jacobian_cmap=None,
        jacobian_kwargs=None,
        df=None,
        df_plot_type="quiver",
        df_opacity=None,
        df_spacing=30,
        df_kwargs=None,
        rois=None,
        roi_names=None,
        rois_to_keep=None,
        rois_to_remove=None,
        roi_plot_type="contour",
        roi_opacity=None,
        roi_linewidth=2,
        roi_info=False,
        roi_info_dp=2,
        roi_kwargs=None,
        compare_rois=False,
        show_compared_rois_only=True,
        length_units="mm",
        area_units="mm",
        vol_units="mm",
        legend=False,
        legend_bbox_to_anchor=None,
        legend_loc="lower left",
        init_roi=None,
        roi_consensus=False,
        consensus_color="blue",
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
        title=None,
        include_image=False,
        **kwargs,
    ):

        # Make Image
        self.image = self.make_image(im, **kwargs)
        self.gs = None  # Gridspec in which to place plot axes
        self.scale_in_mm = scale_in_mm
        self.title = title

        # Load additional overlays
        self.load_dose(dose)
        self.load_df(df)
        self.load_jacobian(jacobian)
        self.load_grid(grid)

        # Load ROIs
        self.init_roi = init_roi
        self.compare_rois = compare_rois
        self.show_compared_rois_only = show_compared_rois_only
        self.roi_consensus = roi_consensus
        self.consensus_color = consensus_color
        self.load_rois(rois, roi_names=roi_names, rois_to_keep=rois_to_keep,
                       rois_to_remove=rois_to_remove)

        # Set initial orientation
        if not init_view:
            view = self.image.get_orientation_view()
            if view:
                init_view = view
            else:
                init_view = 'x-y'
        view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
        if init_view in view_map:
            self.view = view_map[init_view]
        else:
            self.view = init_view

        # Set initial slice number for each orientation
        self.slice = {
            view: np.ceil(self.image.n_voxels[z] / 2) 
            for view, z in _slice_axes.items()
        }
        if init_pos is not None and self.scale_in_mm:
            self.set_slice_from_pos(init_view, init_pos)
        elif init_idx is not None:
            self.set_slice_from_idx(init_view, init_idx)
        else:
            self.set_slice(init_view, init_slice)

        # Assign plot settings
        # Intensity range settings
        # Set initial intensity range from range or window
        self.intensity = intensity
        if intensity is None:
            self.intensity = [
                self.image._default_vmin,
                self.image._default_vmax
            ]
        elif 'auto' == intensity:
            self.intensity = [
                self.image.get_data().min(),
                self.image.get_data().max(),
            ]
        self.intensity_from_width = isinstance(intensity, float) \
                or isinstance(intensity, int)
        self.intensity_width = intensity_width

        # Set upper and lower limits of the intensity slider
        self.intensity_limits = intensity_limits
        if intensity_limits is None:
            if 'auto' == intensity:
                self.intensity_limits = self.intensity
            else:
                if type(self.image) == Image:
                    self.intensity_limits = [-2000, 2000]
                else:
                    self.intensity_limits = 'auto'

        if self.intensity_limits == 'auto':
            self.intensity_limits = [self.image.data.min(),
                    self.image.data.max()]

        # Ensure limits extend to the initial intensity range
        if self.intensity[0] < self.intensity_limits[0]:
            self.intensity_limits[0] = self.intensity[0]
        if self.intensity[1] > self.intensity_limits[1]:
            self.intensity_limits[1] = self.intensity[1]

        # Get step size for intensity slider
        self.intensity_step = intensity_step
        if intensity_step is None:
            self.intensity_step = (
                1 if abs(self.intensity_limits[1] - self.intensity_limits[0]) 
                >= 10 else 0.1
            )

        # Other settings
        self.in_notebook = in_notebook()
        self.mpl_kwargs = mpl_kwargs if mpl_kwargs else {}
        self.figsize = to_inches(figsize)
        self.continuous_update = continuous_update
        self.colorbar = colorbar
        self.colorbar_drawn = False
        self.colorbar_label = colorbar_label
        self.clb_kwargs = clb_kwargs
        self.clb_label_kwargs = clb_label_kwargs
        self.no_xlabel = no_xlabel
        self.no_ylabel = no_ylabel
        self.no_xticks = no_xticks
        self.no_yticks = no_yticks
        self.no_xtick_labels = no_xtick_labels
        self.no_ytick_labels = no_ytick_labels
        self.annotate_slice = annotate_slice
        if self.annotate_slice is None and not self.in_notebook:
            self.annotate_slice = True
        self.save_as = str(save_as) if save_as is not None else None
        self.plotting = False
        self.callbacks_set = False
        self.standalone = standalone
        self.custom_ax_lims = {
            'x-y': [xlim, ylim],
            'y-z': [zlim, ylim],
            'x-z': [zlim, xlim]
        }
        self.zoom = zoom
        self.zoom_centre = zoom_centre
        self.zoom_ui = zoom_ui
        if zoom_ui is None:
            self.zoom_ui = bool(self.has_rois)
        self.major_ticks = major_ticks
        self.minor_ticks = minor_ticks
        self.ticks_all_sides = ticks_all_sides
        self.no_axis_labels = no_axis_labels
        self.legend_bbox_to_anchor = legend_bbox_to_anchor
        self.legend_loc = legend_loc
        self.shift = [None, None, None]
        self.include_image = include_image

        # Mask settings
        self.mask = mask
        self.masked = masked
        self.invert_mask = invert_mask
        self.mask_color = mask_color
        self.has_mask = bool(mask)

        # Overlay plot settings
        self.init_dose_opacity = dose_opacity
        self.dose_kwargs = dose_kwargs if dose_kwargs is not None else {}
        if dose_range is not None:
            self.dose_kwargs["vmin"] = dose_range[0]
            self.dose_kwargs["vmax"] = dose_range[1]
        if dose_cmap is not None:
            self.dose_kwargs["cmap"] = dose_cmap
        self.init_grid_opacity = grid_opacity
        self.grid_kwargs = grid_kwargs if grid_kwargs is not None else {}
        self.init_jacobian_opacity = jacobian_opacity
        self.jacobian_kwargs = jacobian_kwargs if jacobian_kwargs is not None else {}
        if jacobian_range is not None:
            self.jacobian_kwargs["vmin"] = jacobian_range[0]
            self.jacobian_kwargs["vmax"] = jacobian_range[1]
        elif jacobian:
            self.jacobian_kwargs["vmin"] = self.jacobian_kwargs.get(
                    "vmin", jacobian._default_vmin)
            self.jacobian_kwargs["vmax"] = self.jacobian_kwargs.get(
                    "vmax", jacobian._default_vmax)

        if jacobian_cmap is not None:
            self.jacobian_kwargs["cmap"] = jacobian_cmap
        elif jacobian:
            self.jacobian_kwargs["cmap"] = self.jacobian_kwargs.get(
                    "cmap", jacobian._default_cmap)

        try:
            self.init_jacobian_range = [self.jacobian_kwargs[key] for
                    key in ["vmin", "vmax"]]
        except KeyError:
            self.init_jacobian_range = None

        self.df_kwargs = df_kwargs
        self.init_df_opacity = df_opacity if df_opacity is not None else 0.5
        self.df_plot_type = df_plot_type
        self.df_spacing = df_spacing

        # ROI settings
        self.roi_plot_type = roi_plot_type
        self.roi_mask_opacity = 1
        self.roi_filled_opacity = 0.3
        if roi_opacity is not None:
            if roi_plot_type == 'mask':
                self.roi_mask_opacity = roi_opacity
            elif roi_plot_type in ['filled', 'filled centroid']:
                self.roi_filled_opacity = roi_opacity
        self.roi_linewidth = roi_linewidth
        self.legend = legend
        self.legend_bbox_to_anchor = legend_bbox_to_anchor
        self.legend_loc = legend_loc
        self.roi_info = roi_info
        self.roi_info_dp = roi_info_dp
        self.roi_metrics = roi_info if is_list(roi_info) \
            else ["volume", "centroid", "area"]
        self.force_roi_geometry_calc = True
        self.force_roi_comp_calc = True
        self.roi_vol_units = vol_units
        self.roi_area_units = area_units
        self.roi_length_units = length_units
        self.roi_kwargs = roi_kwargs if roi_kwargs is not None else {}
        self.roi_kwargs['roi_colors'] = {}
        if 'roi_colors' in kwargs:
            for roi_name, roi_color in kwargs['roi_colors'].items():
                self.roi_kwargs['roi_colors'][roi_name] = mpl.colors.to_rgba(
                        roi_color)
        self.roi_comp_metrics = compare_rois if is_list(compare_rois) \
                else None

        # Colormap
        if cmap:
            self.mpl_kwargs["cmap"] = cmap
        #  elif self.image.dose_as_im:
            #  self.mpl_kwargs["cmap"] = dose_cmap if dose_cmap else "jet"

        # Display plot
        if standalone:
            self.make_ui()
            self.show(show)

    def make_image(self, im, *args, **kwargs):
        '''Set up image object.'''

        if isinstance(im, Image):
            image = im
        else:
            image = Image(im, *args, **kwargs)
        image.load()

        # Ensure fig and ax are clear
        for attr in ['ax', 'fig']:
            if hasattr(image, attr):
                delattr(image, attr)

        return image

    def load_dose(self, dose):
        """Load dose field."""

        if dose is None:
            self.dose = None
        elif isinstance(dose, Dose):
            self.dose = dose
        elif isinstance(dose, int):
            self.dose = self.image.doses[dose]
        else:
            self.dose = Dose(dose)

        self.has_dose = self.dose is not None

    def load_df(self, df):
        """Load deformation field."""

        if df is None:
            self.df = None
        elif isinstance(df, DeformationField):
            self.df = df
        else:
            self.df = DeformationField(df)

        self.has_df = self.df is not None

    def load_rois(
        self, 
        rois,
        roi_names=None,
        rois_to_keep=None, 
        rois_to_remove=None
    ):
        """
        Load ROIs, apply names/filters, and assign to a single StructureSet.
        """

        # Current ROI attribute (used for ROI centering/jumping)
        self.current_roi = ''

        # Load all ROIs into structure sets
        structure_sets = []
        standalone_rois = StructureSet()  # Extra structure set for single ROIs
        if not is_list(rois):
            rois = [rois]
        for roi in rois:
            if roi is None:
                continue
            elif isinstance(roi, int): 
                structure_sets.append(self.image.structure_sets[roi])
            elif isinstance(roi, StructureSet):
                structure_sets.append(roi)
            elif isinstance(roi, ROI):
                standalone_rois.add_roi(roi)
            else:
                structure_sets.append(StructureSet(roi, image=self.image))
        if len(standalone_rois.get_rois()):
            structure_sets.append(standalone_rois)

        # Load all structure sets and apply filters
        structure_sets_filtered = []
        for ss in structure_sets:
            ss.load()
            structure_sets_filtered.append(
                ss.filtered_copy(names=roi_names, to_keep=rois_to_keep,
                                 to_remove=rois_to_remove, copy_roi_data=False)
            )

        # Get list of all ROI names and StructureSet names
        all_names = []
        ss_names = []
        for ss in structure_sets_filtered:
            all_names.extend(ss.get_roi_names())
            ss_names.append(ss.name)

        # Make list of ROI comparison pairs
        if self.compare_rois:
            if len(structure_sets_filtered) > 2:
                raise RuntimeError("Unable to compare ROIs from more than two "
                                   "StructureSets")

            if self.roi_consensus:
                self.comparison_pairs = []

            # Comparison within a single StructureSet
            elif len(structure_sets) == 1:
                    self.comparison_pairs = \
                            structure_sets_filtered[0].get_comparison_pairs()

            # Compare two StructureSets
            else:
                self.comparison_pairs = \
                        structure_sets_filtered[0].get_comparison_pairs(
                            structure_sets_filtered[1])

            self.compared_rois = []
            for pair in self.comparison_pairs:
                self.compared_rois.extend(pair)

        # Make list of all ROIs and assign unique names
        self.rois = [] 
        name_counts = {name: 0 for name in all_names}
        for ss in structure_sets_filtered:
            for roi in ss.get_rois(ignore_empty=True):

                # Ignore the ROI if not comparing
                if self.compare_rois and self.show_compared_rois_only \
                   and not self.roi_consensus:
                    if roi not in self.compared_rois:
                        continue

                # Check whether this is the user-specified initial ROI
                is_current = (self.init_roi == roi.name) and not self.current_roi

                # Ensure ROI name is unique
                if all_names.count(roi.name) > 1:

                    # Use structure set's name if unique and not None
                    if ss.name is not None and ss_names.count(ss.name) == 1:
                        roi.name = f"{roi.name} ({ss.name})"

                    # Otherwise, use count
                    else:
                        name_counts[roi.name] += 1
                        roi.name = f"{roi.name} {name_counts[roi.name]}"

                self.rois.append(roi)
                if is_current:
                    self.current_roi = roi.name
                    self.init_roi = roi.name

        self.structure_set = StructureSet(self.rois)
        self.roi_names = [roi.name for roi in self.rois]
        self.has_rois = bool(len(self.rois))

    def load_grid(self, grid):
        """Load grid."""

        # Can't plot both grid and dose
        if self.has_dose and grid is not None:
            print("Warning: can't overlay both dose map and grid "
                  "on same image. Overlaying dose map only.")
            self.has_grid = False
            return

        if grid is None:
            self.grid = None
        elif isinstance(grid, Grid):
            self.grid = grid
        else:
            self.grid = Grid(grid)

        self.has_grid = self.grid is not None

    def load_jacobian(self, jacobian):
        """Load jacobian determinant."""

        # Can't plot both jacobian and dose
        if self.has_dose and jacobian is not None:
            print("Warning: can't overlay both dose map and Jacobian "
                  "determinant on same image. Overlaying dose map only.")
            self.has_jacobian = False
            return

        if jacobian is None:
            self.jacobian = None
        elif isinstance(jacobian, Jacobian):
            self.jacobian = jacobian
        else:
            self.jacobian = Jacobian(jacobian)

        self.has_jacobian = self.jacobian is not None

    def set_slice(self, view, sl):
        """Set the current slice number in a specific orientation."""

        if sl is None:
            return
        max_slice = self.image.n_voxels[_slice_axes[view]]
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
        sl = self.image.pos_to_slice(pos, ax)
        self.set_slice(view, sl)

    def set_slice_from_idx(self, view, idx):
        """Set the current slice number from index of slice in image array."""

        ax = _slice_axes[view]
        sl = self.image.idx_to_slice(idx, ax)
        self.set_slice(view, sl)

    def make_ui(self, other_viewer=None, share_slider=True, no_roi=False,
                no_intensity=False):
        '''Make Jupyter notebook UI. If other_viewer contains another SingleViewer
        instance, the UI will be taken from that image. If share_slider is
        False, independent intensity and slice sliders will be created.'''

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
            #  if not self.image.dim2:
            self.main_ui.append(self.ui_view)
        else:
            self.ui_view = other_viewer.ui_view
            self.view = self.ui_view.value

        # Empty list for zoom centre coords
        self.current_centre = {view: None for view in _slice_axes}

        # ROI jumping menu
        # Get list of ROIs
        self.rois_for_jump = {
            '': None,
            **{roi.name: roi for roi in self.rois}
        }
        self.ui_roi_jump = ipyw.Dropdown(
            options=self.rois_for_jump.keys(),
            value=self.current_roi,
            description='Jump to',
            style=_style,
        )
        if self.has_rois and not no_roi:
            self.main_ui.append(self.ui_roi_jump)

        # intensity and slice sliders
        if not share_slider or not shared_ui:

            # Make intensity slider
            intensity_limits = self.intensity_limits

            # Single range slider
            if not self.intensity_from_width:
                vmin = max([self.intensity[0], intensity_limits[0]])
                vmax = min([self.intensity[1], intensity_limits[1]])
                ui_intensity_kwargs = {
                    'min': intensity_limits[0],
                    'max': intensity_limits[1],
                    'value': (vmin, vmax),
                    'description': "Intensity range",
                    'continuous_update': False,
                    'style': _style,
                    'step': self.intensity_step,
                }
                slider_kind = (
                    ipyw.FloatRangeSlider if self.intensity_step < 1 else ipyw.IntRangeSlider
                )
                self.ui_intensity = slider_kind(**ui_intensity_kwargs)
                if not no_intensity:
                    self.main_ui.append(self.ui_intensity)

            # Centre and window sliders
            else:
                self.ui_intensity_centre = ipyw.IntSlider(
                    min=intensity_limits[0],
                    max=intensity_limits[1],
                    value=self.intensity,
                    description='Intensity centre',
                    continuous_update=False,
                    style=_style,
                )
                self.ui_intensity_width = ipyw.IntSlider(
                    min=0,
                    max=abs(intensity_limits[1] - intensity_limits[0]),
                    value=self.intensity_width,
                    description='Intensity width',
                    continuous_update=False,
                    style=_style,
                )
                self.ui_intensity_list = [self.ui_intensity_centre, self.ui_intensity_width]
                if not no_intensity:
                    self.main_ui.extend(self.ui_intensity_list)
                self.ui_intensity = ipyw.VBox(self.ui_intensity_list)

            # Get initial zoom centres
            zoom_centre = self.zoom_centre if is_list(self.zoom_centre) \
                    else [self.zoom_centre] * 3
            im_centre = list(self.image.get_centre())
            self.default_centre = {
                view: [im_centre[ax[0]], im_centre[ax[1]]]
                for view, ax in _plot_axes.items()
            }
            self.current_centre = self.default_centre
            for view in _plot_axes:
                for i, ax in enumerate(_plot_axes[view]):
                    if zoom_centre[ax] is not None:
                        self.current_centre[view][i] = zoom_centre[ax]

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
            #  if not self.image.dim2:
            self.main_ui.append(self.ui_slice)

        else:
            if self.intensity_from_width:
                self.ui_intensity_width = other_viewer.ui_intensity_width
                self.ui_intensity_centre = other_viewer.ui_intensity_centre
            else:
                self.ui_intensity = other_viewer.ui_intensity
            self.ui_slice = other_viewer.ui_slice
            self.slice[self.view] = self.ui_slice.value
            if self.zoom_ui:
                self.ui_zoom = other_viewer.ui_zoom
                self.ui_zoom_centre_x = other_viewer.ui_zoom_centre_x
                self.ui_zoom_centre_y = other_viewer.ui_zoom_centre_y
                self.ui_zoom_reset = other_viewer.ui_zoom_reset
                self.current_centre = other_viewer.current_centre
            self.own_ui_slice = False

        # Extra sliders
        self.extra_ui = []
        if not shared_ui:

            # Mask checkbox
            self.ui_mask = ipyw.Checkbox(
                    value=self.masked, indent=False, description='Apply mask')
            self.ui_mask_invert = ipyw.Checkbox(
                    value=self.invert_mask, indent=False,
                    description='Invert mask')
            if self.has_mask:
                self.extra_ui.append(self.ui_mask)
                self.extra_ui.append(self.ui_mask_invert)

            #  Dose opacity
            self.ui_dose = ipyw.FloatSlider(
                value=self.init_dose_opacity,
                min=0,
                max=1,
                step=0.05,
                description='Dose opacity',
                continuous_update=self.continuous_update,
                readout_format='.2f',
                style=_style,
            )
            if self.has_dose:
                self.extra_ui.append(self.ui_dose)

            #  Grid opacity and range
            self.ui_grid_opacity = ipyw.FloatSlider(
                value=self.init_grid_opacity,
                min=0,
                max=1,
                step=0.05,
                description='Grid opacity',
                continuous_update=self.continuous_update,
                readout_format='.2f',
                style=_style,
            )
            if self.has_grid:
                self.extra_ui.extend([self.ui_grid_opacity])

            #  Jacobian opacity and range
            self.ui_jac_opacity = ipyw.FloatSlider(
                value=self.init_jacobian_opacity,
                min=0,
                max=1,
                step=0.05,
                description='Jacobian opacity',
                continuous_update=self.continuous_update,
                readout_format='.2f',
                style=_style,
            )
            self.ui_jac_range = ipyw.FloatRangeSlider(
                min=-1.0,
                max=2.0,
                step=0.1,
                value=self.init_jacobian_range,
                description='Jacobian range',
                continuous_update=False,
                style=_style,
                readout_format='.1f',
            )
            if self.has_jacobian:
                cmap_name = getattr(self.jacobian_kwargs["cmap"], "name", None)
                if cmap_name != "jacobian":
                    self.extra_ui.extend(
                            [self.ui_jac_opacity, self.ui_jac_range])

            # Plot type, spacing, and opacity for deformation field.
            self.ui_df = ipyw.Dropdown(
                options=['quiver', 'grid', 'x-displacement',
                    'y-displacement', 'z-displacement', '3d-displacement',
                    'none'],
                value=self.df_plot_type,
                description='Deformation field',
                style=_style,
                )

            self.ui_df_spacing = ipyw.FloatSlider(
                value=self.df_spacing,
                min=5,
                max=30,
                step=1,
                description='Deformation-field spacing',
                continuous_update=self.continuous_update,
                readout_format='.0f',
                style=_style,
            )

            self.ui_df_opacity = ipyw.FloatSlider(
                value=self.init_df_opacity,
                min=0,
                max=1,
                step=0.05,
                description='Deformation-field opacity',
                continuous_update=self.continuous_update,
                readout_format='.2f',
                style=_style,
            )

            # ROI UI
            # ROI plot type
            self.ui_roi_plot_type = ipyw.Dropdown(
                options=[
                    'contour',
                    'mask',
                    'filled',
                    'centroid',
                    'filled centroid',
                    'none',
                ],
                value=self.roi_plot_type,
                description='ROI plotting',
                style=_style,
            )
            self.ui_roi_consensus_switch = ipyw.Checkbox(
                description="Plot consensus", value=True)
            consensus_opts = ['majority', 'sum', 'overlap']
            try:
                import SimpleITK
                consensus_opts.append('staple')
            except ModuleNotFoundError:
                pass
            self.ui_roi_consensus_type = ipyw.Dropdown(
                options=consensus_opts,
                description='Consensus type',
                style=_style,
            )
            self.roi_consensus_type = self.ui_roi_consensus_type.value
            if self.init_roi is not None:
                self.roi_to_exclude = self.init_roi
            elif self.rois is not None and len(self.rois):
                self.roi_to_exclude = self.rois[0].name
            else:
                self.roi_to_exclude = None

            # Opacity/linewidth sliders
            self.ui_roi_linewidth = ipyw.IntSlider(
                min=1,
                max=8,
                step=1,
                value=self.roi_linewidth,
                description='Linewidth',
                continuous_update=False,
                style=_style,
            )
            self.ui_roi_opacity = ipyw.FloatSlider(
                min=0,
                max=1,
                step=0.1,
                continuous_update=False,
                description='Opacity',
                style=_style,
            )
            self.update_roi_sliders()

            # Toggle buttons for showing/hiding all ROIs.
            self.ui_roi_select_all = ipyw.ToggleButton(
                    value=False, description='Show all ROIs')
            self.ui_roi_deselect_all = ipyw.ToggleButton(
                    value=False, description='Hide all ROIs')

            # Add all ROI UIs
            if self.has_rois:
                to_add = [
                    self.ui_roi_plot_type,
                    self.ui_roi_linewidth,
                    self.ui_roi_opacity,
                    self.ui_roi_select_all,
                    self.ui_roi_deselect_all,
                ]
                if self.roi_consensus:
                    to_add.append(self.ui_roi_consensus_switch)
                    to_add.append(self.ui_roi_consensus_type)
                self.extra_ui.extend(to_add)

        else:
            to_share = [
                'ui_mask',
                'ui_mask_invert',
                'ui_dose',
                'ui_grid_opacity',
                'ui_jac_opacity',
                'ui_jac_range',
                'ui_df',
                'ui_df_spacing',
                'ui_df_opacity',
                'ui_roi_plot_type',
                'ui_roi_consensus_switch',
                'ui_roi_consensus_type',
                'ui_roi_linewidth',
                'ui_roi_select_all',
                'ui_roi_deselect_all',
                'ui_roi_opacity',
                'roi_to_exclude',
            ]
            for ts in to_share:
                setattr(self, ts, getattr(other_viewer, ts))

        # Make lower
        self.make_lower_ui(no_roi=no_roi)

        # Combine UI elements
        self.upper_ui = [ipyw.VBox(self.main_ui), ipyw.VBox(self.extra_ui)]
        self.upper_ui_box = ipyw.HBox(self.upper_ui)
        self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)

        # Create list of widgets that trigger self.plot()
        self.all_ui = (
            self.main_ui + self.extra_ui + self.ui_roi_checkboxes 
            + [self.trigger]
        )

    def make_lower_ui(self, no_roi=False):

        # Saving UI
        self.lower_ui = []
        self.save_name = ipyw.Text(description='Save plot as:', 
                                   value=self.save_as,
                                   style=_style)
        self.save_button = ipyw.Button(
            description='Save',
            tooltip=("Save figure to a file. Filetype automatically "
                     "determined from filename.")
        )
        self.save_button.on_click(self.save_fig)

        # ROI checkboxes and info table
        # Blank checkbox to align with table header(s)
        blank = ipyw.HTML(value='&nbsp;')
        self.ui_roi_checkboxes = (
            [blank] if not self.roi_info else [blank, blank]
        )

        # Make visibility checkbox for each ROI
        self.roi_checkboxes = {
            s: ipyw.Checkbox(value=True, indent=False,
                layout=ipyw.Layout(height='100%'))
            for s in self.roi_names
        }
        self.ui_roi_checkboxes.extend(list(self.roi_checkboxes.values()))
        for s in self.rois:
            s.checkbox = self.roi_checkboxes[s.name]

        # Get list of currently visible ROIs
        self.visible_rois = self.get_visible_rois()

        # Make widget for ROI table and checkboxes
        self.ui_roi_table = ipyw.HTML()
        ui_roi_lower = [self.ui_roi_table]
        if not no_roi or self.roi_info:
            ui_roi_lower.append(
                ipyw.VBox(
                    self.ui_roi_checkboxes,
                    # layout=ipyw.Layout(width='30px', grid_gap='1.5px')
                ),
            )

        self.ui_roi_lower = ipyw.HBox(ui_roi_lower)

        # Add UI for saving ROI info table to a file
        if self.roi_info:
            self.roi_info_save_name = ipyw.Text(
                description="Save table as:", value="", style=_style)
            self.roi_info_save_button = ipyw.Button(
                description="Save",
                tooltip=("Save ROI geometry table to a file. Filetype will "
                         'be CSV unless filename ends in ".tex"')
            )
            self.roi_info_save_button.on_click(self.save_roi_info_table)
            self.ui_roi_info_save = ipyw.HBox([
                self.roi_info_save_name,
                self.roi_info_save_button])
            self.ui_roi_lower = ipyw.VBox([self.ui_roi_lower,
                                           self.ui_roi_info_save])

        # Add ROI comparison table
        if self.compare_rois:
            self.ui_roi_comp_table = ipyw.HTML()

            # Save button
            self.roi_comp_save_name = ipyw.Text(
                description="Save table as:", value="", style=_style)
            self.roi_comp_save_button = ipyw.Button(
                description="Save",
                tooltip=("Save ROI comparison table to a file. Filetype will "
                         'be CSV unless filename ends in ".tex"')
            )
            self.roi_comp_save_button.on_click(self.save_roi_comparison_table)
            self.ui_roi_comp_save = ipyw.HBox([
                self.roi_comp_save_name,
                self.roi_comp_save_button])

            self.ui_roi_lower = ipyw.VBox([self.ui_roi_comp_table,
                                           self.ui_roi_comp_save,
                                           self.ui_roi_lower])

        # Add to lower UI
        if not no_roi or self.roi_info or self.compare_rois:
            self.lower_ui.append(self.ui_roi_lower)

        # Fill tables
        if not no_roi or self.roi_info:
            self.update_roi_info_table()
        if self.compare_rois:
            self.update_roi_comparison()
        
        self.ui_save_plot = ipyw.HBox([self.save_name, self.save_button])
        if self.standalone:
            self.lower_ui.append(self.ui_save_plot)

    def roi_is_visible(self, roi):
        """Check whether a given ROI is currently visible. If the ROI isn't 
        in the checkbox list, return True (used to ensure that consensus 
        contours always count as visible for comparison table)."""

        if roi.name in self.roi_checkboxes:
            return self.roi_checkboxes[roi.name].value
        return True

    def get_visible_rois(self):
        '''Get list of names of currently visible ROIs from checkboxes.'''

        return [roi.name for roi in self.rois if self.roi_is_visible(roi)]

    def update_roi_info_table(self):
        '''Update lower ROI info UI to reflect current view/slice/ROI 
        visibility.'''

        # Make list of coloured ROI names if not showing geometric info
        if not self.roi_info:
            rows = []
            for roi in self.rois:
                grey = not self.roi_is_visible(roi)
                color = roi.get_color_from_kwargs(self.roi_kwargs)
                rows.append({'ROI': get_colored_roi_string(roi, grey, color)})
            df_roi_info = pd.DataFrame(rows)
            self.ui_roi_table.value = df_to_html(df_roi_info)

        # Otherwise, make ROI geometry table
        else:

            # Get table for all currently visible ROIs
            non_visible = [roi for roi in self.rois 
                           if not self.roi_is_visible(roi)]
            self.ui_roi_table.value = self.structure_set.get_geometry( 
                metrics=self.roi_metrics,
                view=self.view,
                sl=self.slice[self.view],
                global_vs_slice_header=True,
                units_in_header=True,
                name_as_index=False,
                nice_columns=True,
                decimal_places=self.roi_info_dp,
                vol_units=self.roi_vol_units,
                area_units=self.roi_area_units,
                length_units=self.roi_length_units,
                force=self.force_roi_geometry_calc,
                greyed_out=non_visible,
                colored=True,
                html=True,
                roi_kwargs=self.roi_kwargs,
            )

            # Only force recalculation of global ROI metrics once
            if self.force_roi_geometry_calc:
                self.force_roi_geometry_calc = False

    def update_roi_comparison(self):
        '''Update lower ROI comparison UI to reflect current view/slice/ROI 
        visibility.'''
        
        if not self.compare_rois:
            return

        self.current_pairs = []

        # Update pairs list if using consensus
        if self.roi_consensus:

            rois_for_consensus = [roi for roi in self.rois 
                                  if roi.name != self.roi_to_exclude
                                  and self.roi_is_visible(roi)]
            consensus = StructureSet(rois_for_consensus).get_consensus(
                self.ui_roi_consensus_type.value, color=self.consensus_color)
            excluded = self.structure_set.get_roi(self.roi_to_exclude)
            self.current_pairs = [(excluded, consensus)]

        # Get list of pairs where both ROIs are currently visible
        else:
            for roi1, roi2 in self.comparison_pairs:
                if self.roi_is_visible(roi1) and self.roi_is_visible(roi2):
                    self.current_pairs.append((roi1, roi2))

        # Get table for all currently visible ROIs
        self.ui_roi_comp_table.value = compare_roi_pairs(
            self.current_pairs,
            metrics=self.roi_comp_metrics,
            view=self.view,
            sl=self.slice[self.view],
            global_vs_slice_header=True,
            units_in_header=True,
            name_as_index=False,
            nice_columns=True,
            decimal_places=self.roi_info_dp,
            vol_units=self.roi_vol_units,
            area_units=self.roi_area_units,
            centroid_units=self.roi_length_units,
            force=self.force_roi_comp_calc,
            colored=True,
            html=True,
            roi_kwargs=self.roi_kwargs,
        )

        # Only force recalculation of global ROI metrics once
        if self.force_roi_comp_calc:
            self.force_roi_comp_calc = False

    def show(self, show=True):
        '''Display plot and UI.'''

        if self.in_notebook and not self.no_ui:
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
        if len(self.lower_ui):
            to_display.append(self.lower_ui_box)
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
        #  if self.image.timeseries:
            #  n_date = self.ui_time.value

        # Get ROI settings
        self.visible_rois = self.get_visible_rois()
        rois_to_plot = [roi for roi in self.rois if roi.name in 
                        self.visible_rois]
        self.update_roi_info_table()
        roi_kwargs = self.roi_kwargs
        if self.ui_roi_plot_type.value != self.roi_plot_type:
            self.update_roi_sliders()
        if self.roi_plot_type in [
            'contour',
            'filled',
            'centroid',
            'filled centroid',
        ]:
            self.roi_linewidth = self.ui_roi_linewidth.value
            roi_kwargs['linewidth'] = self.roi_linewidth
        if self.roi_plot_type == 'mask':
            self.roi_mask_opacity = self.ui_roi_opacity.value
            roi_kwargs['opacity'] = self.roi_mask_opacity
        elif self.roi_plot_type in ['filled', 'filled centroid']:
            self.roi_filled_opacity = self.ui_roi_opacity.value
            roi_kwargs['opacity'] = self.roi_filled_opacity

        # Get ROI consensus settings
        if self.roi_consensus and self.ui_roi_consensus_switch.value:
            consensus_type = self.ui_roi_consensus_type.value
        else:
            consensus_type = None
        if self.roi_to_exclude in [roi.name for roi in rois_to_plot]:
            exclude_from_consensus = self.roi_to_exclude
        else:
            exclude_from_consensus = None

        # Update ROI comparison table
        self.update_roi_comparison()

        # Settings for deformation field
        if self.has_df and self.ui_df.value in ["quiver", "grid",
                "x-displacement", "y-displacement", "z-displacement",
                "3d-displacement"]:
            df = self.df
            df_plot_type = self.ui_df.value
            df_spacing = self.ui_df_spacing.value
            df_opacity = self.ui_df_opacity.value
            df_kwargs = self.df_kwargs
        else:
            df = None
            df_plot_type = None
            df_spacing = None
            df_opacity = None
            df_kwargs = None

        # Settings for overlays (grid, dose map or jacobian)
        if self.has_jacobian:
            jacobian = self.jacobian
            jacobian_opacity = self.ui_jac_opacity.value
            jacobian_kwargs = self.jacobian_kwargs
            jacobian_kwargs["vmin"] = self.ui_jac_range.value[0]
            jacobian_kwargs["vmax"] = self.ui_jac_range.value[1]
        else:
            jacobian = None
            jacobian_opacity = None
            jacobian_kwargs = None

        if self.has_grid:
            grid = self.grid
            grid_opacity = self.ui_grid_opacity.value
            grid_kwargs = self.grid_kwargs
        else:
            grid = None
            grid_opacity = None
            grid_kwargs = None

        '''
        # Settings for overlay (grid, dose map or jacobian)
        if self.has_grid:
            overlay = self.grid
            overlay_opacity = self.ui_grid_opacity.value
            overlay_kwargs = self.grid_kwargs
            if self.init_grid_range is not None:
                overlay_kwargs["vmin"] = self.init_grid_range[0]
                overlay_kwargs["vmax"] = self.init_grid_range[1]
        elif self.has_jacobian:
            overlay = self.jacobian
            overlay_opacity = self.ui_jac_opacity.value
            overlay_kwargs = self.jacobian_kwargs
            overlay_kwargs["vmin"] = self.ui_jac_range.value[0]
            overlay_kwargs["vmax"] = self.ui_jac_range.value[1]
        elif self.has_dose:
            overlay = self.dose
            overlay_opacity = self.ui_dose.value
            overlay_kwargs = self.dose_kwargs
        else:
            overlay = None
            overlay_opacity = None
            overlay_kwargs = None
        '''

        # Make plot
        self.plot_image(
            self.image,
            view=self.view,
            sl=self.slice[self.view],
            gs=self.gs,
            mpl_kwargs=mpl_kwargs,
            figsize=self.figsize,
            zoom=self.zoom,
            zoom_centre=self.current_centre[self.view],
            colorbar=colorbar,
            colorbar_label=self.colorbar_label,
            clb_kwargs=self.clb_kwargs,
            clb_label_kwargs=self.clb_label_kwargs,
            no_xlabel=self.no_xlabel,
            no_ylabel=self.no_ylabel,
            no_xticks=self.no_xticks,
            no_yticks=self.no_yticks,
            no_xtick_labels=self.no_xtick_labels,
            no_ytick_labels=self.no_ytick_labels,
            legend_bbox_to_anchor=self.legend_bbox_to_anchor,
            legend_loc=self.legend_loc,
            annotate_slice=self.annotate_slice,
            major_ticks=self.major_ticks,
            minor_ticks=self.minor_ticks,
            ticks_all_sides=self.ticks_all_sides,
            no_axis_labels=self.no_axis_labels,
            show=False,
            xlim=self.custom_ax_lims[self.view][0],
            ylim=self.custom_ax_lims[self.view][1],
            title=self.title,
            rois=StructureSet(rois_to_plot),
            roi_plot_type=self.roi_plot_type,
            roi_kwargs=roi_kwargs,
            dose=self.dose,
            dose_opacity=self.ui_dose.value,
            dose_kwargs=self.dose_kwargs,
            legend=self.legend,
            centre_on_roi=self.init_roi,
            shift=self.shift,
            scale_in_mm=self.scale_in_mm,
            consensus_type=consensus_type,
            exclude_from_consensus=exclude_from_consensus,
            mask=self.mask,
            masked=self.ui_mask.value,
            invert_mask=self.ui_mask_invert.value,
            mask_color=self.mask_color,
            jacobian=jacobian,
            jacobian_opacity=jacobian_opacity,
            jacobian_kwargs=jacobian_kwargs,
            df=df,
            df_plot_type=df_plot_type,
            df_spacing=df_spacing,
            df_opacity=df_opacity,
            df_kwargs=df_kwargs,
            grid=grid,
            grid_opacity=grid_opacity,
            grid_kwargs=grid_kwargs,
            **kwargs
        )
        self.plotting = False
        self.colorbar_drawn = True

        # Ensure callbacks are set if outside jupyter
        if not self.in_notebook:
            self.set_callbacks()

        # Update figure
        if not self.in_notebook:
            self.image.fig.canvas.draw_idle()
            self.image.fig.canvas.flush_events()

    def plot_image(self, im, view, **kwargs):
        '''Plot an Image, reusing existing axes if outside a Jupyter
        notebook.'''

        # Get axes
        ax = None
        if not self.in_notebook and hasattr(im, 'ax'):
            ax = getattr(im, 'ax')
            ax.clear()

        # If image is linked to another image,
        # add overlay flag to keyword arguments.
        if hasattr(im, 'image'):
            kwargs['include_image'] = self.include_image

        # Plot image
        im.plot(ax=ax, view=view, **kwargs)

        # Check y axis points in correct direction
        if hasattr(im, "plot_extent"):
            if not (im.plot_extent[view][3] > im.plot_extent[view][2]) == (
                im.ax.get_ylim()[1] > im.ax.get_ylim()[0]
            ):
                im.ax.invert_yaxis()

    def save_fig(self, _=None):
        '''Save figure to a file.'''

        outname = self.save_name.value
        if outname is None:
            print("Please provide a filename")
            return
        self.image.fig.savefig(outname)

    def save_roi_info_table(self, _=None):
        '''Save ROI geometric info table to a file.'''

        outname = self.roi_info_save_name.value
        if outname is None:
            print("Please provide a filename")
            return

        # Get pandas DataFrame, ignoring invisible ROIs
        visible_rois = [roi for roi in self.rois if self.roi_is_visible(roi)]
        ss = StructureSet(visible_rois)
        df = ss.get_geometry( 
            metrics=self.roi_metrics,
            view=self.view,
            sl=self.slice[self.view],
            global_vs_slice_header=True,
            units_in_header=True,
            name_as_index=False,
            nice_columns=True,
            decimal_places=self.roi_info_dp,
            vol_units=self.roi_vol_units,
            area_units=self.roi_area_units,
            length_units=self.roi_length_units,
            force=self.force_roi_geometry_calc,
            colored=False
        )

        # Write to latex or CSV
        if outname.endswith(".tex"):
            df.fillna('--', inplace=True)
            with open(outname, 'w') as file:
                tex = df.to_latex(index=False, multicolumn_format='c')
                file.write(tex)
        else:
            if '.' not in outname:
                outname += '.csv'
            df.to_csv(outname, index=False)

    def save_roi_comparison_table(self, _=None):
        '''Save ROI comparison table to a file.'''

        outname = self.roi_comp_save_name.value
        if outname is None:
            print("Please provide a filename")
            return

        # Get pandas DataFrame
        df = compare_roi_pairs(
            self.current_pairs,
            metrics=self.roi_comp_metrics,
            view=self.view,
            sl=self.slice[self.view],
            global_vs_slice_header=True,
            units_in_header=True,
            name_as_index=False,
            nice_columns=True,
            decimal_places=self.roi_info_dp,
            vol_units=self.roi_vol_units,
            area_units=self.roi_area_units,
            centroid_units=self.roi_length_units,
            force=self.force_roi_comp_calc,
            colored=False,
        )

        # Write to latex or CSV
        if outname.endswith(".tex"):
            df.fillna('--', inplace=True)
            with open(outname, 'w') as file:
                tex = df.to_latex(index=False, multicolumn_format='c')
                file.write(tex)
        else:
            if '.' not in outname:
                outname += '.csv'
            df.to_csv(outname, index=False)

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
        self.jump_to_roi()
        self.slice[self.view] = self.slider_to_slice(self.ui_slice.value)
        if not self.scale_in_mm:
            self.update_slice_slider_desc()

        # Get intensity range
        self.v_min_max = self.get_intensity_range()

        # Get zoom settings
        if self.zoom_ui:
            self.zoom = self.ui_zoom.value
            for i, ui in enumerate([self.ui_zoom_centre_x, self.ui_zoom_centre_y]):
                self.current_centre[self.view][i] = ui.value

    def jump_to_roi(self):
        '''Jump to the mid slice of an ROI.'''

        if self.ui_roi_jump.value == '':
            return

        self.current_roi = self.ui_roi_jump.value
        roi = self.rois_for_jump[self.current_roi]
        if not roi.is_empty():
            if not roi.on_slice(self.view, sl=self.slice[self.view]):
                z_ax = _slice_axes[self.view]
                mid = roi.idx_to_slice(roi.get_mid_idx(self.view), ax=z_ax)
                self.ui_slice.value = self.slice_to_slider(mid, ax=z_ax)
                self.slice[self.view] = mid
            self.centre_on_roi(roi)
        self.ui_roi_jump.value = ''
        self.roi_to_exclude = self.current_roi

    def centre_on_roi(self, roi):
        '''Set the current zoom centre to be the centre of an ROI.'''

        if not self.zoom_ui or self.ui_zoom.value == 1:
            return

        sl = self.slice[self.view]
        centre = roi.get_centre(self.view, single_slice=True, sl=sl)

        if None in centre:
            return

        self.current_centre[self.view] = centre
        self.update_zoom_sliders()

    def get_intensity_range(self):
        '''Get vmin and vmax from intensity sliders.'''

        if self.intensity_from_width:
            w = self.ui_intensity_width.value / 2
            centre = self.ui_intensity_centre.value
            return {'vmin': centre - w, 'vmax': centre + w}
        else:
            return {'vmin': self.ui_intensity.value[0], 
                    'vmax': self.ui_intensity.value[1]}

    def update_roi_sliders(self):
        '''Update ROI sliders depending on current plot type.'''

        self.roi_plot_type = self.ui_roi_plot_type.value

        # Disable irrelevant sliders
        self.ui_roi_opacity.disabled = self.roi_plot_type in [
            'contour',
            'none',
            'centroid',
        ]
        self.ui_roi_linewidth.disabled = self.roi_plot_type in ['mask', 'none']

        # Set opacity of masked or filled rois
        if self.roi_plot_type == 'mask':
            self.ui_roi_opacity.value = self.roi_mask_opacity
        elif self.roi_plot_type in ['filled', 'filled centroid']:
            self.ui_roi_opacity.value = self.roi_filled_opacity

    def set_callbacks(self):
        '''Set up matplotlib callback functions for interactive plotting.'''

        if not self.standalone or self.callbacks_set:
            return

        self.image.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.image.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.callbacks_set = True

    def on_key(self, event):
        '''Events run on keypress outside jupyter notebook.'''

        # Settings
        n_small = 1
        n_big = 5

        # Press v to change view
        if event.key == 'v':
            next_view = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'x-y'}
            self.ui_view.value = next_view[self.ui_view.value]

        # Press d to change dose opacity
        elif event.key == 'd':
            if self.has_dose:
                doses = [0, 0.15, 0.35, 0.5, 1]
                next_dose = {
                    doses[i]: doses[i + 1] if i + 1 < len(doses) else doses[0]
                    for i in range(len(doses))
                }
                diffs = [abs(d - self.ui_dose.value) for d in doses]
                current = doses[diffs.index(min(diffs))]
                self.ui_dose.value = next_dose[current]

        # Press m to switch mask on and off
        elif event.key == 'm':
            if self.has_mask:
                self.ui_mask.value = not self.ui_mask.value

        # Press c to change ROI plot type
        elif event.key == 'c':
            if self.has_rois:
                next_type = {
                    'mask': 'contour',
                    'contour': 'filled',
                    'filled': 'mask',
                }
                self.ui_roi_plot_type.value = next_type[
                    self.ui_roi_plot_type.value
                ]

        # Press j to jump between ROIs
        elif event.key == 'j' and self.has_rois:
            if self.current_roi == '':
                current_idx = 0
            else:
                current_idx = self.roi_names.index(self.current_roi)
            new_idx = current_idx + 1
            if new_idx == len(self.roi_names):
                new_idx = 0
            new_roi = self.roi_names[new_idx]
            self.ui_roi_jump.value = new_roi

        # Press arrow keys to scroll through many slices
        elif event.key == 'left':
            self.decrease_slice(n_small)
        elif event.key == 'right':
            self.increase_slice(n_small)
        elif event.key == 'down':
            self.decrease_slice(n_big)
        elif event.key == 'up':
            self.increase_slice(n_big)

        else:
            return

        # Remake plot
        if self.standalone:
            self.plot()

    def on_scroll(self, event):
        '''Events run on scroll outside jupyter notebook.'''

        if event.button == 'up':
            self.increase_slice()
        elif event.button == 'down':
            self.decrease_slice()
        else:
            return

        # Remake plot
        if self.standalone:
            self.plot()

    def increase_slice(self, n=1):
        '''Increase slice slider value by n slices.'''

        new_val = self.ui_slice.value + n * self.ui_slice.step
        if new_val <= self.ui_slice.max:
            self.ui_slice.value = new_val
        else:
            self.ui_slice.value = self.ui_slice.max

    def decrease_slice(self, n=1):
        '''Decrease slice slider value by n slices.'''

        new_val = self.ui_slice.value - n * self.ui_slice.step
        if new_val >= self.ui_slice.min:
            self.ui_slice.value = new_val
        else:
            self.ui_slice.value = self.ui_slice.min

    def on_view_change(self):
        '''Deal with a view change.'''

        self.update_slice_slider()
        if self.zoom_ui:
            self.update_zoom_sliders()

    def update_slice_slider(self):
        '''Update the slice slider to show the axis corresponding to the
        current view, with value set to the last value used on that axis.'''

        if not self.own_ui_slice:
            return

        # Get new min and max
        ax = _slice_axes[self.view]
        if self.scale_in_mm:
            new_min = min(self.image.lims[ax])
            new_max = max(self.image.lims[ax])
        else:
            new_min = 1
            new_max = self.image.n_voxels[ax]

        # Set slider values
        val = self.slice_to_slider(self.slice[self.view])
        self.update_slider(self.ui_slice, new_min, new_max, val)

        # Set step and description
        self.ui_slice.step = abs(self.image.voxel_size[ax]) if self.scale_in_mm else 1
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
            pos = self.image.slice_to_pos(self.slider_to_slice(self.ui_slice.value), ax)
            self.ui_slice.description = f'{_axes[ax]} ({pos:.1f} mm)'

    def update_zoom_sliders(self):
        '''Update zoom sliders to reflect the current view.'''

        if not self.zoom_ui:
            return

        units = ' (mm)' if self.scale_in_mm else ''
        for i, ui in enumerate([self.ui_zoom_centre_x, self.ui_zoom_centre_y]):

            # Set min, max, and value
            ax = _plot_axes[self.view][i]
            new_min = min(self.image.lims[ax])
            new_max = max(self.image.lims[ax])
            self.update_slider(ui, new_min, new_max, 
                               self.current_centre[self.view][i])

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
            return self.image.slice_to_pos(sl, ax)
        else:
            return sl

    def slider_to_slice(self, val, ax=None):
        '''Convert a slider value to a slice number.'''

        if ax is None:
            ax = _slice_axes[self.view]

        if self.scale_in_mm:
            return self.image.pos_to_slice(val, ax)
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
