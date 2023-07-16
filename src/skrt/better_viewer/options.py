"""
Module for functions relevant to image display:

    - set_viewer_options():
      Define options for use with skrt.better_viewer.BetterViewer.
"""
import matplotlib.pyplot as plt

from skrt.core import is_list

def set_viewer_options(to_exclude=None, usetex=False, **kwargs):
    """
    Define options for use with skrt.better_viewer.BetterViewer.

    Options for Matplotlib runtime configuration are applied automatically.
    The options dictionary returned by this function may be passed
    to BetterViewer as **kwargs.

    All options may be overwritten before calling BetterViewer.

    **Parameter:**
    to_exclude: list
        List of keys not to include in the returned dictionary.

    usetex: bool, default=False
        Indicate whether to use LaTeX for text rendering.  Setting to True
        when no LaTeX installation is avaiable will generate an error.

    kwargs: dict
        Key-value pairs with which to update BetterViewer options dictionary
        created here.
    """
    # Set Matplotlib runtime configuration.
    # For details of Matplotlib configuration, see:
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    # Sizes are in points.

    # For axes, set spacing (pad) and size of label and title
    plt.rc("axes", labelpad=0, labelsize=25, titlepad=17, titlesize=25)

    # Set default text characteristics.
    # Possible weight values are:
    # 100, 200, 300, 400 / "normal", 500, 600, 700 / "bold", 800, 900.
    plt.rc("font", family="serif", size=20, weight=400)
    if usetex:
        plt.rc("font", serif=["Times"])

    # Set default font size for legends.
    plt.rc("legend", fontsize=16)

    # Set mathematics typeface when using matplotlib's built-in fonts.
    plt.rc("mathtext", fontset="dejavuserif")

    # Use TeX/LaTeX for typesetting.  (This requires working TeX/LaTeX installation.)
    plt.rc("text", usetex=usetex)

    # For ticks, set label size and direction ("in", "out", "inout").
    plt.rc(("xtick", "ytick"), labelsize=25, direction="out")

    # For major and minor ticks, set size and width.
    # For major ticks, set spacing (pad) of label.
    plt.rc(("xtick.major"), pad=3)
    plt.rc(("xtick.major", "ytick.major"), size=9, width=1.0)
    plt.rc(("xtick.minor", "ytick.minor"), size=4.5, width=1.0)
    plt.rc(("ytick.major"), pad=2)

    # Create dictionary of BetterViewer image-display options.
    options = {
        # Set figure size in inches.
        "figsize": (10, 6),
        # Show major ticks at specified interval (axis units).
        # "major_ticks": 25,
        # Show minor ticks for specified number of intervals
        # between major ticks.
        "minor_ticks": 5,
        # Indicate whether axis units should be mm or numbers of voxels.
        "scale_in_mm" : True,
        # Indicate whether ticks should be shown on all sides.
        "ticks_all_sides": True,
        # Colour map for images.  For pre-defined colour maps, see:
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        # "gray": black to white; "binary": white to black.
        "cmap": "gray",
        # Define options for colour bars.  For possibilities, see:
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html#matplotlib.pyplot.colorbar
        "clb_kwargs": {"pad":0.05},
        # Define options for colour-bar labels.  For possibilities, see:
        # https://matplotlib.org/stable/api/colorbar_api.html#matplotlib.colorbar.Colorbar.set_label
        "clb_label_kwargs": {"labelpad": 5},
        }

    # Update dictionary of BetterViewer image-display options,
    # using any values passed as keyword arguments.
    options.update(kwargs)

    # Remove from dictionary any keys that are to be excluded.
    if isinstance(to_exclude, str):
        to_exclude = [to_exclude]
    if is_list(to_exclude):
        for key in to_exclude:
            options.pop(key, None)

    return options
