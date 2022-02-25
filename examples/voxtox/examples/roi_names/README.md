##voxtox/examples/roi\_names##

This directory contains functions, and example data, for creating
modules defining dictionaries that allow standardisation of
ROI names in scikit-rt.

The workflow is as follows:

1. `get_roi_names.py` is run over a collection of patient datasets,
to extract lists of the ROI names used in structure sets.  Example
outputs can be found in:

- `data_ct.py`
- `data_djn.py`
- `data_jes.py`
- `data_mvct.py`

Files may need to be manually edited to group lists of ROI names
into collections.  (This has been done for `data_mvct.py`.)

2. A module similar to `sort_head_and_neck.py` or `sort_prostate.py`,
which rely on functions defined in `sort_rois.py` should created.
This module should read in the saved output from `get_roi_names.py`,
and should define how actual ROI names are to be matched to
standardised names.  It is then run to produce the standardisation
dictionaries.
