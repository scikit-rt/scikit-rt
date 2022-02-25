'''
Module for creating dictionaries to standarise ROI names.

The same ROI may be labelled differently in different structure sets.
This module provides functions that applies user-defined rules to
construct dictionaries where keys are standardised ROI names,
and values are lists of alternative names that may be used for the ROI:

- get_collection_rois():
  Create dictionary for standardisation of ROI names
  (higher level than get_rois()).
- get_rois():
  Create dictionary for standardisation of ROI names.
- get_side():
    Determine whether an ROI name refers to left or right.
- print_rois():
    Print dictionaries for standardising ROI names.

The idea is that the output from print_rois() should be directed
to a file with extension .py, from which the standardisation
dictrionaries can then be imported.
'''

def get_collection_rois(roi_module, Collector, collection=''):
    '''
    Create dictionary for standardisation of ROI names.

    This function takes higher-level arguments that get_rois().

    **Parameters:**
    
    roi_module : module
        Module from which lists of ROI names are to be imported.

    Collector : class
        Class used as container for functions and data
        that can be passed to get_rois() relative to a given
        collection or ROI names.

    collection : str, default=''
        Name associated with the collection of ROI names to be processed.
    '''
    in_rois = getattr(roi_module, collection, [])
    assign_special = getattr(Collector, f'{collection}_assign', None)
    names = getattr(Collector, f'{collection}_names', {})
    lr_rois = getattr(Collector, f'{collection}_lr_rois', None)
    ordered_names = getattr(
        Collector, f'{collection}_ordered_names', None)

    rois = get_rois(in_rois, names, lr_rois, ordered_names, assign_special)

    return rois

def get_rois(in_rois=[], names={}, lr_rois=None, ordered_names = None,
                  assign_special=None):
    '''
    Create dictionary for standardisation of ROI names.

    **Parameters:**

    in_rois : list, default=[]
        Input list of names that may refer to any ROI.

    names : dict, default={}
        Dictionary where keys are standarised ROI names
        and values are lists of strings for identifying
        actual ROI names, amont those contained in in_rois.

    lr_rois : list, default=[]
        List of standardised ROI names for bilateral ROIs.

    ordered_names : list, default=[]
        List of standardised ROI names, in order in which
        processing is to be performed.  Forcing a particular
        order may be needed if the rules for identifying
        one ROI are a subset of the rules for identifying
        another ROI.

    assign_special : function, default=None
        User-defined function that takes an ROI name as argument,
        and applies custom logic to determine whether this
        corresponds to particular standardised names.
    '''
    
    # Ensure that each ROI name is included only once.
    in_rois = set(in_rois)

    # By default, there are taken to be no bilateral ROIs.
    if lr_rois is None:
        lr_rois = []

    # By default, order of processing is unimportant.
    if ordered_names is None:
        ordered_names = names.keys()

    # By default, assign_special is a function that returns None.
    if assign_special is None:
        assign_special = lambda *args, **kwargs: None

    # Initialise dictionary with key for ROIs not matched
    # to any standardised name.
    out_rois = {'another': []}

    # Loop over ROIs.
    for roi in in_rois:

        # First perform matches based on custom logic.
        name = assign_special(roi=roi)
        if name:
            if not name in out_rois:
                out_rois[name] = []
            out_rois[name].append(roi)
            continue

        # Loop over possible sub-strings until a match is found
        roi_assigned = False
        for name in ordered_names:
            if roi_assigned:
                break
            for str_to_match in names[name]:
                if str_to_match.lower() in roi.lower():
                    if not name in out_rois:
                        out_rois[name] = []
                    out_rois[name].append(roi)
                    roi_assigned = True

        # Store ROI names not matched to any standardised name.
        if not roi_assigned:
            out_rois['another'].append(roi)

    # Divide bilateral ROIs as those on the left, those on the right,
    # and those with laterality unknown.
    sides = ['left', 'right', 'unknown']
    for name in lr_rois:
        chiral_names = {}
        name_elements = name.split('_')
        if name_elements[-1].isnumeric():
            # Deal with case where ROI is numbered,
            # for example as can happen for ROIs
            # outlined for intra-observer studies.
            for side in sides:
                chiral_names[side] = '_'.join(
                        name_elements[:-1] + [side] + name_elements[-1:])
        else:
            for side in sides:
                chiral_names[side] = '_'.join(name_elements + [side])
        for side in sides:
            out_rois[f'{chiral_names[side]}'] = []
        for roi in out_rois[name]:
            out_rois[f'{chiral_names[get_side(roi)]}'].append(roi)

    return out_rois

def get_side(roi=''):
    '''
    Determine whether an ROI name refers to left or right.

    **Parameter:**

    roi - str, default=''
        Name of ROI for which laterality is to be determined.
    '''
    side = 'unknown'

    # Rules for deciding that an ROI is on the left.
    if 'left' in roi.lower():
        side = 'left'
    if roi.lower().startswith('l'):
        side = 'left'
    if roi.lower().startswith('prv l'):
        side = 'left'
    if roi.lower().startswith('prvl'):
        side = 'left'
    if roi.lower().endswith('l'):
        side = 'left'
    if roi.lower().endswith('l1'):
        side = 'left'
    if roi.lower().endswith('(l)'):
        side = 'left'
    if 'lt' in roi.lower():
        side = 'left'
    if '_l_' in roi.lower():
        side = 'left'
    if '_l ' in roi.lower():
        side = 'left'
    if ' l ' in roi.lower():
        side = 'left'

    # Rules for deciding that an ROI is on the right.
    if 'right' in roi.lower():
        side = 'right'
    if roi.lower().startswith('r'):
        side = 'right'
    if roi.lower().startswith('prv r'):
        side = 'right'
    if roi.lower().startswith('prvr'):
        side = 'right'
    if roi.lower().endswith('r') and not roi.lower().endswith('ar'):
        side = 'right'
    if roi.lower().endswith('r1'):
        side = 'right'
    if roi.lower().endswith('(r)'):
        side = 'right'
    if 'rt' in roi.lower():
        side = 'right'
    if '_r_' in roi.lower():
        side = 'right'
    if '_r ' in roi.lower():
        side = 'right'
    if ' r ' in roi.lower():
        side = 'right'

    return side

def print_rois(rois={}, lr_rois=None, collection='rois', filter=True):
    '''
    Print dictionaries for standardising ROI names.

    **Parameters:**

    rois : dict, default={}
        Dictionary where keys are standardised ROI names and values
        are lists of actual names used in structure sets.

    lr_rois : list, default=[]
        List of standardised ROI names for bilateral ROIs.

    collection : str, default=''
        Name associated with the collection of ROI names processed.

    filter : bool, default=True
        If True, exclude from print out information for ROIs not
        matched to any standardised name, and information for
        bilateral ROIs not identified as left or right.  If False,
        include print out for all ROIs.
    '''
    print(f'\n{collection} = {{}}')
    for roi, values in sorted(rois.items()):
        values.sort(key=str.casefold)
        if (roi not in ['composite', 'another'] and
            roi not in lr_rois and 'unknown' not in roi) or not filter:
                print(f'{collection}[\'{roi}\'] = {values}')
