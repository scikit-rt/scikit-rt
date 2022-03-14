'''Package providing dictionaries for standardising ROI labels.'''

# Import dictionaries for head-and-neck cohort.
from voxtox.roi_names.head_and_neck_roi_names import head_and_neck_plan,\
            head_and_neck_voxtox, head_and_neck_iov, head_and_neck_mvct,\
            head_and_neck_parotid_fiducials, head_and_neck_tre

# Import dictionaries for prostate cohort.
from voxtox.roi_names.prostate_roi_names import prostate_plan,\
            prostate_voxtox, prostate_iov, prostate_mvct

def get_roi(structure_set, roi_name, data_type='plan'):
    '''
    Obtain named ROI from structure set.

    A standardised name is passed as argument, and the possible actual
    names are determined based on data type.

    **Parameters:**

    structure_set : skrt.structures.StructureSet
        Structure set from which ROI is to be retrieved.

    roi_name : str
        Name identifying ROI to be retrieved.

    data_type : str, default='plan'
        Type of data considered.  This affects the possible actual ROI
        names corresponding to a standardised name.  The available
        data types include: 'plan', 'voxtox', 'iov', 'mvct'.
    '''

    from importlib import import_module
    roi = None
    for site in ['head_and_neck', 'prostate']:
        if roi is None:
            # Obtain dictionary of ROI names relative to site and data type.
            package = import_module(f'voxtox.roi_names.{site}_roi_names')
            roi_names = getattr(package, f'{site}_{data_type}', [])
            # Search for ROI.
            if roi_name in roi_names:
                ss = structure_set.filtered_copy(
                        names={roi_name: roi_names[roi_name]})
                if roi_name in ss.get_roi_names():
                    roi = ss.get_roi_dict().get(roi_name)
    return roi
