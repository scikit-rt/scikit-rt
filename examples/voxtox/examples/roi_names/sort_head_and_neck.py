'''Module for standardising ROI names in VoxTox head-and-neck data.'''
from importlib import import_module

from sort_rois import get_collection_rois, print_rois


class Collector:
    '''
    Container class for functions and data for collections of ROI names.

    For each collection, the following may be defined, where each
    name listed is prefixed by the collection name:

    names : dict, mandatory
        Dictionary where keys are standarised ROI names
        and values are lists of strings for identifying
        actual ROI names, amont those contained in in_rois.

    lr_rois : list, optional
        List of standardised ROI names for bilateral ROIs.

    ordered_names : list, optional
        List of standardised ROI names, in order in which
        processing is to be performed.  Forcing a particular
        order may be needed if the rules for identifying
        one ROI are a subset of the rules for identifying
        another ROI.

    assign_special : function, optional
        User-defined function that takes an ROI name as argument,
        and applies custom logic to determine whether this
        corresponds to particular standardised names.
    '''

    # Data relative to intra-observer study.
    head_and_neck_iov_names = {}
    head_and_neck_iov_names['parotid_1'] = ['parotid']
    head_and_neck_iov_names['parotid_2'] = ['parotid 2']
    head_and_neck_iov_names['smg_1'] = ['smg']
    head_and_neck_iov_names['smg_2'] = ['smg 2']
    head_and_neck_iov_names['spc_1'] = ['pha']
    head_and_neck_iov_names['spc_2'] = ['spc 2']
    head_and_neck_iov_names['spinal_cord_1'] = ['cord']
    head_and_neck_iov_names['spinal_cord_2'] = ['cord2']

    head_and_neck_iov_lr_rois = ['parotid_1', 'parotid_2', 'smg_1', 'smg_2']

    head_and_neck_iov_ordered_names = [
        'parotid_2', 'parotid_1', 'smg_2', 'smg_1', 'spc_2', 'spc_1',
        'spinal_cord_2', 'spinal_cord_1']

    # Data relative to treatment scans.
    head_and_neck_mvct_names = {}
    head_and_neck_mvct_names['parotid'] = ['parotid']
    head_and_neck_mvct_names['smg'] = ['smg']
    head_and_neck_mvct_names['spc'] = ['pha']
    head_and_neck_mvct_names['spinal_cord'] = ['cord']

    head_and_neck_mvct_lr_rois = ['parotid', 'smg']

    # Data relative to parotid fiducials.
    head_and_neck_parotid_fiducials_names = {}
    head_and_neck_parotid_fiducials_names['anterior'] = ['ant']
    head_and_neck_parotid_fiducials_names['lateral'] = ['lateral']
    head_and_neck_parotid_fiducials_names['mandible'] = ['mandible']
    head_and_neck_parotid_fiducials_names['mastoid'] = ['mastoid']
    head_and_neck_parotid_fiducials_names['medial'] = ['medial']
    head_and_neck_parotid_fiducials_names['parotid'] = ['parotid']
    head_and_neck_parotid_fiducials_names['posterior'] = ['post']

    head_and_neck_parotid_fiducials_lr_rois = [
            'anterior', 'lateral', 'mandible', 'mastoid',
            'medial', 'parotid', 'posterior']

    # Function and data relative to clinical outining for planning scans.
    def head_and_neck_plan_assign(roi=''):

        if '-' in roi:
            name = 'composite'
        else:
            name = None

        return name

    head_and_neck_plan_lr_rois = [
            'parotid', 'smg', 'optic_nerve', 'eye', 'lens', 'globe',
            'cochlea']

    head_and_neck_plan_names = {}
    head_and_neck_plan_names['another'] = [
        'plstr', 'relevant',
        'block', 'avoid', 'inf', 'sup', 'density', 'real', 'body', 'ring',
        '+bs', '+', 'buffer', '/', 'old', '54', 'ptv', 'ctv']
    head_and_neck_plan_names['bolus'] = ['bolus', 'blous', 'bo']
    head_and_neck_plan_names['brainstem'] = ['stem']
    head_and_neck_plan_names['cochlea'] = ['coch', 'coclea']
    head_and_neck_plan_names['composite'] = []
    head_and_neck_plan_names['eye'] = ['eye']
    head_and_neck_plan_names['globe'] = ['globe']
    head_and_neck_plan_names['lens'] = ['lens']
    head_and_neck_plan_names['mandible'] = ['mandible']
    head_and_neck_plan_names['optic_chiasm'] = ['chiasm']
    head_and_neck_plan_names['optic_nerve'] = ['nerve']
    head_and_neck_plan_names['parotid'] = ['parotid']
    head_and_neck_plan_names['pituitary_gland'] = [
        'pituitary', 'pituiatry', 'piuitary', 'Pituatary']
    head_and_neck_plan_names['smg'] = ['subm', 'smg', 'sm gland']
    head_and_neck_plan_names['spc'] = ['const']
    head_and_neck_plan_names['spinal_cord'] = ['cord', 'canal']

    # Data relative to target registration error.
    head_and_neck_tre_names = {}
    head_and_neck_tre_names['c2_c3']= ['C2/3']
    head_and_neck_tre_names['parotid_anterior']= ['parotid ant']
    head_and_neck_tre_names['parotid_mastoid']= ['parotid mastoid']
    head_and_neck_tre_names['pterygoid_plate']= ['pterygoid plate']

    head_and_neck_tre_lr_rois = [
            'parotid_anterior', 'parotid_mastoid', 'pterygoid_plate']

    # Function and data relative to VoxTox outining for planning scans.
    def head_and_neck_voxtox_assign(roi=''):

        if 'prv' in roi.lower():
            name = 'another'
        else:
            name = None

        return name

    head_and_neck_voxtox_lr_rois = ['parotid', 'smg']

    head_and_neck_voxtox_names = {}
    head_and_neck_voxtox_names['alterio_pc'] = ['alertio', 'alterio']
    head_and_neck_voxtox_names['mpc'] = ['mpc']
    head_and_neck_voxtox_names['oral_cavity'] = ['cavity']
    head_and_neck_voxtox_names['parotid'] = ['parotid']
    head_and_neck_voxtox_names['sg_larynx'] = ['larynx', 'Groningen']
    head_and_neck_voxtox_names['smg'] = ['submand', 'smg']
    head_and_neck_voxtox_names['spc'] = ['spc']
    head_and_neck_voxtox_names['spinal_cord'] = ['cord']

def run():
    '''Perform processing.'''

    # Associations of modules to lists of ROI names.
    roi_collections = {}
    roi_collections['data_ct'] = ['head_and_neck_plan']
    roi_collections['data_djn'] = ['head_and_neck_voxtox']
    roi_collections['data_mvct'] = ['head_and_neck_mvct']
    roi_collections['data_mvct'] += ['head_and_neck_iov']
    roi_collections['data_mvct'] += ['head_and_neck_parotid_fiducials']
    roi_collections['data_mvct'] += ['head_and_neck_tre']

    # Create and output standardisation dictionarie.
    print('\'\'\'Dictionaries for standardising ROI'
          + ' labels of head-and-neck cohort.\'\'\'')
    for module_name in sorted(roi_collections):
        roi_module = import_module(module_name)
        for collection in sorted(roi_collections[module_name]):
            rois = get_collection_rois(roi_module, Collector, collection)
            lr_rois = getattr(Collector, f'{collection}_lr_rois', [])
            print_rois(rois, lr_rois, collection, filter=True)

if '__main__'  == __name__:
    run()
