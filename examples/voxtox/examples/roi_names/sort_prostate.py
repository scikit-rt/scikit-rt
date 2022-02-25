'''Module for standardising ROI names in VoxTox prostate data.'''

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

    # Function and data relative to inter-observer study.
    def prostate_iov_assign(roi=''):

        if 'rectum' == roi.lower():
            name = 'rectum_jes'
        else:
            name = None

        return name

    prostate_iov_names = {}
    prostate_iov_names['rectum_am'] = ['rectum a*m*']
    prostate_iov_names['rectum_cw'] = ['rectum c*w*']
    prostate_iov_names['rectum_gh'] = ['rectum g*h*']
    prostate_iov_names['rectum_gb'] = ['rectum g*b*']
    prostate_iov_names['rectum_jes'] = []
    prostate_iov_names['rectum_lhd'] = ['rectum l*h*d*']
    prostate_iov_names['rectum_nb'] = ['rectum n*b*']
    prostate_iov_names['rectum_rb'] = ['rectum r*b*']
    prostate_iov_names['rectum_sr'] = ['rectum s*r*']
    prostate_iov_names['rectum_yr'] = ['rectum y*r*']

    # Data relative to treatment scans.
    prostate_mvct_names = {}
    prostate_mvct_names['femoral_heads'] = ['femoral']
    prostate_mvct_names['pelvic_floor_muscles'] = ['pelvic']
    prostate_mvct_names['rectum'] = ['rectum']

    # Function and data relative to clinical outining for planning scans.
    def prostate_plan_assign(roi=''):

        if '-' in roi:
            name = 'composite'
        elif (('pr' in roi.lower() or 'pro' in roi.lower())
            and ('sv' in roi.lower() or 'sem' in roi.lower())):
            name = 'prostate_and_seminal_vesicles'
        elif 'pr' == roi.lower():
            name = 'prostate'
        else:
            name = None

        return name

    prostate_plan_lr_rois = ['femoral_head', 'prosthetic_hip']

    prostate_plan_names = {}
    prostate_plan_names['femoral_head'] = ['fem', 'hip']
    prostate_plan_names['rectum'] = ['rec']
    prostate_plan_names['air'] = ['air']
    prostate_plan_names['bladder'] = ['blad']
    prostate_plan_names['sigmoid_colon'] = ['sigmoid']
    prostate_plan_names['prosthetic_hip'] = [
        'artificial', 'art hip', 'prosth', 'lt pros', 'lpros',
        'rt pros', 'r pros']
    prostate_plan_names['prostate_bed'] = ['bed']
    prostate_plan_names['prostate'] = ['pro']
    prostate_plan_names['prostate_and_seminal_vesicles'] = []
    prostate_plan_names['pelvic_floor_muscles'] = ['floor']
    prostate_plan_names['seminal_vesicles'] = ['semin', 'sv']
    prostate_plan_names['composite'] = []
    prostate_plan_names['another'] = [
        'block', 'direct', 'metal', 'avoid', 'ptv', 'sort',
        'margin', 'anu', 'post', 'plan', 'base', 'ring',
        'shell', 'tpsv', 'apron', 'ln ctv',
        'ext ap', '+bs', 'opt']

    prostate_plan_ordered_names = [
        'air',
        'another',
        'bladder',
        'prosthetic_hip',
        'femoral_head',
        'pelvic_floor_muscles',
        'prostate_bed',
        'prostate',
        'prostate_and_seminal_vesicles',
        'rectum',
        'sigmoid_colon',
        'seminal_vesicles'
        ]

    # Data relative to VoxTox outining for planning scans.
    prostate_voxtox_names = {}
    prostate_voxtox_names['rectum'] = ['rec']
    prostate_voxtox_names['pelvic_floor_muscles'] = ['pelvic']

def run():
    '''Perform processing.'''

    # Associations of modules to lists of ROI names.
    roi_collections = {}
    roi_collections['data_ct'] = ['prostate_plan']
    roi_collections['data_jes'] = ['prostate_voxtox']
    roi_collections['data_mvct'] = ['prostate_mvct']
    roi_collections['data_mvct'] = ['prostate_iov']

    # Create and output standardisation dictionarie.
    print('\'\'\'Dictionaries for standardising ROI'
          + ' labels of prostate cohort.\'\'\'')
    for module_name in sorted(roi_collections):
        roi_module = import_module(module_name)
        for collection in sorted(roi_collections[module_name]):
            rois = get_collection_rois(roi_module, Collector, collection)
            lr_rois = getattr(Collector, f'{collection}_lr_rois', [])
            print_rois(rois, lr_rois, collection, filter=True)

if '__main__'  == __name__:
    run()
