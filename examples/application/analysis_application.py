'''
Application that runs analysis algorithm.

!!! To run this application, parameter paths (towards bottom of file)
!!! must be set to a list of paths to patient data
'''

import glob
import os

import pandas as pd

from skrt.application import Algorithm, Application


class AnalysisAlgorithm(Algorithm):
    '''Subclass of Algorithm, for analysing patient data'''

    def __init__(self, opts={}, name=None, log_level=None):
        '''
        Create instance of Algorithm class.

        Parameters
        ----------
        opts : dict, default={}
            Dictionary for setting algorithm attributes.
        name : str, default=''
            Name for identifying algorithm instance.
        log_level: str/int/None, default=None
            Severity level for event logging.  If the value is None,
            log_level is set to the value of skrt.core.Defaults().log_level.
        '''

        # If no name provided for instance, use class name
        class_name = str(type(self)).split('.')[-1].split('\'')[0]
        name = class_name if name is None else name

        # Configurable variables
        # Maximum number of patients to be analysed
        self.max_patient = 10000

        # Call to __init__() method of base class
        # sets values for object properties based on dictionary opts,
        # and creates event logger with speficied name and log_level.
        Algorithm.__init__(self, opts, name, log_level)

        # List for storing data records
        self.roi_records = []

        # Counter of number of patients to be analysed
        self.n_patient = 0

    # The execute() method is called once for each patient, the data for
    # which is passed via an instance of the skrt.patient.Patient class.
    def execute(self, patient=None):

        # Increase patient count
        self.n_patient += 1

        # Print patient identifier
        print(f'\n{self.n_patient:03d} {patient.id}')

        # Only consider the first study.
        # In VoxTox, patients usually have only a single study,
        # but have more studies in the few cases where treatment is replanned.
        study = patient.studies[0]

        if len(study.ct_structs) >= 2:
            # Assume that earliest structure set is from clinical planning
            planning_structs = study.ct_structs[0]\
                    .copy(names=self.roi_map, keep_renamed_only=True)
            # Assume that latest structure set is from VoxTox study
            voxtox_structs = study.ct_structs[-1]\
                    .copy(names=self.roi_map, keep_renamed_only=True)
            
            # Calculate dice scores for planning ROIs versus VoxTox ROIs,
            # and add to the list of data records.
            # Note: the ROI class defines methods for calculating
            # a number of comparison metrics.
            for planning_roi in planning_structs:
                voxtox_roi = voxtox_structs[planning_roi.name]
                dice = planning_roi.get_dice(voxtox_roi)
                self.roi_records.append({
                    'id': patient.id,
                    'roi': planning_roi.name,
                    'dice': dice,
                    })
                self.logger.info(f'{planning_roi.name}: dice = {dice:.4f}')
                # If dice score is 1.0, the same ROI may have been picked up
                # got planning and VoxTox - worth checking.
                if dice > 0.999:
                    self.logger.warning(
                            f'Dice score of {dice:.4f} is suspicious!')

        # Set non-zero status code if maximum number of patients reached
        if self.n_patient >= self.max_patient:
            self.status.code = 1
            self.status.reason = f'Reached {self.n_patient} patients\n'
            self.finalise()

        return self.status

    def finalise(self):

        print (f'\nNumber of patients analysed = {self.n_patient}')

        # Create dataframe from data records
        # and save in csv format
        df = pd.DataFrame(self.roi_records)
        df.to_csv('roi_info.csv')

        return self.status

if '__main__' == __name__:

    # Create a dictionary of options to be passed to the algorithm
    opts = {}

    # Set the maximum number of patients to be analysed
    opts['max_patient'] = 3

    # Create dictionary where each key is a name to be assigned
    # to a region of interest (ROI), and the associated value
    # is the list of names that may have been used during contouring.
    opts['roi_map']= {}
    opts['roi_map']['parotid_left'] = [
        'left parotid (dn)', 'left parotid - dn',
        'l parotid', 'left parotid', 'lt parotid', 'parotid lt', 'parotid_l',
        'parotid l', 'parotid_l_', 'parotid_l1', 'parotid left',
        'parotid_l_sp', 'left  parotid', 'l  parotid', 'l parotid_old',
        'l parotid b', 'leftparotid']
    opts['roi_map']['parotid_right'] = [
        'right parotid (dn)', 'right parotid - dn',
        'r parotid', 'right parotid', 'rt parotid', 'parotid rt', 'parotid_r',
        'parotid r', 'parotid_r_', 'parotid_r1', 'parotid right',
        'parotid_r_sp', 'right  parotid', 'r  parotid', 'r parotid_old',
        'r parotid b', 'rightparotid']
    opts['roi_map']['smg_left'] = [
        'l submandibular gland', 'left submandibular', 'lt submandibular',
        'left submandibular gland', 'l submandibular', 'l submandib',
        'left submandibular glan', 'l submand', 'l submand gland',
        'lt submand', 'submandibular lt', 'submandibular gland lt',
        'lt submandibular gland', 'submand left', 'submandibular l',
        'lt submang', 'submandibular left', 'subman lt', 'l sub mandibular',
        'l submandibular galnd', 'l sub mand gland', 'left smg', 'l smg',
        'lt smg', 'l  submandibular', 'left subm gland', 'lt submandib',
        'left sm gland', 'left submandibular aj', 'lt submandibular galnd',
        'submnd_salv_l1', 'lt sm Gland', 'lt submand gland',
        'left submandibular gl', 'l sm gland', 'left submand  gland',
        'left submandibular g', 'l submandidular', 'left submand gl',
        'left submand gland', 'lt submandblr', 'submand left (in ptv)',
        'l smgland', 'l submadibular', 'left sumandibular gland',
        'l submandibular_old', 'l sum mandib']
    opts['roi_map']['smg_right'] = [
        'r submandibular gland', 'right submandibular', 'rt submandibular',
        'right submandibular gland', 'r submandibular', 'r submandib',
        'right submandibular glan', 'r submand', 'r submand gland',
        'rt submand', 'submandibular rt', 'submandibular gland rt',
        'rt submandibular gland', 'submand right', 'submandibular r',
        'rt submang', 'submandibular right', 'subman rt', 'r sub mandibular',
        'r submandibular galnd', 'r sub mand gland', 'right smg', 'r smg',
        'rt smg', 'r  submandibular', 'right subm gland', 'rt submandib',
        'right sm gland', 'right submandibular aj', 'rt submandibular galnd',
        'submnd_salv_r1', 'rt sm Gland', 'rt submand gland',
        'right submandibular gl', 'r sm gland', 'right submand  gland',
        'right submandibular g', 'r submandidular', 'right submand gl',
        'right submand gland', 'rt submandblr', 'submand right (in ptv)',
        'r smgland', 'r submadibular', 'right sumandibular gland',
        'r submandibular_old', 'r sum mandib']
    opts['roi_map']['spinal_cord' ] = [
        'spinal cord - dn', 'cord', 'spinal cord', 'spinal_cord',
        'spinal cord sjj', 'spinal cord - sjj', 'spinal_cord_sp'
        'spine', 'spinal_canal_sp', 'spinal_cord_', 'spinal canal',
        'spinal_canal', 'spinal cord sld', 'cord b', 'SC', 'spinalcord']

    # Define list of paths for patient data to be analysed
    paths = []
    #paths = glob.glob('/Users/karl/data/head_and_neck/vspecial/3_patients__multiple_structures__all_mv/VT*')

    # Severity level for event logging.
    # Defined values are: 'NOTSET' (0), 'DEBUG' (10), 'INFO' (20),
    # 'WARNING' (30), 'ERROR' (40), 'CRITICAL' (50)
    log_level = 'INFO'

    # Create algorithm object
    alg = AnalysisAlgorithm(opts=opts, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application object
    app = Application(algs)

    # Run application for the selected data
    app.run(paths)
