"""
Application that runs simple algorithm.
"""

import glob
try:
    import ROOT
    root_available = True
except ModuleNotFoundError:
    root_available = False

from skrt.application import Algorithm, Application


class RootAlgorithm(Algorithm):
    """Subclass of Algorithm, for analysing patient data"""

    def __init__(self, opts={}):

        # Configurable variables
        # Maximum number of patients to be analysed
        self.max_patient = 10

        # Initialise to null ROOT file
        self.root_file = None
        self.root_file_name = None

        # Call to __init__() method of base class
        # sets values for object properties based on dictionary optDict
        super().__init__(opts)

        # If ROOT package available and filename specified, set up ROOT file.
        if root_available:
            # Include histogram underflows and overflows in statistics.
            ROOT.TH1F.StatOverflows(ROOT.kTRUE)
            if self.root_file_name:
                self.root_file = ROOT.TFile(self.root_file_name, 'RECREATE')
        else:
            self.status.code = 1
            self.status.reason = 'ROOT not available - try:\n    conda install'\
                    ' --strict-channel-priority --channel conda-forge root'

        # Counter of number of patients to be analysed
        self.n_patient = 0


    # The execute() method is called once for each patient, the data for
    # which is passed via an instance of the skrt.patient.Patient class.
    def execute(self, patient=None):

        # Increase patient count
        self.n_patient += 1

        # Print patient identifier and path
        print(f"{patient.id}: {patient.path}")

        # Set non-zero status code if maximum number of patients reached
        if self.n_patient >= self.max_patient:
            self.status.code = 1
            self.status.reason = f"Reached {self.n_patient} patients"
            self.finalise()

        return self.status

    def finalise(self):

        print (f"Number of patients analysed = {self.n_patient}")

        if self.root_file:
            self.root_file.Write()

        return self.status

if "__main__" == __name__:

    # Create a dictionary of options to be passed to the algorithm
    opts = {}
    # Set the maximum number of patients to be analysed
    opts["max_patient"] = 2
    opts['root_file_name'] = 'analysis.root'

    # Create algorithm object
    alg = RootAlgorithm(opts)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application object
    app = Application(algs)

    # Define list of paths for patient data to be analysed
    paths = []

    # Run application for the selected data
    if app.status.ok():
        app.run(paths)
    else:
        print(app.status.reason)
