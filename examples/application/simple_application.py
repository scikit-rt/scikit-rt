"""
Application that runs simple algorithm.
"""

import glob

from skrt.application import Algorithm, Application


class SimpleAlgorithm(Algorithm):
    """Subclass of Algorithm, for analysing patient data"""

    def __init__(self, opts={}):

        # Configurable variables
        # Maximum number of patients to be analysed
        self.max_patient = 10

        # Call to __init__() method of base class
        # sets values for object properties based on dictionary optDict
        Algorithm.__init__(self, opts)

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

        return self.status

if "__main__" == __name__:

    # Create a dictionary of options to be passed to the algorithm
    opts = {}
    # Set the maximum number of patients to be analysed
    opts["max_patient"] = 2

    # Create algorithm object
    alg = SimpleAlgorithm(opts)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application object
    app = Application(algs)

    # Define list of paths for patient data to be analysed
    paths = []

    # Run application for the selected data
    app.run(paths)
