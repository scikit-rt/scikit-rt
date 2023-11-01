'''
Application to write patient images and structure sets for nnU-net.

For information about nnU-net (non-new U-net), see:
    https://doi.org/10.1038/s41592-020-01008-z
    https://github.com/MIC-DKFZ/nnUNet/
'''
import os
import subprocess

from pathlib import Path
from platform import system
from subprocess import run
from sys import argv

from skrt.application import Algorithm, Application
from skrt.core import fullpath, is_list, make_dir


class NnunetTrain(Algorithm):
    """
    Application to write patient images and structure sets for nnU-net.

    Methods:
        __init__ -- Return instance of CreateNnunetDataset class,
                    with properties set according to options dictionary.
        execute  -- Write patient images and structure sets for nnU-net.
        finalise -- Write metadata for training dataset.
    """

    def __init__(self, opts=None, name=None, log_level=None):
        """
        Create instance of CreateNnunetDataset class.

        opts: dict, default=None
            Dictionary for setting algorithm attributes.  If null,
            set to empty dictionary.

        name: str, default=None
            Name for identifying algorithm instance.  If null,
            set to empty dictionary.

        log_level: str/int/None, default=None
            Severity level for event logging.  If None, log_level is
            set to the value of skrt.core.Defaults().log_level.
        """
        self.setup_script = fullpath("nnunet_conda.sh")

        # Top-level data directory.  Within this, there will be a dataset
        # sub-directory.  When creating a dataset for training, this will
        # contain sub-directories for training images (imagesTr) and training
        # segmentations (labelsTr).  If self.test_fraction is greater than
        # zero, the dataset sub-directory will also contain sub-directories
        # for test images (imagesTs) and test segmentations (labelsTs).
        # When creating a dataset for inference, images will be placed
        # directly in the dataset sub-directory.
        self.topdir = './data'

        # Identifier(s) of dataset(s) to be processed.
        self.dataset_ids = [1]

        self.preprocess = True
        self.train = True

        self.config = "3d_fullres"

        self.folds = range(5)
        self.trainer = "nnUNetTrainer"
        self.device = "cpu"

        # Override default properties, based on contents of opts dictionary.
        super().__init__(opts, name, log_level)

        self.nnunet_setup = get_nnunet_setup(self.setup_script)
        self.nnunet_env = get_nnunet_env(self.topdir)
        for nnunet_dir in self.nnunet_env.values():
            make_dir(nnunet_dir, overwrite=False)
        self.nnunet_env = {**os.environ.copy(), **self.nnunet_env}

        if not is_list(self.dataset_ids):
            self.dataset_ids = [self.dataset_ids]
        if not is_list(self.folds):
            self.folds = [self.folds]

    def finalise(self):
        if self.preprocess:
            command = (f"{self.nnunet_setup}nnUNetv2_plan_and_preprocess "
                    f"-d {get_dataset_ids_as_string(self.dataset_ids)} "
                    f"--verify_dataset_integrity {get_config(self.config)}")
            print("\n" + command.split(";")[-1] + "\n")
            subprocess.run(command, env=self.nnunet_env, shell=True)

        if self.train:
            for dataset_id in self.dataset_ids:
                for fold in self.folds:
                    command = (f"{self.nnunet_setup}nnUNetv2_train "
                            f"{dataset_id} {self.config} {fold} "
                            f"-tr {self.trainer} -device {self.device}")
                    print("\n" + command.split(";")[-1] + "\n")
                    subprocess.run(command, env=self.nnunet_env, shell=True)
        return self.status

def get_nnunet_setup(setup_script=None):
    if not setup_script:
        return ""
    return f"source {setup_script};"

def get_dataset_ids_as_string(dataset_ids):
    if not is_list(dataset_ids):
        dataset_ids = [dataset_ids]
    return " ".join([str(dataset_id) for dataset_id in dataset_ids])

def get_config(config=None):
    if not config:
        return ""
    return f" -c {config}"

def get_nnunet_env(topdir="./nnunet"):
    """
    Define environment variables for running nnU-net.
    """
    topdir = fullpath(topdir, pathlib=True)
    return {f"nnUNet_{var}": str(topdir / f"nnUNet_{var}")
            for var in ["raw", "preprocessed", "results"]}

def get_app(setup_script=''):
    '''
    Define and configure application to be run.
    '''
    opts = {}

    if "Linux" == system():
        opts["topdir"] = str(Path("~/codeshare/nnunet/data").expanduser())
    else:
        opts["topdir"] = "."

    opts["dataset_id"] = 1
    opts["preprocess"] = False
    opts["train"] = True
    opts["trainer"] = "nnUNetTrainer_001"
    opts["folds"] = 1

    if 'Ganga' in __name__:
        opts['alg_module'] = fullpath(argv[0])

    # Set the severity level for event logging
    log_level = 'INFO'

    # Create algorithm object
    alg = NnunetTrain(opts=opts, name=None, log_level=log_level)

    # Create the list of algorithms to be run (here just the one)
    algs = [alg]

    # Create the application
    app = Application(algs=algs, log_level=log_level)

    return app

if '__main__' == __name__:
    # Define and configure the application to be run.
    app = get_app()

    # Run application for the selected data
    app.run()

if 'Ganga' in __name__:
    # Define scripts for setting analysis environment
    skrt_setup = fullpath('skrt_conda.sh')
    nnunet_setup = fullpath('nnunet_conda.sh')

    # Define and configure the application to be run.
    ganga_app = SkrtApp._impl.from_application(get_app(), skrt_setup)
    ganga_app.algs[0].opts["setup_script"] = nnunet_setup

    # Define processing system.
    if "Linux" == system():
        backend = Condor()
        backend.cdf_options["request_memory"]="16G"
    else:
        backend = Local()

    # Define job name
    name = "nnunet_train"

    # Define list of outputs to be saved
    outbox = []

    # Create the job, and submit to processing system
    j = Job(application=ganga_app, backend=backend,
            postprocessors=[], name=name)
    j.submit()
