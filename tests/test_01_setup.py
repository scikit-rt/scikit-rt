'''Test that environment is set up correctly.'''
import subprocess

def test_skrt():
    # Test that skrt can be imported.
    try:
        import skrt
        skrt_imported = True
    except ModuleNotFoundError:
        skrt_imported = False

    assert skrt_imported == True

def test_version_defined():
    # Test that version number is defined.
    try:
        import skrt
        version = skrt.__version__
    except ModuleNotFoundError:
        version = None

    assert isinstance(version, str)

def test_jupyter():
    # Test that jupyter can be run.
    try:
        returncode = subprocess.run(
                ["jupyter", "--version"], check=True).returncode
    except subprocess.CalledProcessError as called_process_error:
        returncode = called_process_error.returncode

    assert returncode == 0
