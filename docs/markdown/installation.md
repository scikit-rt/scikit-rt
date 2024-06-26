# Installation

Scikit-rt has a number of dependencies.  To avoid possible version conflicts,
it's recommended that it be installed to a clean virtual
environment.  This is achieved in the instructions below using
[conda](https://docs.conda.io/).  You can perform a user installation
with [pip](https://pip.pypa.io/), or a developer installation with
[git](https://git-scm.com/).   Installations have been tested
for Python 3.8 and Python 3.10.

For functionality involving image registration, it is also necessary
to <a href="#installation-and-setup-for-image-registration">install
image-registration software</a>.

## Installation and setup of scikit-rt

To be able to use scikit-rt, you can perform either a user installation
(recommended if you're not planning to contribute to the code base) or
a developer installation.

1. User installation
```
conda create --name skrt python=3.10
conda activate skrt
pip install scikit-rt
```

2. Developer installation
```
git clone https://github.com/scikit-rt/scikit-rt
cd scikit-rt
conda env create --file environment.yml
conda activate skrt
```

3. Environment activation and deactivation

Following either of the installations above, the scikit-rt environment
setup is included - that is, the conda `skrt` environment is active.  When
starting a new session, the environment can be activated
and deactivated with:
```
# Activate envrionment
conda activate skrt

# Deactivate environment
conda deactivate
```

4. Installation test

As a minimal test that the installation has been successful, try:
```
python -c "import skrt; print(skrt.__version__)"
```
This may take some time, while python code is being compiled, but
should eventually print the scikit-rt version number, and exit without errors.

## Updating scikit-rt

Scikit-rt is in active development. Following an initial installation,
it's possible to update to the latest version.

1. User installation

With the conda `skrt` environment active, run the command:

```
pip install --upgrade scikit-rt
```

2. Developer installation

From the scikit-rt directory, and assuming that no unmerged changes have
been made to the local copy of the code, run the command:

```
git pull
```

## Installation and setup for image registration

For image registration, and for atlas-based segmentation, scikit-rt requires
that at least one of the following image-registration packages be installed:

- [elastix](https://elastix.lumc.nl/)  
  For installation and environment setup, see *Getting started* section of
  the elastix manual:  
  [https://elastix.lumc.nl/download/elastix-5.1.0-manual.pdf](https://elastix.lumc.nl/download/elastix-5.1.0-manual.pdf).

- [matlab-skrt](https://github.com/kh296/matlab-skrt/)  
  For installation and environment setup, see:  
  [https://github.com/kh296/matlab-skrt/blob/main/examples/scikit-rt.md](https://github.com/kh296/matlab-skrt/blob/main/examples/scikit-rt.md)

- [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)  
  For installation and environment setup, see:  
  [http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)

There are three options for setting up the environment to allow use by
scikit-rt of the registration software (registration engine):

1. Before starting scikit-rt, follow the instructions linked for
   the relevant registration package.

2. At run time, use code similar to the following :

   ```
   from skrt.registration import set_engine_dir

   set_engine_dir("/path/to/elastix/directory", engine="elastix")
   set_engine_dir("/path/to/niftyreg/directory", engine="niftyreg")

    # matlab-skrt option 1: use source code with MATLAB executable.
    Defaults().matlab_app = "/path/to/directory/containing/MATLAB/executable"
    set_engine_dir("/path/to/directory/containing/mskrt/package"

    # matlab-skrt option 2: use compiled code with MATLAB runtime environment.
    Defaults().matlab_runtime = "/path/to/matlab/runtime/install/directory"
    set_engine_dir("/path/to/directory/containing/matlabreg/executable", engine="matlab")
    ```

   The parameter `engine` passed to the `set_engine_dir()` function
   may be omitted if the name of the registration engine is a substring
   of the installation path.

3. When creating registration or segmentation objects, pass the
   installation directory of the registration engine to be used via
   the `engine_dir` parameter.  When using `matlab-skrt`, the location
   of the MATLAB executable or of the MATLAB runtime installation must
   still be set:

    ```
    from skrt.core import Defaults
    
    # matlab-skrt option 1: define location of MATLAB executable.
    Defaults().matlab_app = "/path/to/directory/containing/MATLAB/executable"

    # matlab-skrt option 2: define location of MATLAB runtime installation.
    Defaults().matlab_runtime = "/path/to/matlab/runtime/install/directory"
    ```
