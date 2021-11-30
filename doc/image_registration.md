# Image registration with scikit-rt and elastix

## Setup

### 1. Installing scikit-rt
1. Ensure you have either [pip](https://pypi.org/project/pip/) or [anaconda](https://docs.anaconda.com/anaconda/install/index.html)/[miniconda](https://docs.conda.io/en/latest/miniconda.html) (on Windows I would recommend Anaconda) installed on your machine.
2. Open a terminal (if using pip on mac/linux) or a conda terminal (if using Anaconda - access by opening Anaconda-Navigator, clicking "Environments", then clicking the play button and "Open Terminal").
3. Run the command `pip install scikit-rt` to install the scikit-rt package.
4. If on windows, also run `conda install -c conda-forge shapely` to install the correct version of the shapely package.

Now you should be able to launch a Jupyter server, either by:
(a) Running `jupyter notebook` in the terminal, or;
(b) If using Anaconda, clicking the play button then "Open with Jupyter Notebook"

If you create a new Jupyter notebook, you should be able to import scikit-rt by running `import skrt`.


### 2. Installing elastix

1. Download the elastix binaries corresponding to your operating system from here: https://github.com/SuperElastix/elastix/releases/tag/5.0.1
2. Extract the folder and either make a note of where you saved it, or add the `bin` folder to your `$PATH` environment variable and `lib` folder to your `$DYLDLIBRARYPATH` variable (if using mac/linux).


### 3. Downloading example elastix parameter files

A collection of example elastix parameter files is available at the [elastix parameter zoo](https://elastix.lumc.nl/modelzoo/). 
- To obtain the files on your machine, download the git repo: https://github.com/SuperElastix/ElastixModelZoo. 
- The parameter files for each example can be found inside the `models/` directory. 
- The subdirectories `Par0001` etc refer to the parameter sets listed on the model zoo webpage (https://elastix.lumc.nl/modelzoo/).


## Usage

This code is best used inside a Jupyter notebook, but can also be implemented via a python script.

Here is a code snippet to perform a registration and qualitatively view the results:
```
from skrt.registration import Registration, set_elastix_dir

# Set elastix location
set_elastix_dir("some/location/elastix-5.0.1-mac")

# Perform registration
reg = Registration(
  fixed="some/location/fixed_image_dicoms",
  moving="some/location/moving_image_dicoms",
  pfile=[
    "some/location/ElastixModelZoo/models/default/Parameters_Rigid.txt",
    "some/location/ElastixModelZoo/models/default/Paramterers_BSpline.txt"
  ],
  outdir="my_results"
)

# View initial and final images
reg.view_init()
reg.view_comparison()
```

Explanation of each part of the code:

1. Import the image registration class
```
from skrt import Registration
```

2. If you didn't add your elastix installation to your `$PATH` variable, you'll need to manually set the location of your elastix installation:
```
from skrt import set_elastix_dir
set_elastix_dir("some/location/elastix-5.0.1-mac")
```
On mac/linux, this should be the directory containing the `bin` and `lib` folders; on Windows, it should be the  directory containing `elastix.exe`.

3. Create a Registration object; this will automatically run the registration, if it hasn't already been run.
```
reg = Registration(
  fixed="some/location/fixed_image_dicoms",
  moving="some/location/moving_image_dicoms",
  pfile=[
    "some/location/ElastixModelZoo/models/default/Parameters_Rigid.txt",
    "some/location/ElastixModelZoo/models/default/Paramterers_BSpline.txt"
  ],
  outdir="my_results"
)
```
The arguments to the `Registration` class initialisation are:
- `fixed`: Path to a directory containing the dicom files for the fixed image. Can also be a numpy array or the path to a single nifti file.
- `moving`: Path to a directory containing the dicom files for the moving image. Can also be a numpy array or the path to a single nifti file.
- `pfile`: Path or list of paths to elastix parameter file(s). If a list is given, each parameter file will be used in series, with its results passed to the next step of the registration.
- `outdir`: Directory in which to save the results of the registration. If not given, results will be saved in the current directory.

For each parameter file, a subdirectory will be created in `outdir` with the same name as the parameter file, minus `.txt`. If an elastix transform file is already found in that directory, the registration will not be rerun for that parameter file, and instead the existing transform will be used. To force rerunning of the registration, set `force=True` when initialising the `Registration` class.

4. Viewing the images
A `Registration` can be used to launch an interactive viewer showing either the initial or final images side-by-side. To view the initial fixed and moving images, run
```
reg.view_init()
```
To view the fixed image and transformed moving image, run:
```
reg.view_comparison()
```
