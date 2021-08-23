# Patients and Studies

The `Patient` and `Study` classes allow multiple medical images and structure sets associated with a single patient to be read into one object. 

## Expected file structure

### Patient and study file structure

The files for a patient should be sorted into a specific structure in order for the `Patient` class to be able to read them in correctly.

The top level directory represents the entire **patient**; this should be a directory whose name is the patient's ID.

The next one or two levels represent **studies**. A study is identified by a directory whose name is a timestamp, which is a string with format `YYYYMMDD_hhmmss`. These directories can either appear within the patient directory, or be nested in a further level of directories, for example if you wished to separate groups of studies.

A valid file structure could look like this:
```
mypatient1
--- 20200416_120350
--- 20200528_100845
```
This would represent a patient with ID `mypatient1`, with two studies, one taken on 16/04/2020 and one taken on 28/05/2020.

Another valid file structure could be:
```
mypatient1
--- planning
------ 20200416_120350
------ 20200528_100845
--- relapse
------ 20211020_093028
```
This patient would have three studies, two in the "planning" category and two in the "relapse" category.

### Files within a study

Each study can contain images of various imaging modalities, and associated structure sets. Within a study directory, there can be three "special" directories, named `RTSTRUCT`, `RTDOSE`, and `RTPLAN` (currently only `RTSTRUCT` does anything; the others will be updated soon), containing structure sets, dose fields, and radiotherapy plans, respectively.

Any other directories within the study directory are taken to represent an **imaging modality**. The structure sets associated with this modality should be nested inside the `RTSTRUCT` directory inside directories with the same name as the image directories. The images and structure sets themselves should be further nested inside a timestamp directory representing the time at which that image was taken.

For example, if a study containined both CT and MR images, as well as two structure sets associated with the CT image, the file structure should be as follows:
```
20200416_120350
--- CT
------ 20200416_120350
--------- 1.dcm
--------- 2.dcm ... etc
--- MR
------ 20200417_160329
--------- 1.dcm
--------- 2.dcm ... etc
--- RTSTRUCT
------ CT
---------- 20200416_120350
-------------- RTSTRUCT_20200512_134729.dcm
-------------- RTSTRUCT_20200517_162739.dcm
```

## The Patient class

A `Patient` object is created by providing the path to the top-level patient directory:
```
from quickviewer.prototype import Patient
p = Patient('mypatient1')
```

A list of the patient's associated studies is stored in `p.studies`.

Additional properties can be accessed:
- Patient ID: `p.id`
- Patient sex: `p.get_sex()`
- Patient age: `p.get_age()`
- Patient birth date: `p.get_birth_date()`

### Writing a patient tree

A patient's files can be written out in nifti or numpy format. By default, files will be written to compress nifti (`.nii.gz`) files. This is done via the `write` method:
```
p.write('some_dir')
```
where `some_dir` is the directory in which the new patient folder will be created (if not given, it will be created in the current directory).

By default, all imaging modalities will be written. To ignore a specific modality, a list `to_ignore` can be provided, e.g. to ignore any MR images:
```
p.write('some_dir', to_ignore=['MR'])
```

To write out structure sets as well as images, the `structure_set` argument should be set. This can be either:
- `'all'`: write all structure sets.
- The index of the structure set (e.g. to write only the newest structure set for each image, set `structure_set=-1`)
- A list of indices of structure sets to write (e.g. to write only the oldest and newest, set `structure_set=[0, -1]`)

By default, no structure sets will be written, as conversion of structures from contours to masks can be slow for large structures.


## The Study class

A `Study` object stores images and structure sets. A list of studies can be extracted from a `Patient` object via the property `Patient.studies`. You can access the study's path via `Study.path`. If the study was nested inside a subdirectory, the name of that subdirectory is accessed via `Study.subdir`.

### Images

For each imaging modalitiy subdirectory inside the study, a new class property will be created to contain a list of images of that modality, called `{modality}_scans`, where the modality is taken from the subdirectory name (note, this is always converted to lower case). E.g. if there were directories called `CT` and `MR`, the `Study` object would have properties `ct_scans` and `mr_scans` containing lists of `Image` objects.

### Structure sets

The study's structure sets can be accessed in two ways. Firstly, the structure sets associated with an image can be extracted from the `structs` property of the `Image` itself; this is a list of `RtStruct` objects. E.g. to get the newest structure set for the oldest CT image in the oldest study, you could run:
```
p = Patient('mypatient1')
s = p.studies[0]
structure_set = s.ct_scans[0].structs[-1]
```

In addition, structures associated with each imaginging modality will be stored in a property of the `Study` object called `{modality}_structs`. E.g. to get the oldest CT-related structure set, you could run:
```
structure_set = s.ct_structs[0]
```

The `RtStruct` object also has an associated `image` property (`structure_set.image`), which can be used to find out which `Image` is associated with that structure set.
