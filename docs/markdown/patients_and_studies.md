# Patients and studies

The `Patient` and `Study` classes allow multiple images, structure sets,
doses and plans associated with a single patient to be read into one
object.  These classes
can load DICOM and/or NIfTI files organised as described below (sorted data),
and can load arbitrarily organised DICOM files (unsorted data).  In the
cases of both sorted and unsorted data, all files relating to a single
patient should be placed under a separate directory, the name of which
is taken as the patient's identifier.

## Sorted data

### Patient and study directories

The top level directory represents the entire **patient**.

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

Each study can contain images of various imaging modalities, and associated structure sets. Within a study directory, there can be three "special" directories, named `RTSTRUCT`, `RTDOSE`, and `RTPLAN`, containing structure sets, dose fields, and radiotherapy plans, respectively.

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

## Patient class

A `Patient` object is created by providing the path to the top-level patient directory, and indicating if the data are unsorted DICOM data:
```
from skrt import Patient
p = Patient('mypatient1') # load sorted data
p = Patient('mypatient2', unsorted_dicom=True) # load unsorted DICOM data
```
A list of the patient's associated studies is stored in `p.studies`.

Additional properties can be accessed:
- Patient ID: `p.id`
- Patient sex: `p.get_sex()`
- Patient age: `p.get_age()`
- Patient birth date: `p.get_birth_date()`

### Sorted and unsorted DICOM files

If a patient's data consists of sorted DICOM files, the data may be read
both with `unsorted_dicom=False` (default) and with `unsorted_dicom=True`.
Loading of sorted data tends to be faster, with object linking (for
example, association of a structure set with an image) based
on file organisation.  When loading unsorted DICOM files, object linking
is based on unique identifiers and references that the files contain.

Unsorted DICOM files can be copied as sorted files:
```
from skrt import Patient
# Load unsorted DICOM files.
p = Patient('mypatient2', unsorted_dicom=True)
# Write sorted DICOM files to sub-directory mypatient2 of directory my_sorted_dicom.
p.copy_dicom('my_sorted_dicom')
```

<!---
#### Writing a patient tree

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
--->

## Study class

A `Study` object stores images and structure sets. A list of studies can be extracted from a `Patient` object via the property `Patient.studies`. You can access the study's path via `Study.path`. If the study was nested inside a subdirectory, the name of that subdirectory is accessed via `Study.subdir`.

### Images

For each imaging modalitiy subdirectory inside the study, a new class property will be created to contain a list of images of that modality, called `{modality}_images`, where the modality is taken from the subdirectory name (note, this is always converted to lower case). E.g. if there were directories called `CT` and `MR`, the `Study` object would have properties `ct_images` and `mr_images` containing lists of `Image` objects.

### Structure sets

The study's structure sets can be accessed in two ways. Firstly, the structure sets associated with an image can be extracted from the `structure_sets` property of the `Image` itself; this is a list of `StructureSet` objects. E.g. to get the newest structure set for the oldest CT image in the oldest study, you could run:
```
p = Patient('mypatient1')
s = p.studies[0]
structure_set = s.ct_images[0].structure_sets[-1]
```

In addition, structure sets associated with each imaginging modality will be stored in a property of the `Study` object called `{modality}_structure_sets`. E.g. to get the oldest CT-related structure set, you could run:
```
structure_set = s.ct_structure_sets[0]
```

The `StructureSet` object also has an associated `image` property (`structure_set.image`), which can be used to find out which `Image` is associated with that structure set.

### Treatment plans
Data from radiotherapy treatment plans are made available as `Plan` objects.  `Study` objects group `Plan` objects in lists with names of the form `<machine>_plans`, where `<machine>` identifies the treatment machine to which the plan relates.  `Image` objects and `StructureSet` objects group together all plans associated with them, in a list `plans`.  A `Dose` object is only ever derived from a single plan, referenced as `plan`.  Different ways of accessing a `Plan` object are shown below.

```
# Access plan from Study.
plan = s.la3_plans[0]

# Access plan from Image to which it relates.
plan = s.ct_images[0].plans[0]

# Access plan from StructureSet to which it relates.
plan = s.ct_structure_sets[0].plans[0]

# Access plan from Dose (each of which can relate to only a single plan).
plan = s.ct_doses[0].plan
```

Planning information that can then be accessed include the following:

```
# Obtain plan name.
name = plan.get_name()

# Obtain plan approval status.
status = plan.get_approval_status()

# Obtain planned number of treatment fractions.
n_fraction = plan.get_n_fraction()

# Obtain planned dose to target (i.e. tumour).
target_dose = plan.get_target_dose()

# Obtain list of ROI objects defined as targets.
targets = plan.get_targets()

# For any target, obtain the planning constraints and weight.
target_constraint_weight = targets[0].constraint.weight
target_prescription_dose = targets[0].constraint.prescription_dose
target_maximum_dose = targets[0].constraint.maximum_dose
target_minimum_dose = targets[0].constraint.minimum_dose
target_underdose_volume_fraction = targets[0].constraint.underdose_volume_fraction

# Obtain list of ROI objects defined as organs at risk.
oars = plan.get_organs_at_risk()

# For any organ at risk, obtain the planning constraints and weight.
oar_constraint_weight = oars[0].constraint.weight
oar_maximum_dose = oars[0].constraint.maximum_dose
oar_full_volume_dose = oars[0].constraint.full_volume_dose
oar_overdose_volume_fraction = oars[0].constraint.overdose_volume_fraction

# For any target or organ at risk, obtain a Dose object, representing
# a dose objective passed to the optimiser.  Allowed values for the
# objective correspond to the constraints and weights.
# Data of the resulting object may be visualised in the same way
# as for any other Dose object.
max_dose = plan.get_dose_objective('maximum_dose')
max_dose.view()
```

The following is an Example plot of maximum-dose objective.

<img src="docs/images/maximum_dose.png" alt="Plot of maximum-dose objective" height="400"/>
