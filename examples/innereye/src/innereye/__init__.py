"""Classes related to Patients and Studies."""

from pathlib import Path
import shutil

import nibabel

from skrt.core import (fullpath, get_data_indices, get_indexed_objs,
        get_subdir_paths, make_dir)

def write_study(study, outdir="./innereye_datasets",
        image_types=None, images_to_write=None, verbose=True,
        image_size=None, voxel_size=None, fill_value=None,
        bands=None, require_structure_set=None, structure_sets_to_write=-1,
        roi_names=None, force_roi_nifti=False, bilateral_names=None,
        overwrite=True):
    """
    Save study images and associated structure sets for use by InnerEye.

    Except for <overwrite> parameters are passed on to the methods
    save_images_as_nifti() and/or save_structure_sets_as_nifti().

    **Parameters:**

    study - skrt.patient.Study
        Study object for which images and associated structure sets
        are to be written.

    overwrite - bool, default=True
        If False, skip images with pre-existing output directories.
        If True, delete pre-existing output directories.

    outdir - str, default='.'
        Top-level directory to which nifti files for InnerEye will
        be written for study.  Each output image will be in a separate
        sub-directory, along with a file per associated ROI.

    image_types - list/str/None, default=None
        Images types to be saved: None to save all, otherwise a list
        of image types to save, or a string specifying a single image
        type to save.

    images_to_write : dict, default=None
        Dictionary where the keys are image types and the values are
        lists of timestamp indices for the images to be saved,
        0 being the earliest and -1 being the most recent.  If set to
        None, all images are saved.

    verbose - bool, default=True
       Flag indicating whether to report progress.

    image_size - tuple, default=None
        Image dimensions (dx, dy, dz) in voxels for image resizing
        prior to writing.  If None, original image size is kept.

    voxel_size - tuple, default=None
        Voxel dimensions (dx, dy, dz) in mm for image resizing
        prior to writing.  If None, original voxel size is kept.

    fill_value - int/float, default=None
        Value used when extrapolating image outside data area.
        If None, use mininum value inside data area.

    bands - dict, default=None
        Nested dictionary of value bandings to be applied before
        image saving.  The primary key defines the type of image to
        which the banding applies.  Secondary keys specify band limits,
        and associated values indicte the values to be assigned.
        For example:

        - bands{'mvct' : {-100: -1024, 100: 0, 1e10: 1024}}

        will band an image of type 'mvct':

        - value <= -100 => -1024;
        - -100 < value <= 100 => 0;
        - 100 < value <= 1e10 => 1024.

    require_structure_set - list, default=None
        List of image types for which data are to be written only
        if there is an associated structure set.

    structure_sets_to_write: dict, default=-1
        Dictionary where the keys are image types and the values are
        lists of file indices for structure sets to be saved for
        a given image, 0 being the earliest and -1 being the most recent.
        If set to None, all of an image's structure sets are saved.

    roi_names : dict, default=None
        Dictionary of names for renaming ROIs, where the keys are new 
        names and values are lists of possible names of ROIs that should
        be assigned the new name. These names can also contain wildcards
        with the '*' symbol.

    force_roi_nifti : bool, default=False
        If True, force writing of dummy NIfTI file for all named ROIs
        not found in structure set.  If False, named ROIs not found in
        structure set are disregarded.

    bilateral_names : list, default=None
        List of names of ROIs that should be split into
        left and right parts.
    """

    # If require_structure_set is null, set to an empty list.
    require_structure_set = require_structure_set or []

    # Define study output directory.
    study_dir = Path(fullpath(outdir))

    # Obtain set of image types to be saved.
    if isinstance(image_types, str):
        image_types = [image_types]
    save_types = set(image_types or study.image_types).intersection(
            set(study.image_types))

    # Define image(s) to be saved for each image type.
    if not isinstance(images_to_write, dict):
        images_to_write = {save_type: images_to_write
                for save_type in save_types}

    # Define structure set(s) to be saved for each image type.
    if not isinstance(structure_sets_to_write, dict):
        structure_sets_to_write = {save_type: structure_sets_to_write
                for save_type in save_types}

    # Obtain dictionary associating indices to images.
    image_indices = get_data_indices(images_to_write, save_types)

    # Loop over image types.
    for save_type in save_types:

        # Obtain list of non-negative indices for images to be written.
        images = study.image_types[save_type]
        indices = image_indices.get(save_type, [])
        if isinstance(indices, int):
            indices = [indices]
        idxs = []
        if indices is True:
            idxs = range(len(images))
        else:
            for idx in indices:
                jdx = idx if idx >=0 else idx + len(images)
                idxs.append(jdx)

        # Loop over images of current type.
        for idx, im in enumerate(images):
            # Check whether image index satisfies requirements.
            if not idx in idxs:
                continue

            # Check for associated structure set(s).
            if (save_type in require_structure_set
                    and not get_indexed_objs(
                        im.structure_sets,
                        structure_sets_to_write.get(save_type, False))):
                continue

            # Check if output directory already exists.
            im_dir = (study_dir / save_type.upper()
                    / f"{im.timestamp}_{idx+1:03}")
            if not overwrite and im_dir.exists():
                if verbose:
                    print("Not overwriting for image with existing "
                            f"output directory: {str(im_dir)}")
                continue
            if im_dir.exists():
                shutil.rmtree(im_dir)
            im_dir.mkdir(parents=True, exist_ok=True)

            # Write image.
            study.save_images_as_nifti(
                    outdir=im_dir,
                    image_types=[save_type],
                    times={save_type: [idx]},
                    verbose=verbose,
                    image_size=image_size,
                    voxel_size=voxel_size,
                    fill_value=fill_value,
                    bands=bands,
                    require_structure_set=require_structure_set)

            # Write associated structure set(s).
            study.save_structure_sets_as_nifti(
                    outdir=im_dir,
                    image_types=[save_type],
                    times={save_type: [idx]},
                    files={save_type:
                        structure_sets_to_write.get(save_type, -1)},
                    verbose=verbose,
                    image_size=image_size,
                    voxel_size=voxel_size,
                    roi_names=roi_names,
                    force_roi_nifti=force_roi_nifti,
                    bilateral_names=bilateral_names)

def write_patient(patient,
        outdir="./innereye_datasets", studies_to_write=None,
        overwrite=True, **kwargs):

    """
    Save patient images and associated structure sets for use by InnerEye.

    **Parameters:**

    patient - skrt.patient.Patient
        Patient object for which images and associated structure sets
        are to be written.

    outdir - str, default='.'
        Top-level output directory.  Within this, there will be a
        patient sub-directory, containing a sub-directory for each
        study, containing in turn a sub-directory for each output
        image along with associated structure set(s).

    studies_to_write : list, default=None
        List of indices of studies for which data are to be written,
        0 being the earliest study and -1 being the most recent.  If set to
        None, data for all studies are written.

    overwrite - bool, default=True
        If True, delete delete any pre-existing patient folder.

    **kwargs
        Keyword arguments passed on to
            innereye.write_study().
        For details, see this method's documentation.
    """
    # Define patient output directory.
    patient_dir = make_dir(Path(fullpath(outdir)) / patient.id,
            overwrite=overwrite)

    # If study_indices is None, set to select all studies.
    if studies_to_write is None:
        studies_to_write = True

    # Process selected studies.
    for study in get_indexed_objs(patient.studies, studies_to_write):
        study_dir = patient_dir / study.timestamp
        write_study(study, patient_dir / study.timestamp, overwrite=False,
                **kwargs)

def write_catalogue(
        datadir="innereye_datasets",
        outdir=None,
        out_csv="dataset.csv",
        nii_suffixes="nii.gz",
        nz_min=16,
        roi_names=None,
        require_rois_in_image=True):
    """
    Create catalogue of NIfTI files, in format required by InnerEye software.

    The catalogue is in CVS format.  The values stored for each file are:

    - subject : unique positive integer;
    - filePath : path relative to top-level directory;
    - channel : string identifying data as "ct" or ROI;
    - seriesId : DICOM series identifier (left blank);
    - institutionId : Institution identifier (lef blank);
    - DIM_X : data array x-dimension;
    - DIM_Y : data array y-dimension;
    - DIM_Z : data array z-dimension.

    **Parameters:**

    datadir: str, default="innereye_datasets"
        Path to directory containing data to be catalogued.

    outdir: str, default=None
        Directory to which catalogue is to be written.  If None,
        write to <datadir>.

    out_csv: str, defaul="dataset.csv"
        Name of file to which catalogue is to be written.

    nii_suffixes: str, list, default="nii.gz":
        Suffix, or list of suffixes, for NIfTI files.

    nz_min: int, default=16
        Minimum number of z-slices required for an image.

    roi_names: list, default=None
        List of names of ROIs to be catalogued.  If None, catalogue
        all ROIs.

    require_rois_in_image: bool, default=True
        If True, only catalogue images with associated ROI data.
    """
    # Initialise variables.
    datadir = Path(fullpath(datadir))
    outdir = Path(fullpath(outdir or datadir))
    if isinstance(nii_suffixes, str):
        nii_suffixes = [nii_suffixes]
    roi_names = roi_names or []
    catalogue_entries = {}

    # Loop over patients.
    for patientdir in get_subdir_paths(datadir):
        # Loop over studies.
        for studydir in get_subdir_paths(patientdir):
            # Loop over image types.
            for imagedir in get_subdir_paths(studydir):
                # Loop over timestamps.
                for timedir in get_subdir_paths(imagedir):
                    channels = {}

                    # Loop over NIfTI files.
                    for niipath in sorted(
                            list(timedir.glob(f"**/*.{nii_suffixes}*"))):

                        # Determine channel from filename.
                        name_parts = str(niipath.name).split(".")[0].split("_")
                        channel_ok = True
                        if "RTSTRUCT" == name_parts[0]:
                            channel = "_".join(name_parts[5:]).lower()
                            channel_ok = ((channel in roi_names)
                                    or (not roi_names))
                        elif "MVCT" == name_parts[0]:
                            channel = "ct"
                        else:
                            channel = name_parts[0].lower()
                        if not channel_ok:
                            continue

                        # Load NIfTI data, and determine size.
                        nii = nibabel.load(niipath)
                        fdata = nii.get_fdata()
                        nx, ny, nz = fdata.shape

                        # Skip dataset if the number of z-slices is too small.
                        if nz < nz_min:
                            print(f"{timedir} - {nz} slices - skipping")
                            break

                        # Skip dataset if a required ROI may not be entirely
                        # inside the image (part of mask at the image border).
                        in_scan = True
                        if require_rois_in_image and channel != "ct":
                            in_scan = (fdata[:,:,0].sum() < 0.5
                                    and fdata[:,:, nz - 1].sum() < 0.5
                                    and fdata[:, 0, :].sum() < 0.5
                                    and fdata[:, ny - 1, :].sum() < 0.5
                                    and fdata[0, :, :].sum() < 0.5
                                    and fdata[nx - 1, :, :].sum() < 0.5)
                        if not in_scan:
                            print(f"{timedir} - {channel} not in image "
                                   "- skipping")
                            continue

                        # Store file data.
                        channels[channel] = [
                                str(niipath.relative_to(datadir)),
                                "", "", nx, ny, nz]

                # Skip dataset if a required ROI is absent;
                # otherwise catalogue the file data.
                missing_roi_names = []
                if roi_names:
                    for roi_name in roi_names:
                        if roi_name not in channels:
                            missing_roi_names.append(roi_name)
                if missing_roi_names:
                    print(f"{timedir} - missing {missing_roi_names} ")
                else:
                    catalogue_entries[timedir] = channels

    # Write to CSV file values required by InnerEye software.
    lines = [
            "subject,filePath,channel,seriesId,institutionId,DIM_X,DIM_Y,DIM_Z"
            ]
    for idx, timedir in enumerate(catalogue_entries):
        subject = idx + 1
        for channel in catalogue_entries[timedir]:
            file_path, series_id, institution_id, nx, ny, nz = \
                catalogue_entries[timedir][channel]
            lines.append(f"{subject},{file_path},{channel},{series_id},"
                             f"{institution_id},{nx},{ny},{nz}")
    with open(outdir / out_csv, "w") as out_file:
        out_file.write("\n".join(lines))
