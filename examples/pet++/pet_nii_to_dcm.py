"""
Module defining function for NIfTI-to_DICOM conversion of PET++ reconstruction.
"""
from pathlib import Path

from skrt import Image
from skrt.core import get_logger

def pet_nii_to_dcm(nii_path=None, outdir=None, overwrite=True,
        patient_id=None, root_uid=None, header_extras=None,
        z0=-198.25, log_level="INFO"):
    """
    Read NIfTI file from PET++ reconstruction, and write out in DICOM format.

    **Parameters:**

    nii_path: str/pathlib.Path, default=None
        Path to NIfTI file to be read.

    outdir : str/pathlib.Path, default=None
        Path to directory where DICOM data are to be written.

    overwrite : bool, default=True
        If True, delete any pre-existing DICOM files from output
        directory before writing.

    patient_id : str, default="pet_patient"
        Patient identifier.

    root_uid : str, default=None
        Root to be used in Globally Unique Identifiers (GUIDs).  This
        should uniquely identify the institution or group generating
        the GUID.  If None, the value of pydicom.uid.PYDICOM_ROOT_UID
        is used.

        A unique root identifier can be obtained free of charge from
        Medical Connections:
        https://www.medicalconnections.co.uk/FreeUID/

    header_extras : dict, default={}
        Dictionary of attribute-value pairs for applying arbitary
        updates to a pydicom.dataset.FileDataset

    z0 : float, default=-198.25
        Coordinate along z-axis of centre of most-inferior image slice.

    log_level: str/int, default="INFO"
        Severity level for event logging.
    """
    # Create message logger.
    logger = get_logger(name="pet_nii_to_dcm", log_level=log_level)

    # Check that path to input NIfTI file is defined.
    if nii_path is None:
        logger.error("Input NIfTI file undefined - exiting.")
        return

    # Check that path to output directory is defined.
    if outdir is None:
        logger.error("Output directory undefined - exiting.")
        return

    # Check that input NIfTI file exists.
    nii_path = Path(nii_path)
    if not nii_path.exists():
        logger.error(f"Input NIfTI file not found at '{nii_path}' - exiting.")
        return

    # Create Image object from NIfTI data.
    nii1 = Image(path=str(nii_path))

    # In Scikit-rt NIfTI data are handled using nibabel:
    #     https://nipy.org/nibabel/
    # and DICOM data are handled using pydicom:
    #     https://pydicom.github.io/pydicom/
    # Both external packages store images as intensity values
    # in numpy arrays, but there are two important differences:
    #     - indices for an image slice have the order
    #       [row][column] in pydicom vs [column][row] in nibabel;
    #     - axis definitions follow radiology convention in pydicom
    #       vs neurology convention in nibabel - for discussion of
    #       the conventions, see:
    #           https://nipy.org/nibabel/neuro_radio_conventions.html
    #
    # The NIfTI output from the PET++ reconstruction doesn"t conform
    # with the nibabel conventions.  To correct for this, the
    # first and second axes of the numpy need to be transposed,
    # and the two axes need to be either reflected about zero
    # (x -> -x) or inverted (x -> x_max - x).
    #
    # WARNING: It"s currently a guess that the first two axes should
    #          be reflected.  The only check has been for an image
    #          centred on (0,0), where reflection and inversion
    #          give the same result.
    #
    # Origin along z-axis isn"t set correctly in the reconstruction,
    # and needs to be set manually here.
    nii2_data = nii1.get_data().transpose(1, 0, 2)[::-1, ::-1, :]
    nii2_affine = nii1.get_affine(standardise=True)
    nii2_affine[2][3] = z0
    nii2 = Image(path=nii2_data, affine=nii2_affine)

    # Write the image in dicom format.
    nii2.write(outname=outdir, patient_id=patient_id, modality="PT",
        root_uid=root_uid, header_extras=header_extras)

if "__main__" == __name__:
    # Set parameter values for performing NIfTI-to-DICOM conversion.
    # WARNING: At least nii_path and root_uid should be changed
    #          to meaningful values.
    nii_path = None
    outdir = "dicom_dir"
    overwrite = True
    patient_id = "patient_id"
    root_uid = None
    header_extras = {
        "PatientName": "patient_name",
        "RescaleSlope": 0.001,
        }
    z0=-198.25
    log_level = "INFO"

    # Perform NIfTI-to-DICOM conversion.
    pet_nii_to_dcm(nii_path=nii_path, outdir=outdir, overwrite=overwrite,
            patient_id=patient_id, root_uid=root_uid,
            header_extras=header_extras, z0=z0, log_level=log_level)
