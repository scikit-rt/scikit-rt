"""
Class DicomWriter for writing data in DICOM format.

DICOM writing is supported for instances of:

- skrt.dose.Dose;
- skrt.image.Image;
- skrt.simulation.SyntheticImage.
"""

import datetime
from pathlib import Path
import time

import numpy as np
import pydicom
from pydicom import uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import pydicom._storage_sopclass_uids as sop

from skrt.core import fullpath


class DicomWriter:
    """
    Class to help with writing data in DICOM format.

    DICOM writing is supported for instances of:

    - skrt.dose.Dose;
    - skrt.image.Image;
    - skrt.structures.StructureSet.

    **Methods:**

    - **_init__()** : Create instance of DicomWriter class.
    - **add_image_info()** : Add image information to file dataset.
    - **add_structure_set_info()** : Add structure-set information
      to file dataset.
    - **create_file_dataset()** : Create new pydicom.dataset.FileDataset object.
    - **get_file_dataset()** : Retrive pydicom.dataset.FileDataset from source.
    - **get_media_storage_sop_class_uid()** : Determine uid given modality.
    - **get_path_with_timestamp()** : Obtain path to file with timestamped name.
    - **initialise_outdir()** : Initialis directory for writing output.
    - **new_rt_roi_observation()** : Create new pydicom.dataset.Dataset object
      representing an RT ROI observation.
    - **new_roi_contour()** : Create new pydicom.dataset.Dataset object
      representing an ROI contour.
    - **set_data_and_time()** : Set current date and time.
    - **set_geometry_and_scaling()** : Add geometry and scaling to dataset.
    - **set_image()** : Add to dataset information for full (3d) image.
    - **set_image_slice()** : Add to dataset information for (2d) image slice.
    - **set_structure_set()** : Add to dataset information for structure set.
    - **update_dataset()** : Make arbitrary updates to dataset.
    - **write()** : Write data in DICOM format.
    """

    def __init__(
        self,
        outdir=None,
        data=None,
        affine=None,
        overwrite=True,
        header_source=None,
        orientation=None,
        patient_id=None,
        patient_position=None,
        modality=None,
        root_uid=None,
        header_extras=None,
        source_type=None,
        outname=None,
    ):
        """
        Create instance of DicomWriter class.

        **Parameters:**

        outdir : str/pathlib.Path, default=None
            Path to directory where data are to be written.

        data : numpy.array/skrt.structures.StructureSet, default=None
            Data to be written in DICOM format.  For source type
            "Dose" or "Image", this should be image-type data, stored
            in a numpy array.  For source type "StructureSet", this
            should be a skrt.structures.StructureSet object.

        affine : 4x4 array, default=None
            Array containing the affine matrix associated with image-type data.

        overwrite : bool, default=True
            If True, delete any pre-existing DICOM files from output
            directory before writing.

        header_source : str/pydicom.dataset.FileDataset/None, default=None
            Source from which to create DICOM header.  This can be:
            (a) a path to a dicom file, which will be used as the header;
            (b) a path to a directory containing dicom files, where
                the first file alphabetically will be used as the header;
            (c) a pydicom.dataset.FileDataset object;
            (d) None, in which case a header will be created from scratch,
                inculding new UIDs.

        orientation : 6-element list, default=None
            Direction cosines indicating image orientation.  If None,
            set to standard DICOM-style orientation, [1, 0, 0, 0, 1, 0],
            such that [column, row, slice] correspond to the axes [x, y, z].

        patient_id : str, default=None
            Patient identifier.

        patient_position : str, default=None
            Indicator of patient position relative to imaging equipment.
            Examples include: "HFS" (head first, supine),
            "FFP" (feet first, prone), "FFS" (feet first, supine).  If None,
            use "HFS".

        modality : str, default=None
            Modality of data collection.  Supported modalities are
            'CT' and 'RTDOSE'.  If None, the modality is taken to be 'CT'.

        root_uid : str, default=None
            Root to be used in Globally Unique Identifiers (GUIDs).  This
            should uniquely identify the institution or group generating
            the GUID.  If None, the value of pydicom.uid.PYDICOM_ROOT_UID
            is used.

            A unique root identifier can be obtained free of charge from
            Medical Connections:
            https://www.medicalconnections.co.uk/FreeUID/

        header_extras : dict, default=None
            Dictionary of attribute-value pairs for applying arbitary
            updates to a pydicom.dataset.FileDataset.

        source_type : str, default=None
            Name of the class for which the DicomWriter instance is
            created.  This affects the information included when a
            pydicom.dataset.FileDataset is created from scratch.  Values of
            source_type for which relevant behaviour is implemented are:

            - 'Dose';
            - 'Image';
            - 'StructureSet'.

        outname : str, default=None
            Name to use for output file for Dose and StructureSet.  If None,
            a name including modality and timestamp is generated.  This
            parameter is disregarded for Images, where names correspond
            to sequential numbering.
        """
        # Set attribute values from input parameters.
        self.outdir = Path(fullpath(str(outdir)))
        self.outname = outname
        self.data = data
        self.affine = affine
        self.overwrite = overwrite
        self.header_source = header_source
        self.orientation = orientation or [1, 0, 0, 0, 1, 0]
        self.patient_id = patient_id or "Unknown"
        self.patient_position = patient_position or "HFS"
        self.modality = modality.upper() if modality is not None else "CT"
        self.header_extras = header_extras or {}
        self.source_type = source_type

        if root_uid:
            self.root_uid = root_uid
        else:
            self.root_uid = pydicom.uid.PYDICOM_ROOT_UID

        # Set date and time.
        self.set_date_and_time()
        # Attempt to retrive dataset from source.
        self.dset = self.get_file_dataset()
        # Create new dataset.
        if self.dset is None:
            self.dset = self.create_file_dataset()
            # Add image-specific information.
            if self.source_type in ["Dose", "Image"]:
                self.add_image_info()
            # Add structure-set-specific information.
            elif self.source_type == "StructureSet":
                self.add_structure_set_info()
        # Set null initial value for ReferenceFrameOfReferenceUID.
        self.referenced_frame_of_reference_uid = None

        # Add information for geometry and scaling to dataset.
        if self.source_type in ["Dose", "Image"]:
            self.set_geometry_and_scaling()

        # Update dataset with values supplied by header_extras.
        self.update_dataset()

        # Store value of MediaStorageSOPInstanceUID.
        # This will be the root UID when saving data for image slices.
        self.media_storage_sop_instance_uid = getattr(
            self.dset.file_meta, "MediaStorageSOPInstanceUID", None
        )

    def add_image_info(self):
        """
        Add image-specfic information to own file dataset.
        """

        self.dset.AcquisitionDate = self.date
        self.dset.AcquisitionTime = self.time
        self.dset.BitsAllocated = 16
        self.dset.BitsStored = 16
        self.dset.HighBit = 15
        self.dset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        self.dset.InstanceNumber = None
        self.dset.KVP = None
        self.dset.PhotometricInterpretation = "MONOCHROME2"
        self.dset.PixelRepresentation = 0
        self.dset.PositionReferenceIndicator = None
        self.dset.SamplesPerPixel = 1
        self.dset.SpecificCharacterSet = "ISO_IR 100"
        self.dset.FrameOfReferenceUID = f"{self.dset.SeriesInstanceUID}.1.1"

    def add_structure_set_info(self):
        """
        Add structure-set-specfic information to own file dataset.
        """
        self.dset.OperatorsName = None
        self.dset.StructureSetDate = self.date
        self.dset.StructureSetLabel = "Unknown"
        self.dset.StructureSetTime = self.time

    def create_file_dataset(self):
        """
        Create new pydicom.dataset.FileDataset object.
        """

        # SOP: Service-Object Pair
        # UID: Unique Identifier
        # Correspondence between SOP Class Name and SOP Class UID given at:
        # http://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
        media_storage_sop_class_uid = self.get_media_storage_sop_class_uid()
        transfer_syntax_uid = getattr(uid, "ExplicitVRLittleEndian")

        # Define metadata
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b"\x00\x01"
        file_meta.MediaStorageSOPClassUID = media_storage_sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = self.generate_shortened_uid()
        file_meta.TransferSyntaxUID = transfer_syntax_uid
        version = pydicom.__version__
        file_meta.ImplementationClassUID = f"{self.root_uid}{version}"
        file_meta.ImplementationVersionName = f"pydicom-{version}"

        # Create FileDataset instance
        filename = None
        preamble = b"\x00" * 128
        dset = FileDataset(filename, {}, file_meta=file_meta, preamble=preamble)
        dset.is_little_endian = transfer_syntax_uid.is_little_endian
        dset.is_implicit_VR = transfer_syntax_uid.is_implicit_VR

        dset.AccessionNumber = None
        dset.ContentDate = self.date
        dset.ContentTime = self.time
        dset.DerivationDescription = None
        dset.InstanceCreationDate = self.date
        dset.InstanceCreationTime = self.time
        dset.InstitutionName = None
        dset.Manufacturer = "Unknown"
        dset.ManufacturerModelName = None
        dset.Modality = self.modality
        dset.PatientID = self.patient_id
        dset.PatientPosition = self.patient_position
        dset.PatientAge = None
        dset.PatientBirthDate = None
        dset.PatientName = self.patient_id
        dset.PatientSex = None
        dset.ReferringPhysicianName = None
        dset.SeriesDate = self.date
        dset.SeriesDescription = None
        dset.SeriesInstanceUID = self.generate_shortened_uid()
        dset.SeriesNumber = None
        dset.SeriesTime = self.time
        dset.SoftwareVersions = f"pydicom-{version}"
        dset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dset.StationName = None
        dset.StudyDate = self.date
        dset.StudyDescription = None
        dset.StudyInstanceUID = self.generate_shortened_uid()
        dset.StudyTime = self.time
        dset.StudyID = None

        return dset

    def generate_shortened_uid(self, max_length=55):
        """
        Return shortened UID.

        UIDs generated by pydicom.uid.generate_uid() are up to 64 characters.
        Some UIDs need to have additional characters appended, for example
        to identify image slices.  This can result in UIDs that are longer
        than is allowed by the DICOM standard.  To avoid this, the initial
        UIDs may be shortened.

        **Parameter:**

        max_length : int, default=-10
            Length to which UID should be shortened.  If the UID initially
            generated is shorter than this, then it's returned unchanged.
        """
        new_uid = generate_uid(self.root_uid)
        return new_uid[: min(len(new_uid), max_length)]

    def get_file_dataset(self):
        """
        Attempt to retrieve pydicom.dataset.FileDataset from source.
        """
        dset = None

        if self.header_source:
            # See if the source is already a FileDataset.
            if isinstance(self.header_source, FileDataset):
                dset = self.header_source
            # See if the source points to a DICOM file.
            else:
                dcm_path = None
                source = Path(fullpath(str(self.header_source)))
                if source.is_file():
                    dcm_path = source
                elif source.is_dir():
                    dcms = list(source.glob("**/*.dcm"))
                    if dcms:
                        dcm_path = dcms[0]
                if dcm_path:
                    try:
                        dset = pydicom.dcmread(str(dcm_path), force=True)
                    except pydicom.errors.InvalidDicomError:
                        pass

        return dset

    def get_media_storage_sop_class_uid(self):
        """
        Map to media storage SOP class UID for selected modalities.

        Note that in general the mapping between modality and
        media storage SOP class UID is one to many.
        """

        storage_classes = {
            "RTDOSE": "RTDoseStorage",
            "RTPLAN": "RTPlanStorage",
            "RTSTRUCT": "RTStructureSetStorage",
            "CT": "CTImageStorage",
            "MR": "MRImageStorage",
            "PT": "PositronEmissionTomographyImageStorage",
        }

        media_storage_sop_class_name = storage_classes.get(self.modality)
        media_storage_sop_class_uid = getattr(
            sop, media_storage_sop_class_name, None
        )

        return media_storage_sop_class_uid

    def get_path_with_timestamp(self, check_interval=0.5):
        """
        Obtain path to file with timestamped name.

        The timestamp is iteratively updated to ensure that any
        existing file isn't overwritten.

        **Parameter:**

        check_interval: float, default=0.5
            Interval in seconds to pause before updating timestamp.
        """
        file_exists = True
        while file_exists:
            outname = f"{self.dset.Modality}_{self.date}_{self.time}.dcm"
            outpath = self.outdir / outname
            file_exists = Path(outpath).exists()
            if file_exists:
                time.sleep(check_interval)
                self.set_date_and_time()

        return outpath

    def initialise_outdir(self):
        """
        Prepare directory where output data are to be written.

        If the output directory doesn't exist then it is created.
        If overwrite is True and the output directory contains DICOM files,
        these are deleted.
        """
        # Create directory if it doesn't exist.
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Delete any pre-existing dicom files.
        if self.overwrite:
            for dcm in self.outdir.glob("**/*.dcm"):
                dcm.unlink()

    def new_roi_contour(self, number, roi, image):
        """
        Create dataset representing an ROI contour.

        **Parameters:**

        number : int
            Integer identifying ROI.

        roi : skrt.structures.ROI
            ROI object for which contour data are to be stored.

        image
        """
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [
            f"{255 * rgb : .0f}" for rgb in roi.color[0:3]
        ]
        roi_contour.ReferencedROINumber = number
        roi_contour.ContourSequence = Sequence()

        for z_point, contours in sorted(roi.get_contours().items()):
            image_dset = (
                image.get_dicom_dataset(pos=z_point)
                if image is not None
                else None
            )
            referenced_sop_class_uid = getattr(image_dset, "SOPClassUID", None)
            referenced_sop_instance_uid = getattr(
                image_dset, "SOPInstanceUID", None
            )

            for xy_points in contours:
                xyz_points = [
                    [x_point, y_point, z_point]
                    for x_point, y_point in xy_points
                ]

                contour = Dataset()
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour_data = [
                    item for sublist in xyz_points for item in sublist
                ]
                contour.ContourData = list(map(str, contour_data))
                contour.NumberOfContourPoints = len(xyz_points)
                contour.ContourImageSequence = Sequence()
                contour_image = Dataset()
                contour_image.ReferencedSOPClassUID = referenced_sop_class_uid
                contour_image.ReferencedSOPInstanceUID = (
                    referenced_sop_instance_uid
                )
                contour.ContourImageSequence.append(contour_image)
                roi_contour.ContourSequence.append(contour)

        return roi_contour

    def new_rt_roi_observation(self, number, name):
        """
        Create dataset representing an RT ROI observation.

        **Parameters:**

        number

        name
        """
        rt_roi_observation = Dataset()
        rt_roi_observation.ObservationNumber = number
        rt_roi_observation.ReferencedROINumber = number
        rt_roi_observation.ROIObservationLabel = name
        rt_roi_observation.RTROIInterpretedType = None
        rt_roi_observation.ROIInterpreter = None
        return rt_roi_observation

    def set_date_and_time(self):
        """
        Obtain current time and store formatted values for date and time.
        """

        # Set creation date/time
        now = datetime.datetime.now()
        self.date = now.strftime("%Y%m%d")
        self.time = now.strftime("%H%M%S")

    def set_geometry_and_scaling(self):
        """
        Add to dataset information relating to geometry and scaling.
        """

        # Set voxel sizes and other geometrical information from affine matrix.
        self.dset.ImageOrientationPatient = self.orientation
        self.dset.PixelSpacing = [self.affine[0, 0], self.affine[1, 1]]
        self.dset.SliceThickness = self.affine[2, 2]
        self.dset.ImagePositionPatient = list(self.affine[:-1, 3])
        self.dset.ImagePositionPatient[2] += (
            self.data.shape[2] - 1
        ) * self.affine[2, 2]
        self.dset.SliceLocation = self.affine[2, 3]
        self.dset.Rows = self.data.shape[0]
        self.dset.Columns = self.data.shape[1]
        self.dset.RescaleIntercept = min(
            getattr(self.dset, "RescaleIntercept", 0), np.min(self.data), 0
        )

        # Set attributes specific to 'Image' type.
        if self.source_type == "Image":
            self.dset.ImagesInAcquisition = self.data.shape[2]
            intensity_range = self.data.max() - self.data.min()
            if intensity_range < 1000:
                default_rescale_slope = intensity_range / 1000
            else:
                default_rescale_slope = 1
            self.dset.RescaleSlope = getattr(
                self.dset, "RescaleSlope", default_rescale_slope
            )

        # Set attributes specific to 'Dose' type.
        elif self.source_type == "Dose":
            self.dset.DoseGridScaling = getattr(
                self.dset, "DoseGridScaling", 0.001
            )
            self.dset.ImagesInAcquisition = 1
            self.dset.NumberOfFrames = self.data.shape[2]
            self.dset.FrameIncrementPointer = (0x3004, 0x000C)
            self.dset.DoseUnits = "GY"
            self.dset.DoseType = "PHYSICAL"
            self.dset.DoseComment = None
            self.dset.DoseSummationType = None
            self.dset.GridFrameOffsetVector = []
            for idx in range(self.data.shape[2]):
                self.dset.GridFrameOffsetVector.append(-idx * self.affine[2, 2])

    def set_image(self):
        """
        Add to dataset information for full (3d) image.
        """
        # Rescale parameters
        slope = getattr(self.dset, "RescaleSlope", 1)
        slope = getattr(self.dset, "DoseGridScaling", slope)
        intercept = getattr(self.dset, "RescaleIntercept", 0)
        pixel_array = self.data.copy()
        pixel_array = (pixel_array - intercept) / slope
        pixel_array = np.rint(pixel_array).astype(np.uint16)
        pixel_array = pixel_array[:, :, ::-1].transpose(2, 0, 1)
        if not uid.UID(self.dset.file_meta.TransferSyntaxUID).is_little_endian:
            pixel_array = pixel_array.byteswap()
        self.dset.PixelData = pixel_array.tobytes()

    def set_image_slice(self, idx=0):
        """
        Add to dataset information for (2d) image slice.

        **Parameters:**

        idx : int, default=0
            Slice array index.
        """

        # Rescale parameters
        slope = getattr(self.dset, "RescaleSlope", 1)
        slope = getattr(self.dset, "DoseGridScaling", slope)
        intercept = getattr(self.dset, "RescaleIntercept", 0)
        slice_index = self.data.shape[2] - idx
        pos = self.affine[2, 3] + idx * self.affine[2, 2]
        xy_slice = self.data[:, :, idx].copy()
        xy_slice = (xy_slice - intercept) / slope
        xy_slice = xy_slice.astype(np.uint16)
        if not uid.UID(self.dset.file_meta.TransferSyntaxUID).is_little_endian:
            xy_slice = xy_slice.byteswap()
        self.dset.PixelData = xy_slice.tobytes()
        self.dset.SliceLocation = pos
        self.dset.ImagePositionPatient[2] = pos
        self.dset.InstanceNumber = slice_index
        self.dset.file_meta.MediaStorageSOPInstanceUID = (
            f"{self.media_storage_sop_instance_uid}.{slice_index}"
        )
        self.dset.SOPInstanceUID = (
            self.dset.file_meta.MediaStorageSOPInstanceUID
        )

    def set_referenced_frame_of_reference_sequence(self, image):
        """
        Set attributes for referenced frame of reference sequence.

        **Parameter:**

        image : skrt.image.Image
            Image from which attributes of referenced frame
            of reference are to be determined.  If no associated DICOM
            dataset can be loaded for this image, then
            self.dset.ReferencedFrameOfReferenceSequence is set as
            an empty sequence.
        """
        # Try to determine referenced frame of reference UID from image.
        dset = image.get_dicom_dataset() if image is not None else None
        self.referenced_frame_of_reference_uid = getattr(
            dset, "FrameOfReferenceUID", None
        )

        # Create new sequence if not already defined from header source,
        # or if existing sequence is to be overwritten from input object.
        if self.referenced_frame_of_reference_uid is not None or not hasattr(
            self.dset, "ReferencedFrameOfReferenceSequence"
        ):
            self.dset.ReferencedFrameOfReferenceSequence = Sequence()

        # Exit if unable to determine referenced frame of reference UID.
        if self.referenced_frame_of_reference_uid is None:
            return

        # Create dataset for referenced frame of reference.
        referenced_frame_of_reference = Dataset()
        referenced_frame_of_reference.FrameOfReferenceUID = (
            self.referenced_frame_of_reference_uid
        )
        referenced_frame_of_reference.RTReferencedStudySequence = Sequence()

        # Create dataset for rt referenced study.
        rt_referenced_study = Dataset()
        rt_referenced_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.2"
        rt_referenced_study.ReferencedSOPInstanceUID = getattr(
            dset, "StudyInstanceUID", None
        )
        rt_referenced_study.RTReferencedSeriesSequence = Sequence()

        # Create dataset for rt referenced series.
        rt_referenced_series = Dataset()
        rt_referenced_series.SeriesInstanceUID = getattr(
            dset, "SeriesInstanceUID", None
        )
        rt_referenced_series.ContourImageSequence = Sequence()

        # Create the contour-image sequence.
        for idx in range(image.get_n_voxels()[2]):
            dset = image.get_dicom_dataset(idx + 1)
            contour_image = Dataset()
            contour_image.ReferencedSOPClassUID = getattr(
                dset, "SOPClassUID", None
            )
            contour_image.ReferencedSOPInstanceUID = getattr(
                dset, "SOPInstanceUID", None
            )
            rt_referenced_series.ContourImageSequence.append(contour_image)

        # Append completed datasets to sequences.
        rt_referenced_study.RTReferencedSeriesSequence.append(
            rt_referenced_series
        )
        referenced_frame_of_reference.RTReferencedStudySequence.append(
            rt_referenced_study
        )
        self.dset.ReferencedFrameOfReferenceSequence.append(
            referenced_frame_of_reference
        )

    def set_structure_set(self):
        """
        Add to dataset information for structure set.

        The contour data saved is all that's needed for scikit-rt.
        Several DICOM sequences are left empty, or partially populated,
        which outside of scikit-rt may sometimes cause problems.
        """
        # Obtain reference to any image associated with structure set.
        image = self.data.get_image()

        # Initialise sequences.
        self.set_referenced_frame_of_reference_sequence(image)
        self.dset.StructureSetROISequence = Sequence()
        self.dset.RTROIObservationsSequence = Sequence()
        self.dset.ROIContourSequence = Sequence()
        rois = self.data.get_rois()
        numbers = [roi.number for roi in rois if roi.number is not None]
        index = max(numbers or [0])

        for name, roi in sorted(self.data.get_roi_dict().items()):
            if roi.number is not None:
                number = roi.number
            else:
                index += 1
                number = index

            structure_set_roi = Dataset()
            structure_set_roi.ROINumber = number
            structure_set_roi.ROIName = name
            structure_set_roi.ReferencedFrameOfReferenceUID = (
                self.referenced_frame_of_reference_uid
            )
            structure_set_roi.ROIGenerationAlgorithm = None
            self.dset.StructureSetROISequence.append(structure_set_roi)
            self.dset.RTROIObservationsSequence.append(
                self.new_rt_roi_observation(number, name)
            )
            self.dset.ROIContourSequence.append(
                self.new_roi_contour(number, roi, image)
            )

    def update_dataset(self):
        """
        Make arbitary updates to dataset, based on self.header_extras.
        """

        for attribute, value in self.header_extras.items():
            setattr(self.dset, attribute.replace(" ", ""), value)

    def write(self):
        """
        Write data in DICOM format.
        """

        # Prepare output directory for writing.
        self.initialise_outdir()

        # Write image as single slice per file.
        # Write dose as single file for all slices.
        if self.source_type == "Dose":
            # Write single file.
            self.set_image()
            if self.outname is None:
                outpath = self.get_path_with_timestamp()
            else:
                outpath = str(self.outdir / self.outname)
            self.dset.save_as(outpath, write_like_original=False)

        elif self.source_type == "Image":
            # Write file per slice.
            for i in range(self.data.shape[2]):
                self.set_image_slice(i)
                outname = f"{self.dset.InstanceNumber}.dcm"
                outpath = self.outdir / outname
                self.dset.save_as(outpath, write_like_original=False)

        elif self.source_type == "StructureSet":
            self.set_structure_set()
            if self.outname is None:
                outpath = self.get_path_with_timestamp()
            else:
                outpath = str(self.outdir / self.outname)
            self.dset.save_as(outpath, write_like_original=False)

        return self.dset
