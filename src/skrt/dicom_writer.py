'''
Class DicomWriter for writing data in DICOM format.

DICOM writing is supported for instances of:

- skrt.image.Image;
- skrt.dose.Dose.
'''

import datetime
from pathlib import Path
import random

import numpy as np
import pydicom
from pydicom import uid
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
import pydicom._storage_sopclass_uids as sop

from skrt.core import fullpath

class DicomWriter:
    '''
    Class to help with writing data in DICOM format.

    DICOM writing is supported for instances of:

    - skrt.image.Image;
    - skrt.dose.Dose.

    ** Methods:**
    - **_Init__()** : Create instance of DicomWriter class.
    - **add_to_image_dataset()** : Add to dataset image-specific information.
    - **create_file_dataset()** : Create new pydicom.dataset.FileDataset object.
    - **get_file_dataset()** : Retrive pydicom.dataset.FileDataset from source.
    - **get_media_storage_sop_class_uid()** : Determine uid given modality.
    - **initialise_outdir()** : Initialis directory for writing output.
    - **set_data_and_time()** : Set current date and time.
    - **set_geometry_and_scaling()** : Add geometry and scaling to dataset.
    - **set_image()** : Add to dataset information for full (3d) image.
    - **set_image_slice()** : Add to dataset information for (2d) image slice.
    - **update_dataset()** : Make arbitrary updates to dataset.
    - **write()** : Write data in DICOM format.
    '''

    def __init__(self, outdir=None, data=None, affine=None, header_source=None,
            orientation=None, patient_id=None, modality=None, root_uid=None,
            header_extras={}, source_type=None):
        '''
        Create instance of DicomWriter class.

        **Parameters:**

        outdir : str/pathlib.Path, default=None
            Path to directory where data are to be written.

        data : numpy.array, default=None
            Data to be written in DICOM format.  This should be image-type
            data, stored in a numpy array.

        affine : 4x4 array, default=None
            Array containing the affine matrix associated with image-type data.

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

        header_extras :
            Dictionary of attribute-value pairs for applying arbitary
            updates to a pydicom.dataset.FileDataset.

        source_type : str, default=None
            Name of the class for which the DicomWriter instance is
            created.  This affects the information included when a
            pydicom.dataset.FileDataset is created from scratch.  Values of
            source_type for which relevant behaviour is implemented are:
            
            - 'Dose';
            - 'Image'.
        '''
        # Set attribute values from input parameters.
        self.outdir = Path(fullpath(str(outdir)))
        self.data = data
        self.affine = affine
        self.header_source = header_source
        self.orientation = orientation if orientation is not None \
                else [1, 0, 0, 0, 1, 0]
        self.patient_id = patient_id
        self.modality = modality if modality is not None else 'CT'
        self.header_extras = header_extras
        self.source_type = source_type

        if root_uid:
            self.root_uid = root_uid
        else:
            self.root_uid = pydicom.uid.PYDICOM_ROOT_UID

        # Set date and time.
        self.set_date_and_time()
        # Attempt to retrive dataset from source.
        self.ds = self.get_file_dataset()
        # Create new dataset.
        if self.ds is None:
            self.ds = self.create_file_dataset()

        # Add information for geometry and scaling to dataset.
        if self.source_type in ['Dose', 'Image']:
            self.set_geometry_and_scaling()

        # Update dataset with values supplied by header_extras.
        self.update_dataset()

        # Store value of MediaStorageSOPInstanceUID.
        # This will be the root UID when saving data for image slices.
        self.mediaStorageSOPInstanceUID = (
                self.ds.file_meta.MediaStorageSOPInstanceUID)

    def add_to_image_dataset(self, ds):
        '''
        Add image-specfic information to pydicom.dataset.FileDataset.

        **Parameters:**

        ds : pydicom.dataset.FileDataset
            FileDataset to which data are to be added.
        '''

        ds.AcquisitionDate = self.date
        ds.AcquisitionTime = self.time
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.InstanceCreationDate = self.date
        ds.InstanceCreationTime = self.time
        ds.InstanceNumber = None
        ds.KVP = None
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelRepresentation = 0
        ds.PositionReferenceIndicator = None
        ds.SamplesPerPixel = 1
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.FrameOfReferenceUID = f'{ds.SeriesInstanceUID}.1.1'

        return ds

    def create_file_dataset(self):
        '''
        Create new pydicom.dataset.FileDataset object.
        '''

        # SOP: Service-Object Pair
        # UID: Unique Identifier
        # Correspondence between SOP Class Name and SOP Class UID given at:
        # http://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
        media_storage_sop_class_uid = self.get_media_storage_sop_class_uid()
        transfer_syntax_uid = getattr(uid, 'ExplicitVRLittleEndian')

        # Define metadata
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = media_storage_sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid(self.root_uid)
        file_meta.TransferSyntaxUID = transfer_syntax_uid
        version = pydicom.__version__
        file_meta.ImplementationClassUID = f'{self.root_uid}.{version}'
        file_meta.ImplementationVersionName = f'pydicom-{version}'

        # Create FileDataset instance
        filename = None
        preamble = b'\x00' * 128
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=preamble)
        ds.is_little_endian = transfer_syntax_uid.is_little_endian
        ds.is_implicit_VR = transfer_syntax_uid.is_implicit_VR

        ds.AccessionNumber = None
        ds.ContentDate = self.date
        ds.ContentTime = self.time
        ds.DerivationDescription = None
        ds.InstitutionName = None
        ds.Manufacturer = None
        ds.ManufacturerModelName = None
        ds.Modality = self.modality
        ds.PatientID = self.patient_id
        ds.PatientPosition = None
        ds.PatientAge = None
        ds.PatientBirthDate = None
        ds.PatientName = None
        ds.PatientSex = None
        ds.ReferringPhysicianName = None
        ds.SeriesDate = self.date
        ds.SeriesDescription = None
        ds.SeriesInstanceUID = generate_uid(self.root_uid)
        ds.SeriesNumber = None
        ds.SeriesTime = self.time
        ds.SoftwareVersions = None
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StationName = None
        ds.StudyDate = self.date
        ds.StudyDescription = None
        ds.StudyInstanceUID = generate_uid(self.root_uid)
        ds.StudyTime = self.time
        ds.StudyID = None

        if self.source_type in ['Dose', 'Image']:
            ds = self.add_to_image_dataset(ds)

        return ds

    def get_file_dataset(self):
        '''
        Attempt to retrieve pydicom.dataset.FileDataset from source.
        '''
        ds = None

        if self.header_source:
            # See if the source is already a FileDataset.
            if isinstance(self.header_source, FileDataset):
                ds = self.header_source
            # See if the source points to a DICOM file.
            else:
                dcm_path = None
                source = Path(fullpath(str(self.header_source)))
                if source.is_file():
                    dcm_path = source
                elif source.is_dir():
                    dcms = list(source.glob('**/*.dcm'))
                    if dcms:
                        dcm_path = dcms[0]
                if dcm_path:
                    try:
                        ds = pydicom.dcmread(str(dcm_path), force=True)
                    except pydicom.errors.InvalidDicomError:
                        pass

        return ds

    def get_media_storage_sop_class_uid(self):
        '''
        Map to media storage SOP class UID for selected modalities.

        Note that in general the mapping between modality and
        media storage SOP class UID is one to many.
        '''

        storage_classes = {
                'RTDOSE' : 'RTDose',
                'RTPLAN' : 'RTPlan',
                'RTSTRUCT' : 'RTStructureSet',
                'CT' : 'CTImageStorage',
                'MR' : 'MRImageStorage',
                }

        media_storage_sop_class_name = storage_classes.get(self.modality)
        media_storage_sop_class_uid = getattr(
                sop, media_storage_sop_class_name, None)

        return media_storage_sop_class_uid

    def initialise_outdir(self):
        '''
        Prepare directory where output data are to be written.

        If the output directory doesn't exist then it is created.i
        If the output directory contains DICOM files, these are deleted.
        '''
        # Create directory if it doesn't exist.
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Delete any pre-existing dicom files.
        for dcm in self.outdir.glob('**/*.dcm'):
            dcm.unlink()

    def set_date_and_time(self):
        '''
        Obtain current time and store formatted values for date and time.
        '''

        # Set creation date/time
        now = datetime.datetime.now()
        self.date = now.strftime("%Y%m%d")
        self.time = now.strftime("%H%M%S")

    def set_geometry_and_scaling(self):
        '''
        Add to dataset information relating to geometry and scaling.
        '''

        # Set voxel sizes and other geometrical information from affine matrix.
        self.ds.ImageOrientationPatient = self.orientation
        self.ds.PixelSpacing = [self.affine[0, 0], self.affine[1, 1]]
        self.ds.SliceThickness = self.affine[2, 2]
        self.ds.ImagePositionPatient = list(self.affine[:-1, 3])
        self.ds.ImagePositionPatient[2] += (
                (self.data.shape[2] - 1) * self.affine[2, 2])
        self.ds.SliceLocation = list(self.ds.ImagePositionPatient)
        self.ds.Rows = self.data.shape[0]
        self.ds.Columns = self.data.shape[1]
        self.ds.RescaleIntercept = min(
                getattr(self.ds, 'RescaleIntercept', 0), np.min(self.data), 0)

        # Set attributes specific to 'Image' type.
        if self.source_type == 'Image':
            self.ds.ImagesInAcquisition = self.data.shape[2]
            self.ds.RescaleSlope = getattr(self.ds, 'RescaleSlope', 1)

        # Set attributes specific to 'Dose' type.
        elif self.source_type == 'Dose':
            self.ds.DoseGridScaling = getattr(self.ds, 'DoseGridScaling', 0.001)
            self.ds.ImagesInAcquisition = 1
            self.ds.NumberOfFrames = self.data.shape[2]
            self.ds.FrameIncrementPointer = (0x3004, 0x000c)
            self.ds.DoseUnits = 'GY'
            self.ds.DoseType = 'PHYSICAL'
            self.ds.DoseComment = None
            self.ds.DoseSummationType = None
            self.ds.GridFrameOffsetVector = []
            for idx in range(self.data.shape[2]):
                self.ds.GridFrameOffsetVector.append(-idx * self.affine[2,2])

    def set_image(self):
        '''
        Add to dataset information for full (3d) image.
        '''
        # Rescale parameters
        slope = getattr(self.ds, 'RescaleSlope', 1)
        slope = getattr(self.ds, 'DoseGridScaling', slope)
        intercept = getattr(self.ds, 'RescaleIntercept', 0)
        pixel_array = self.data.copy()
        pixel_array = (pixel_array - intercept) / slope
        pixel_array = np.rint(pixel_array).astype(np.uint16)
        pixel_array = pixel_array[:, :, :: -1].transpose(2, 0, 1)
        self.ds.PixelData = pixel_array.tobytes()

    def set_image_slice(self, idx=0):
        '''
        Add to dataset information for (2d) image slice.

        **Parameters:**

        idx : int, default=0
            Slice array index. 
        '''

        # Rescale parameters
        slope = getattr(self.ds, 'RescaleSlope', 1)
        slope = getattr(self.ds, 'DoseGridScaling', slope)
        intercept = getattr(self.ds, 'RescaleIntercept', 0)
        sl = self.data.shape[2] - idx
        pos = self.affine[2, 3] + idx * self.affine[2, 2]
        xy_slice = self.data[:, :, idx].copy()
        xy_slice = (xy_slice - intercept) / slope
        xy_slice = xy_slice.astype(np.uint16)
        self.ds.PixelData = xy_slice.tobytes()
        self.ds.SliceLocation = pos
        self.ds.ImagePositionPatient[2] = pos
        self.ds.InstanceNumber = sl
        self.ds.file_meta.MediaStorageSOPInstanceUID = (
                f'{self.mediaStorageSOPInstanceUID}.{sl}')

    def update_dataset(self):
        '''
        Make arbitary updates to dataset, based on self.header_extras.
        '''

        for attribute, value in self.header_extras.items():
            setattr(self.ds, attribute, value)

    def write(self):
        '''
        Write data in DICOM format.
        '''

        # Prepare output directory for writing.
        self.initialise_outdir()
        

        # Write image as single slice per file.
        if self.source_type == 'Image':
            # Obtain rescale parameters.
            slope = getattr(self.ds, 'RescaleSlope', 1)
            intercept = getattr(self.ds, 'RescaleIntercept', 0)
            # Write file per slice.
            for i in range(self.data.shape[2]):
                self.set_image_slice(i)
                outname = f'{self.ds.InstanceNumber}.dcm'
                outpath = self.outdir / outname
                self.ds.save_as(outpath)

        # Write dose as single file for all slices.
        elif self.source_type == 'Dose':
            # Obtain rescale parameters.
            slope = getattr(self.ds, 'DoseGridScaling', 1)
            intercept = getattr(self.ds, 'RescaleIntercept', 0)
            # Write single file.
            self.set_image()
            outname = f'{self.ds.Modality}_{self.date}_{self.time}.dcm'
            outpath = self.outdir / outname
            self.ds.save_as(outpath)

        return self.ds
