'''Core functions specific to analysis of VoxTox data'''

import pydicom

couch_shifts_group = 0x0099
rotations_element = 0x1012
translations_element = 0x1011

def get_couch_rotations(
        ds=None, group=couch_shifts_group, element=rotations_element):
    '''
    Retreive couch rotations stored as DICOM private data.

    Couch rotations are stored and retrieved as:
        pitch (about x-axis), yaw (about y-axis), roll (about z-axis)
    These represent the second part of the transformation
    (translations + rotations) for mapping from the guidance scan
    to the planning scan.  In practice, pitch and yaw are always zero.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset/None, default=None
        Dataset containing couch shifts stored as private data.
    group : int, default=couch_shits_group
        Location of the private group where couch shifts are stored.
    element: int, default=rotations_element
        Location of the private element, within the private group,
        where couch rotations are stored (default corresponds to VoxTox).
    '''
    return unpack_couch_shifts(ds, group, element)

def get_couch_shifts(im=None):
    '''
    Retreive couch shifts for a given CT guidance scan.

    In VoxTox, couch shifts for an MV CT guidance scan define the adjustments
    applied by a radiographer after examining the scan, to match the position
    at planning time as closely as possible.

    Couch shifts are retrieved as:
        translations: dx, dy, dz
        rotations: pitch (about x-axis), yaw (about y-axis), roll (about z-axis)
    These represent the transformation for mapping from the guidance scan
    to the planning scan.  In practice, pitch and yaw are always zero.

    Parameter
    ---------
    im : skrt.image.Image/None, default=None
        Image object for which couch shifts are to be retrieved.
    '''

    try:
        ds = im.get_dicom_dataset()
    except AttributeError:
        ds = None

    rotations = get_couch_rotations(ds)
    translations = get_couch_translations(ds)

    return (translations, rotations)

def get_couch_translations(
        ds=None, group=couch_shifts_group, element=translations_element):
    '''
    Retreive couch translations stored as DICOM private data.

    For historical reasons, couch translations are stored as:
        +dx, -dz, +dy
    They are rearranged here as:
        +dx, +dy, +dz
    These represent the first part of the transformation
    (translations + rotations) for mapping from the guidance scan
    to the planning scan.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset/None, default=None
        Dataset containing couch shifts stored as private data.
    group : int, default=couch_shifts_group
        Location of the private group where couch shifts are stored.
    element: int, default=translations_element
        Location of the private element, within the private group,
        where couch translations are stored (default corresponds to VoxTox).
    '''
    dx, dz_bar, dy = unpack_couch_shifts(ds, group, element)
    return (dx, dy, -dz_bar)

def unpack_couch_shifts(ds=None, group=None, element=None, n_shift=3):
    '''
    Unpack couch shifts stored as DICOM private data.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset/None, default=None
        Dataset containing couch shifts stored as private data.
    group : int/None, default=None
        Location of the private group where couch shifts are stored.
    element: int/None, default=None
        Location of the private element, within the private group,
        where couch shifts are stored.
    n_shift: int, default = 3
        Number of shifts stored in an element (default corresponds to VoxTox).
    '''

    shifts_ok = False

    # Extract element value
    if type(ds) == pydicom.dataset.FileDataset:
        try:
            element_value = ds[group, element].value
        except KeyError:
            element_value = None

    if element_value is not None:
        # Extract list of shifts from element value
        if isinstance(element_value, str):
            shifts = element_value.split('\\')
        elif isinstance(element_value, bytes):
            shifts = element_value.decode().split('\\')
        else:
            shifts = list(element_value)

        # Check that number of shifts is as expected,
        # and convert to floats
        if n_shift == len(shifts):
            shifts_ok = True
            for i in range(n_shift):
                try:
                    shifts[i] = float(shifts[i])
                except ValueError:
                    shifts_ok = False
                    break
    if not shifts_ok:
        shifts = [None, None, None]

    return tuple(shifts)
