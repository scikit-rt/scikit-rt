'''VoxTox extenstions to Scikit-rt for ROIs and StructureSets'''

import glob
import os
import shutil
import statistics

import numpy as np

import skrt

class ROI(skrt.structures.ROI):
    '''VoxTox-specific extensions to Scikit-rt ROI class.'''

    def load(self, force=False):
        '''
        Load ROI from file or source.

        When loading an ROI from a point cloud, the default precision with
        which z-coordinates are matched to z-planes is key_value=0.1.
        If a different value is wanted, this should be specified when
        creating the ROI instance.

        For details of loading from sources other than point-cloud files,
        see documnetation for skrt.structures.ROI.load().

        **Parameter:**

        force : bool, default=False
            If True, force loading, even if ROI data already loaded.
        '''
        if self.loaded and not force:
            return

        # Ensure that key_precision and ext are definedi, removing
        # if necessary from self.kwargs, which will be used to initialise
        # an Image instance.
        if not hasattr(self, 'key_precision'):
            self.key_precision = 0.1
        self.key_precision = self.kwargs.pop(
                'key_precision', self.key_precision)

        if isinstance(self.source, str):
            if self.source.endswith('.txt'):

                with open(self.source, "r", encoding='ascii') as in_file:
                    lines = in_file.readlines()

                # Extract dictionary of contours from point-cloud data.
                contours_3d = {}
                for line in lines:
                    i_point, j_point, k_point = [
                            float(value) for value in line.split()]
                    k_point = self.image.shape[2] - k_point
                    x_point = self.image.idx_to_pos(i_point, 'x')
                    y_point = self.image.idx_to_pos(j_point, 'y')
                    z_point = self.image.idx_to_pos(k_point, 'z')

                    key = get_key(z_point, contours_3d, self.key_precision)
                    if not key in contours_3d:
                        contours_3d[key] = []
                    contours_3d[key].append((x_point, y_point, z_point))

                contours = contours_mean_z(contours_3d)
                self.__init__(contours, self.name, self.color, self.load,
                        self.image, self.shape, self.affine, self.voxel_size,
                        self.origin, self.mask_threshold,
                        self.default_geom_method, self.overlap_level,
                        **self.kwargs)

        if not self.loaded:
            skrt.structures.ROI.load(self)

    def write(self, outname=None, outdir=".", ext=None, **kwargs):
        '''
        Write ROI data.

        **Parameters:**

        outdir : str, default='.'
            Directory where point-cloud file is to be written.

        outname : str, default=None
            Name to be used for the point-cloud file.  If not already
            present, the extension defined by ext, or '.txt' if ext is None,
            is appended.

        ext : str, default=None
            Extension to be added to the name of the output file.

        kwargs : dict, default={}
            Dictionary containing arbitrary parameter-value pairs
            passed in the function call:

            - point_cloud : bool, default=False
                  If True, write ROI data in VoxTox point-cloud format.
                  Each file contains contour data for a single ROI.
                  Each line gives (column, row, slice) coordinates
                  for a single point.  Points are ordered sequentially
                  around a slice.  Slice numbers are inverted with respect
                  to the usual DICOM numbering.

            - all other values are passed to the contstructor of the
              associated skrt.image.Image object.  See skrt.image.Image
              documentation for details of available parameters.
        '''
        self.load()

        # When point-cloud format not requested,
        # format using write() method of base class.
        point_cloud = kwargs.get('point_cloud', False)
        if not point_cloud:
            skrt.structures.ROI.write(self, outname, outdir, ext, **kwargs)
            return

        # Define path to output file.
        if outname is None:
            outname = self.name
        if ext is None:
            ext = '.txt'
        if not ext.startswith('.'):
            ext = f'.{ext}'
        if not outname.endswith(ext):
            outname = f'{self.name}{ext}'
        point_cloud_path = f'{outdir}/{outname}'

        # Extract contour points.
        lines = []
        for z_point, contours in self.get_contours().items():
            k_point = self.shape[2] - self.pos_to_idx(
                    z_point, 'z', return_int=True)
            for contour in contours:
                for x_point, y_point in contour:
                    i_point = self.pos_to_idx(x_point, 'x', return_int=False)
                    j_point = self.pos_to_idx(y_point, 'y', return_int=False)
                    lines.append(f'{i_point:.3f} {j_point:.3f} {k_point}')

        # Write point-cloud file
        with open(point_cloud_path, "w", encoding='ascii') as out_file:
            out_file.write('\n'.join(lines))


class StructureSet(skrt.structures.StructureSet):
    '''VoxTox-specific extensions to Scikit-rt StructureSet class.'''

    def load(self, sources=None, force=False):

        '''
        Load structure set form sources.  If None, load from self.sources.

        When loading a structure set from point clouds, the default
        precision with which z-coordinates are matched to z-planes
        is key_value=0.1.  If a different value is wanted, this should
        be specified when creating the StructureSet instance.

        For details of loading from sources other than point-cloud files,
        see documnetation for skrt.structures.StructureSet.load().

        **Parameters:**

        sources : Various, default=None
            Sources from which to load structure set.  If the sources are
            files with the extension '.txt', they are treated as being in
            VoxTox point-cloud format.  The default precision with which
            z-coordinates are matched to z-planes is key_value=0.1.
            If a different value is wanted, this should be specified when
            creating the StructureSet instance.

        force : bool, default=False
            If True, force loading, even if StructureSet data already loaded.
        '''

        if self.loaded and not force and sources is None:
            return

        if sources is None:
            sources = self.sources

        if not skrt.core.is_list(sources):
            sources = [sources]

        # Check for point-cloud files (.txt suffix).
        sources_expanded = []
        for source in sources:
            if isinstance(source, str):
                if os.path.isdir(source):
                    sources_expanded.extend(glob.glob(f'{source}/*.txt'))
                elif source.endswith('.txt'):
                    sources_expanded.extend(glob.glob(f'{source}'))

        # Load structure set from point-cloud files.
        if sources_expanded:
            for source in sources_expanded:
                roi = ROI(source=source, image=self.image, **self.roi_kwargs)
                self.rois.append(roi)
            self.loaded = True

        # Use load method of base class.
        if not self.loaded:
            skrt.structures.StructureSet(self)

    def write(self, outname=None, outdir=".", ext=None, overwrite=False,
            **kwargs):
        '''
        Write ROI data.

        **Parameters:**

        outdir : str, default='.'
            Directory where point-cloud files are to be written.

        outname : str, default=None
            Not used when writing point-cloud files.  The names of
            output point-cloud files match the corresponding ROI names.

        ext : str, default=None
            Not used when writing point-cloud files.  The filename
            extension for point-cloud files is '.txt'.

        overwrite: bool, default=True
            If True, overwrite any pre-existing point files.

        kwargs : dict, default={}
            Dictionary containing arbitrary parameter-value pairs
            passed in the function call:

            - names : dict, default={}
                  Dictionary mapping between possible ROI names within
                  a structure set (dictionary values) and names to be used
                  for point-cloud files (dictionary keys).  The possible names
                  names can contain wildcards with the '*' symbol.

            - point_cloud : bool, default=False
                  If True, write ROI data in VoxTox point-cloud format.
                  Each file contains contour data for a single ROI.
                  Each line gives (column, row, slice) coordinates
                  for a single point.  Points are ordered sequentially
                  around a slice.  Slice numbers are inverted with respect
                  to the usual DICOM numbering.

            - all other values are passed to the contstructors of ROI
              objects loaded as part of the structure set.  See
              skrt.structures.ROI documentation for details of available
              parameters.
        '''

        # When point-cloud format not requested,
        # format using write() method of base class.
        point_cloud = kwargs.get('point_cloud', False)
        if not point_cloud:
            skrt.structures.StructureSet.write(self, outname, outdir, ext,
                    **kwargs)
            return

        # Create temporary StructureSet, with ROIs filtered/renamed.
        names = self.roi_kwargs.pop('names', None)
        ss_tmp = self.filtered_copy(names=names, keep_renamed_only=True)

        # Ensure output directory exists.
        if os.path.exists(outdir):
            if overwrite:
                shutil.rmtree(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Write point clouds.
        for skrt_roi in ss_tmp.get_rois():
            voxtox_roi = ROI(skrt_roi)
            voxtox_roi.write(outdir=outdir, **kwargs)

def contours_mean_z(contours_3d={}):
    '''
    Recalculate point cloud, averaging over z-coordinates in a plane

    **Parameter:**

    point_cloud : dict, default={}
        Dictionary of point-cloud data, where the keys are plane
        z-coordinates, as strings, and the value for a given key
        is a list of (x, y, z) point coordinates.
    '''

    contours = {}
    for points in contours_3d.values():
        xy_points = []
        z_points = []

        for point in points:
            x_point, y_point, z_point = point
            xy_points.append([x_point, y_point])
            z_points.append(z_point)

        z_mean = statistics.mean(z_points)
        key = f'{z_mean:.2f}'
        if not key in contours:
            contours[key] = []
        contours[key].append(np.array(xy_points))

    contours2 = {}
    for key, value in contours.items():
        contours2[float(key)] = value

    return contours2

def get_key(z_point=0, contours_3d={}, key_precision=0.1):
    '''
    Determine z-plane key corresponding to given z-coordinate.

    **Parameters:**

    z_point : float, default=0
        Z-coordinate of a contour point.

    contours_3d : dict, default={}
        Dictionary of contour-point coordinates, stored by z-plane.

    key_precision : float, default=0.1
        Precision with which z-coordinate should match z-plane.
        As a result of rounding errors, and/or small translations
        when applying registration transforms, z-coordinates for points
        in the same plane may not be identical.
    '''

    key_string = f'{z_point:.2f}'
    for key in contours_3d:
        z_key = float(key)
        if abs(z_point - z_key) < key_precision:
            key_string = key
            break

    return key_string
