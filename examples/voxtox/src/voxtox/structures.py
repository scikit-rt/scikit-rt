'''VoxTox extenstions to Scikit-rt for ROIs and StructureSets'''

import glob
import os
import shutil
import statistics

import numpy as np

import skrt

class ROI(skrt.structures.ROI):
    '''VoxTox-specific extensions to Scikit-rt ROI class.'''

    def load_point_cloud(self, path='', image=None, key_precision=0.1,
            roi_name=''):
        '''
        Load data in VoxTox point-cloud format, and convert to contours.

        **Parameters:**

        path : str, default=''
            Path to point-cloud file.

        image : skrt.image.Image/str/None, default=None
            Associated image from which to extract shape and affine matrix.
            Can either be an existing Image object, or a path from which
            an Image can be created.

        roi_name : str, default=''
            Name to be assigned to ROI defined from point-cloud file.
            If roi_name is an empty string, the filename toot is used.

        key_precision : float, default=0.1
            Precision with which z-coordinate should match z-plane.
            As a result of rounding errors, and/or small translations
            when applying registration transforms, z-coordinates for points
            in the same plane may not be identical.
        '''
        if roi_name:
            self.name = roi_name
        else:
            self.name = os.path.splitext(os.path.basename(path))[0]

        if not issubclass(type(image), skrt.image.Image):
            image = skrt.image.Image(image)
        self.set_image(image)

        with open(path, "r", encoding='ascii') as in_file:
            lines = in_file.readlines()

        contours_3d = {}
        i = 0
        for line in lines:
            i_point, j_point, k_point = [float(value) for value in line.split()]
            k_point = self.shape[2] - k_point
            x_point = self.image.idx_to_pos(i_point, 'x')
            y_point = self.image.idx_to_pos(j_point, 'y')
            z_point = self.image.idx_to_pos(k_point, 'z')

            key = get_key(z_point, contours_3d, key_precision)
            if not key in contours_3d:
                contours_3d[key] = []
            contours_3d[key].append((x_point, y_point, z_point))
            i += 1

        self.reset_contours(contours_mean_z(contours_3d))
        
        return self

    def write_point_cloud(self, outdir='.', name='', prefix='', suffix='.txt',
            overwrite=True):
        '''
        Write contour coordinates in VoxTox point-cloud format.

        In the VoxTox point-cloud format, each file contains contour data
        for a single ROI.  Each line gives (column, row, slice) coordinates
        for a single point.  Points are ordered sequentially around a slice.
        Slice numbers are inverted with respect to the usual DICOM numbering.

        **Parameters:**

        outdir : str, default='.'
            Directory where point-cloud file is to be written.

        name : str, default=''
            Name to be used for the point-cloud file.  If this is an
            empty string, the ROI name is used.

        prefix : str, default=''
            String to be placed before point-cloud file name.

        suffix : str, default='.txt'
            String to be place after point-cloud file name.

        overwrite: bool, default=True
            If True, overwrite any pre-existing point file.
        '''

        self.load()

        # Extract contour points
        lines = []
        for z_point, contours in self.get_contours().items():
            k_point = self.shape[2] - self.pos_to_idx(
                    z_point, 'z', return_int=True)
            for contour in contours:
                for x_point, y_point in contour:
                    i_point = self.pos_to_idx(x_point, 'x', return_int=False)
                    j_point = self.pos_to_idx(y_point, 'y', return_int=False)
                    lines.append(f'{i_point:.3f} {j_point:.3f} {k_point}')

        # Set file path
        if not name:
            name = self.name
        point_cloud_path = os.path.join(outdir, f'{prefix}{name}{suffix}')

        # Write point-cloud file
        if not overwrite:
            if os.path.exists(point_cloud_path):
                print(f'File \'{point_cloud_path}\' exists,'\
                        ' and will not be overwritten.')
                print('Call write_point_cloud() with overwrite=True'\
                        ' to overwrite.')
        else:
            with open(point_cloud_path, "w", encoding='ascii') as out_file:
                out_file.write('\n'.join(lines))

class StructureSet(skrt.structures.StructureSet):
    '''VoxTox-specific extensions to Scikit-rt StructureSet class.'''

    def load_point_cloud(self, indir='.', prefix='', suffix='', image=None,
            key_precision=0.1):
        '''
        Load data in VoxTox point-cloud format, and convert to contours.

        **Parameters:**

        indir : str, default='.'
            Path to directory containing point-cloud files.

        prefix : str, default=''
            String to be placed before point-cloud file names.

        suffix : str, default=''
            String to be place after point-cloud file names.

        image= : skrt.image.Image/str/None, default=None
            Associated image from which to extract shape and affine matrix.
            Can either be an existing Image object, or a path from which
            an Image can be created.

        key_precision : float, default=0.1
            Precision with which z-coordinate should match z-plane.
            As a result of rounding errors, and/or small translations
            when applying registration transforms, z-coordinates for points
            in the same plane may not be identical.
        '''

        self.rois = []
        paths = glob.glob(os.path.join(indir,f'{prefix}*{suffix}'))
        for path in paths:
            roi = ROI().load_point_cloud(path, image, key_precision)
            self.rois.append(roi)

        self.loaded = True

        return self

    def write_point_cloud(self, outdir='.', names=None, prefix='',
            suffix='.txt', overwrite=True):
        '''
        Write contours in VoxTox point-cloud format for all, or selected, ROIs.

        In the VoxTox point-cloud format, each file contains contour data
        for a single ROI.  Each line gives (column, row, slice) coordinates
        for a single point.  Points are ordered sequentially around a slice.
        Slice numbers are inverted with respect to the usual DICOM numbering.

        **Parameters:**

        outdir : str, default='.'
            Directory where point-cloud file is to be written.

        names : dict, default={}
            Dictionary mapping between possible ROI names within
            a structure set (dictionary values) and names to be used
            for point-cloud files (dictionary keys).  The possible names
            names can contain wildcards with the '*' symbol.

        prefix : str, default=''
            String to be placed before all point-cloud file names.

        suffix : str, default='.txt'
            String to be place after all point-cloud file names.

        overwrite: bool, default=True
            If True, overwrite any pre-existing point files.
        '''

        # Create temporary StructureSet, with ROIs filtered/renamed.
        ss_tmp = self.filtered_copy(names=names, keep_renamed_only=True)

        if os.path.exists(outdir):
            if overwrite:
                shutil.rmtree(outdir)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Write point clouds.
        for skrt_roi in ss_tmp.get_rois():
            voxtox_roi = ROI(skrt_roi)
            voxtox_roi.write_point_cloud(outdir=outdir, prefix=prefix,
                    suffix=suffix)

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
