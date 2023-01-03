"""Tools for performing image segmentation."""
from pathlib import Path
from shutil import rmtree

from skrt.core import get_logger, is_list, Data, Defaults
from skrt.image import match_image_voxel_sizes, Image
from skrt.structures import StructureSet
from skrt.registration import Registration

class SingleAtlasSegmentation(Data):
    def __init__(
        self, im1=None, im2=None, ss1=None, ss2=None, roi_names=None,
        workdir = "segmentation_workdir", strategy="pull",
        initial_alignment=None, initial_alignment_crop=True,
        initial_alignment_crop_margins=None, initial_transform_name=None,
        crop_to_match_size1=True, voxel_size1=None, bands1=None, pfiles1=None, 
        most_points1=True, roi_crop_margins=None, default_crop_margins=10,
        crop_to_match_size2=True, voxel_size2=None, bands2=None, pfiles2=None, 
        most_points2=True, auto=False, auto_step=-1, overwrite=False,
        capture_output=False, log_level=None, keep_tmp_dir=False):

        # Set up event logging.
        self.log_level = \
                Defaults().log_level if log_level is None else log_level
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)

        # Set images, associated structure sets, and ROI names.
        self.im1 = (im1 if (issubclass(type(im1), Image) or im1 is None)
                else Image(im1))
        self.im2 = (im2 if (issubclass(type(im2), Image) or im2 is None)
                else Image(im2))
        self.ss1 = ss1
        self.ss2 = ss2
        self.roi_names = roi_names

        # Set workdir; optionally delete pre-existing workdir.
        self.workdir = Path(workdir)
        if overwrite and self.workdir.exists():
            rmtree(self.workdir)

        # Define contour-propagation strategies and segmentation steps.
        self.strategies = ["pull", "push"]
        self.steps = ["global", "local"]
        self.auto_step = auto_step

        # Set the default contour-propagation strategy.
        self.strategy = get_option(strategy, None, self.strategies)

        # Set parameters for step-1 registration.
        self.initial_alignment = initial_alignment
        self.initial_alignment_crop = initial_alignment_crop
        self.initial_alignment_crop_margins = initial_alignment_crop_margins
        self.initial_transform_name = initial_transform_name
        self.crop_to_match_size1 = crop_to_match_size1
        self.voxel_size1 = voxel_size1
        self.bands1 = bands1
        self.pfiles1 = pfiles1
        self.most_points1 = most_points1

        # Set parameters for step-2 registration.
        self.roi_crop_margins = roi_crop_margins or {}
        self.default_crop_margins = default_crop_margins
        self.crop_to_match_size2 = crop_to_match_size2
        self.voxel_size2 = voxel_size2
        self.bands2 = bands2
        self.pfiles2 = pfiles2 or pfiles1
        self.most_points2 = most_points2

        # Set parameters for step-1 and step-2 registration.
        self.capture_output = capture_output
        self.keep_tmp_dir = keep_tmp_dir

        # Define dictionaries for storing results.
        self.registrations = {}
        self.segmentations = {}
        for strategy in self.strategies:
            self.registrations[strategy] = {step: {} for step in self.steps}
            self.segmentations[strategy] = {step: {} for step in self.steps}

        # Perform segmentation.
        if auto:
            self.segment(self.strategy, self.auto_step)

    def preprocess(self, im1, im2, ss1=None, ss2=None, roi_names=None,
        alignment=None, alignment_crop=True, crop_margins=None,
        crop_to_match_size=None, voxel_size=None, bands=None):

        im1 = im1.clone_with_structure_set(ss1, roi_names, -1, "Filtered1")
        ss1 = im1.structure_sets[0] if im1.structure_sets[0] else None
        im2 = im2.clone_with_structure_set(ss2, roi_names, -1, "Filtered2")
        ss2 = im2.structure_sets[0] if im2.structure_sets[0] else None

        # Crop primary image to region around alignment structure.
        if alignment_crop:
            if ss2 is not None and alignment in ss2.get_roi_names():
                im2.crop_to_roi(ss2[alignment], crop_margins)

        # Crop images to same size.
        if crop_to_match_size:
            im1.crop_to_image(im2, alignment)
            im2.crop_to_image(im1, alignment)

        # Resample images to same voxel size.
        match_image_voxel_sizes(im1, im2, voxel_size)

        # Perform banding of image grey levels.
        im1.apply_selective_banding(bands)
        im2.apply_selective_banding(bands)

        # Reset structure-set images.
        im1.structure_sets[0].set_image(im1)
        im2.structure_sets[0].set_image(im2)

        return (im1, im2)

    def segment(self, strategy=None, step=None, force=False):
        strategy = get_option(strategy, self.strategy, self.strategies)
        steps = self.get_steps(step)

        if force:
            self.registrations[strategy] = {step: {} for step in steps}
            self.segmentations[strategy] = {step: {} for step in steps}

        if (self.steps[0] in steps
                and not self.registrations[strategy][self.steps[0]]):
            im1, im2 = self.preprocess(
                    im1=self.im1,
                    im2=self.im2,
                    ss1=self.ss1,
                    ss2=self.ss2,
                    roi_names=self.roi_names,
                    alignment=self.initial_alignment,
                    alignment_crop=self.initial_alignment_crop,
                    crop_margins=self.initial_alignment_crop_margins,
                    crop_to_match_size=self.crop_to_match_size1,
                    voxel_size=self.voxel_size1,
                    bands=self.bands1)

            self.ss1_filtered = im1.structure_sets[0]
            self.ss2_filtered = im2.structure_sets[0]

            fixed, moving = get_fixed_and_moving(im1, im2, strategy)

            self.registrations[strategy][self.steps[0]] = (
                    Registration(
                        self.workdir / f"{strategy}1",
                        fixed=fixed,
                        moving=moving,
                        pfiles=self.pfiles1,
                        initial_alignment=self.initial_alignment,
                        initial_transform_name=self.initial_transform_name,
                        capture_output=self.capture_output,
                        log_level=self.log_level,
                        keep_tmp_dir=self.keep_tmp_dir,
                        overwrite=force,
                        )
                    )

            self.segment_structure_set(im2.structure_sets[0],
                    strategy, self.steps[0], self.most_points1)

        if (self.steps[1] in steps
                and not self.registrations[strategy][self.steps[1]]):
            rois = []
            for roi_name in self.roi_names:
                im1, im2 = self.preprocess(
                        im1=self.im1,
                        im2=self.im2,
                        ss1=self.ss1,
                        ss2=self.ss2,
                        roi_names={roi_name: self.roi_names[roi_name]},
                        alignment=roi_name,
                        alignment_crop=True,
                        crop_margins=self.roi_crop_margins.get(
                            roi_name, self.default_crop_margins),
                        crop_to_match_size=self.crop_to_match_size2,
                        voxel_size=self.voxel_size2,
                        bands=self.bands2)

                fixed, moving = get_fixed_and_moving(im1, im2, strategy)

                self.registrations[strategy][self.steps[1]][roi_name] = (
                        Registration(
                            self.workdir / f"{strategy}2" / roi_name,
                            fixed=fixed,
                            moving=moving,
                            pfiles=self.pfiles2,
                            initial_alignment=roi_name,
                            initial_transform_name=self.initial_transform_name,
                            capture_output=self.capture_output,
                            log_level=self.log_level,
                            keep_tmp_dir=self.keep_tmp_dir,
                            overwrite=force,
                            )
                        )

                self.segment_roi(im2.structure_sets[0][roi_name],
                        strategy, self.steps[1], self.most_points2)

            for reg_step in self.segmentations[strategy][self.steps[1]]:
                self.segmentations[strategy][self.steps[1]][reg_step].set_image(
                        self.im1)

    def segment_roi(self, roi, strategy, step, most_points=True):
        for reg_step in self.registrations[strategy][step][roi.name].steps:
            if not reg_step in self.segmentations[strategy][step]:
                self.segmentations[strategy][step][reg_step] = StructureSet(
                        name=f"{strategy}_{step}_{reg_step}")
            self.segmentations[strategy][step][reg_step].add_roi(
                    self.registrations[strategy][step][roi.name].transform(
                        to_transform=roi,
                        step=reg_step,
                        transform_points=(strategy == "push"))
                    )

            if (strategy == "pull" and most_points):
                self.segmentations[strategy][step][reg_step][roi.name]\
                        .reset_contours(most_points=True)

    def segment_structure_set(self, ss, strategy, step, most_points=True):
        for reg_step in self.registrations[strategy][step].steps:
            self.segmentations[strategy][step][reg_step] = (
                    self.registrations[strategy][step].transform(
                        to_transform=ss,
                        step=reg_step,
                        transform_points=(strategy == "push"))
                    )

            if (strategy == "pull" and most_points):
                self.segmentations[strategy][step][reg_step]\
                        .reset_contours(most_points=True)

            self.segmentations[strategy][step][reg_step].set_image(self.im1)
            self.segmentations[strategy][step][reg_step].name = (
                f"{strategy}_{step}_{reg_step}")

    def get_steps(self, step):
        # Make list of steps to run
        if step is None:
            steps_input = self.steps
        elif not isinstance(step, list):
            steps_input = [step]
        else:
            steps_input = step

        # Check all steps exist and convert numbers to names
        steps = []
        for step_now in steps_input:
            if isinstance(step_now, str):
                if not step_now in self.steps:
                    raise RuntimeError(
                            f"Invalid segmentation step: {step}")
            else:
                step_now = self.steps[step_now]

            steps.append(step_now)

        if steps:
            steps_index = max(
                    [self.steps.index(step_now) for step_now in steps])
            steps = self.steps[: steps_index + 1]

        return steps

    def get_registration(self, strategy=None, step=None, roi_name=None,
            force=False):
        strategy = get_option(strategy, self.strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        self.segment(strategy, step, force)

        if "local" == step:
            roi_names = sorted([key
                for key in self.registrations[strategy][step]])
            roi_name = get_option(roi_name, None, roi_names)
            return self.registrations[strategy][step][roi_name]
        else:
            return self.registrations[strategy][step]

    def get_segmentation(self, strategy=None, step=None, reg_step=None,
            force=False):
        strategy = get_option(strategy, self.strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        self.segment(strategy, step, force)
        reg_steps = list(self.segmentations[strategy][step])
        reg_step = get_option(reg_step, None, reg_steps)
        return self.segmentations[strategy][step][reg_step]


def get_option(opt=None, fallback_opt=None, allowed_opts=None):

    if not is_list(allowed_opts) or not len(allowed_opts):
        raise RuntimeError("No allowed options specified - returning None")

    if isinstance(opt, int):
        option = allowed_opts[opt]
        return option

    if opt in allowed_opts:
        option = opt
    elif fallback_opt in allowed_opts:
        option = fallback_opt
    else:
        option = allowed_opts[-1]
    return option

    raise RuntimeError("Unable to determine valid option")

def get_fixed_and_moving(im1, im2, strategy):
    return (im1, im2) if ("pull" == strategy) else (im2, im1)
