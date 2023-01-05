"""Tools for performing image segmentation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree

from skrt.core import get_logger, is_list, Data, Defaults
from skrt.image import match_image_voxel_sizes, Image
from skrt.structures import get_consensus_types, StructureSet
from skrt.registration import Registration

class MultiAtlasSegmentation(Data):
    def __init__(
        self, im1=None, im2=None, ss1=None, ss2=None, log_level=None,
        workdir="segmentation_workdir", overwrite=False, auto=False,
        auto_step=-1, strategy="pull", roi_names=None,
        consensus_types=["majority"], max_workers=1, **kwargs):

        # Set images and associated structure sets.
        self.im1 = ensure_image(im1)
        self.im2 = ensure_dict(im2, ensure_image)
        self.ss1 = ensure_structure_set(ss1)
        self.ss2 = ensure_dict(ss2, ensure_structure_set)
        for idx in self.im2:
            if idx is not self.ss2:
                self.ss2[idx] = None

        # Set up event logging.
        self.log_level = (Defaults().log_level if log_level is None
                else log_level)
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)

        # Set workdir; optionally delete pre-existing workdir.
        self.workdir = Path(workdir)
        self.overwrite = overwrite
        if self.overwrite and self.workdir.exists():
            rmtree(self.workdir)

        # Define contour-propagation strategies and segmentation steps.
        self.strategies = get_contour_propagation_strategies()
        self.steps = get_segmentation_steps()

        # Set parameters for automatic segmentation.
        # Note that these values are passed to the SingleAtlasSegmentation
        # constructor, and automatic segmenation should always be enabled.
        self.auto = True
        self.auto_step = auto_step

        # Set default contour-propagation strategy.
        self.strategy = get_option(strategy, None, self.strategies)

        # Set names of ROIs to be segmented.
        self.roi_names = roi_names

        # Set default method(s) for defining consensus contours.
        self.consensus_types = get_sas_consensus_types(consensus_types)

        # Set maximum number of processes when multiprocessing.
        self.max_workers = max_workers

        # Store parameters to be passes to instances of SingleAtlasSegmentation.
        self.sas_kwargs = kwargs

        # Define dictionaries for storing results.
        self.sass = {idx: None for idx in self.im2}
        self.consensuses = {strategy: {step: {} for step in self.steps}
                for strategy in self.strategies}

        # Perform segmentation.
        if auto:
            self.segment(strategy=self.strategy, step=self.auto_step)

    def segment(self, atlas_ids=None, strategy=None, step=None,
            consensus_types=None, force=False):
        atlas_ids = atlas_ids or self.sass.keys()
        strategy = get_option(strategy, self.strategy, self.strategies)
        steps = get_steps(step)

        active_ids = []
        sas_args = {}
        for idx in atlas_ids:
            if ((self.sass[idx] is None) or (force) or
                    any([self.sass[idx].segmentations[strategy][step] is None
                        for step in steps])):
                active_ids.append(idx)
                args = (self.im1, self.im2[idx], self.ss1, self.ss2[idx],
                        self.log_level, self.workdir / str(idx), self.overwrite,
                        self.auto, self.auto_step, self.strategy,
                        self.roi_names,)
                if self.max_workers == 1:
                    self.sass[idx] = SingleAtlasSegmentation(
                            *args, **self.sas_kwargs)
                else:
                    sas_args[idx] = args

        # Submitting multiple single-atlas segmentations to a ThreadPoolExecutor
        # can give some speed increase compared to using a single thread.
        # The ThreadPoolExecutor can be replaced by ProcessPoolExecutor
        # (import needed, and environment for running registration
        # executables must be set up for each process.  In practice
        # this tends to be slower than single-thread execution, possibly
        # because of the time taken in transferring data to and from
        # the child processes.
        if sas_args:
            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for idx in sas_args:
                    futures[idx] = executor.submit(SingleAtlasSegmentation,
                            *sas_args[idx], **self.sas_kwargs)
            for idx in futures:
                self.sass[idx] = futures[idx].result()

        for idx in active_ids:
            self.sass[idx].segment(strategy, step, force)
            for step in steps:
                reg = self.sass[idx].segmentations[strategy][step]
                for reg_step in reg:
                    reg[reg_step].name = f"{idx}_{reg[reg_step].name}"

        if consensus_types is None:
            consensus_types = self.consensus_types
        if consensus_types:
            for step in steps:
                for reg_step in self.get_sas().segmentations[strategy][step]:
                    self.set_consensuses(
                            strategy, step, reg_step, consensus_types, force)

    def get_consensus(self, strategy=None, step=None, reg_step=None,
            consensus_type=None, force=False):
        strategy = get_option(strategy, self.strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        if not consensus_type:
            consensus_type = (self.consensus_types[0] if self.consensus_types
                    else get_consensus_types()[0])
        self.segment(None, strategy, step, consensus_type, force)
        reg_steps = list(self.get_sas().segmentations[strategy][step])
        reg_step = get_option(reg_step, None, reg_steps)
        return self.consensuses[strategy][step][reg_step][consensus_type]

    def get_sas(self, atlas_id=None):
        if atlas_id is None:
            atlas_id = list(self.sass)[-1]
        return self.sass[atlas_id]

    def get_sas_segmentations(self, atlas_ids=None, strategy=None,
            step=None, reg_step=None, force=False):
        if atlas_ids is None:
            atlas_ids = list(self.sass.keys())
        elif isinstance(atlas_ids, int):
            atlas_ids = [atlas_ids]

        return sum([self.get_sas(atlas_id).get_segmentation(
            strategy, step, reg_step, force) for atlas_id in atlas_ids],
            StructureSet())

    def set_consensuses(self, strategy=None, step=None, reg_step=None,
            consensus_types=None, force=False):

        if consensus_types is None:
            consensus_types = self.consensus_types
        if not consensus_types:
            return
        if isinstance(consensus_types, str):
            consensus_types = [consensus_types]

        strategy = get_option(strategy, self.strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        reg_steps = list(self.get_sas().segmentations[strategy][step])
        reg_step = get_option(reg_step, None, reg_steps)
        if not reg_step in self.consensuses[strategy][step]:
            self.consensuses[strategy][step][reg_step] = {}

        all_rois = {roi_name: [] for roi_name in self.roi_names}
        for sas in self.sass.values():
            ss = sas.get_segmentation(strategy, step, reg_step)
            for roi in ss.get_rois():
                if roi.name in all_rois:
                    all_rois[roi.name].append(roi)

        for consensus_type in consensus_types:
            if (consensus_type in self.consensuses[strategy][step][reg_step]
                    and not force):
                continue
            ss_consensus = StructureSet(
                    name=f"{strategy}_{step}_{reg_step}_{consensus_type}")
            for roi_name, rois in all_rois.items():
                roi = StructureSet(rois).get_consensus(consensus_type)
                roi.name = roi_name
                ss_consensus.add_roi(roi)
                
            ss_consensus.set_image(self.im1)
            ss_consensus.reset_contours(most_points=True)
            self.consensuses[strategy][step][reg_step][consensus_type] = (
                    ss_consensus)

class SingleAtlasSegmentation(Data):
    def __init__(
        self, im1=None, im2=None, ss1=None, ss2=None, log_level=None,
        workdir="segmentation_workdir", overwrite=False,
        auto=False, auto_step=-1, strategy="pull", roi_names=None,
        initial_alignment=None, initial_alignment_crop=True,
        initial_alignment_crop_margins=None, initial_transform_name=None,
        crop_to_match_size1=True, voxel_size1=None, bands1=None, pfiles1=None, 
        most_points1=True, roi_crop_margins=None, default_crop_margins=10,
        crop_to_match_size2=True, voxel_size2=None, bands2=None, pfiles2=None, 
        most_points2=True, capture_output=False, keep_tmp_dir=False):

        # Set images and associated structure sets.
        self.im1 = ensure_image(im1)
        self.im2 = ensure_image(im2)
        self.ss1 = ensure_structure_set(ss1)
        self.ss2 = ensure_structure_set(ss2)

        # Set up event logging.
        self.log_level = (Defaults().log_level if log_level is None
                else log_level)
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)

        # Set workdir; optionally delete pre-existing workdir.
        self.workdir = Path(workdir)
        if overwrite and self.workdir.exists():
            rmtree(self.workdir)

        # Define contour-propagation strategies and segmentation steps.
        self.strategies = get_contour_propagation_strategies()
        self.steps = get_segmentation_steps()

        # Set step for automatic segmenation, default contour-propagation
        # strategy, and names or ROIs to be segmented.
        self.auto_step = auto_step
        self.strategy = get_option(strategy, None, self.strategies)
        self.roi_names = roi_names

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
        steps = get_steps(step)

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

def ensure_any(in_val):
    return in_val

def ensure_image(im):
    return (im if (issubclass(type(im), Image) or im is None) else Image(im))

def ensure_structure_set(ss):
    return (ss if (isinstance(ss, StructureSet) or ss is None)
            else StructureSet(ss))

def ensure_dict(in_val, ensure_type=ensure_any):
    if isinstance(in_val, dict):
        return {key: ensure_type(val) for key, val in in_val.items()}
    elif is_list(in_val):
        return {idx: ensure_type(val) for idx, val in enumerate(in_val)}
    else:
        return {0: in_val}

def get_contour_propagation_strategies():
    return ["pull", "push"]

def get_segmentation_steps():
    return ["global", "local"]

def get_steps(step):
    all_steps = get_segmentation_steps()
    # Make list of steps to run
    if step is None:
        steps_input = all_steps
    elif not is_list(step):
        steps_input = [step]
    else:
        steps_input = step

    # Check that all steps exist, and convert numbers to names.
    steps = []
    for step_now in steps_input:
        if isinstance(step_now, str):
            if not step_now in all_steps:
                raise RuntimeError(
                        f"Invalid segmentation step: {step}")
        else:
            step_now = all_steps[step_now]

        steps.append(step_now)

    if steps:
        steps_index = max(
                [all_steps.index(step_now) for step_now in steps])
        steps = all_steps[: steps_index + 1]

    return steps

def get_sas_consensus_types(consensus_types):
    # Check consensus types.
    if consensus_types is None:
        return get_consensus_types()
    elif not consensus_types:
        return []
    elif isinstance(consensus_types, str):
        return [consensus_types]

    return consensus_types
