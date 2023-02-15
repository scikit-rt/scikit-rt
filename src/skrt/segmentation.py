"""
Classes and functions relating to image segmentation.

This module implements atlas-based segmentation, where an atlas is
a reference image, for which regions of interest (ROIs) have been segmented
in some way.

In single-atlas segmentation, a single atlas is registered to a target image
that is to be segmented.  Atlas segmentations are then mapped to the target
image, using the registration transform.  An ROI may be represented as
a binary mask or as sets of contour points, resulting in two strategies
for mapping ROIs:

    - atlas and target are taken as fixed and moving image respectively for
      registration, then ROI contour points are pushed from atlas to target;
    - atlas and target are taken as moving and fixed image respectively for
      the registration, then ROI binary masks are pulled from atlas to target.

In multi-atlas segmentation, single-atlas segmentation is performed
multiple times, using a different atlas each time, then the results from
the individual segmentations are combined in some way.

The following classes are defined:
SingleAtlasSegmentation() : class for performing single-atlas segmentation;
MultiAtlasSegmentation() : class for performing multi-atlas segmentation.

The following are utility functions used by SingleAtlasSegmentation()
and MultiAtlasSegmentation():
ensure_any()
ensure_dict()
ensure_image()
ensure_structure_set()
get_contour_propagation_strategies()
get_fixed_and_moving()
get_option()
get_options()
get_segmentation_steps()
get_steps()
get_structure_set_index()

The following function allows for comparison of different single-atlas
segmentations:
get_sas_comparisons()
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from inspect import signature
from itertools import combinations
from pathlib import Path
from shutil import rmtree

import pandas as pd

from skrt.core import (
        get_dict_permutations, get_logger, get_stat_functions, is_list,
        Data, Defaults, tic, toc)
from skrt.image import match_images, Image
from skrt.structures import (
        get_by_slice_methods, get_comparison_metrics, get_consensus_types,
        StructureSet)
from skrt.registration import Registration

class MultiAtlasSegmentation(Data):
    def __init__(
        self, im1=None, im2=None, ss1=None, ss2=None, log_level=None,
        workdir="segmentation_workdir", overwrite=False,
        auto=False, auto_step=None, auto_strategies=None,
        auto_reg_setup_only=False,
        default_step=None, default_strategy="pull", roi_names=None,
        metrics=None, slice_stats=None, default_by_slice=None,
        consensus_types="majority", default_consensus_type="majority",
        max_workers=1, **kwargs):

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

        # Note that these values are passed to the SingleAtlasSegmentation
        # constructor, and automatic segmenation should always be enabled.
        self.auto = True

        # Set step and strategies for automatic segmenation,
        # default step and strategy, and names or ROIs to be segmented.
        self.auto_step = get_option(auto_step, default_step, self.steps)
        self.default_step = get_option(default_step, self.auto_step, self.steps)
        self.auto_strategies = get_options(
                auto_strategies, default_strategy, self.strategies)
        self.default_strategy = default_strategy or self.auto_strategies[-1]
        self.roi_names = roi_names

        # Set parameters for post-segmentation ROI comparisons.
        self.metrics = metrics
        self.slice_stats = slice_stats
        self.default_by_slice = (Defaults().by_slice
                                 if default_by_slice is None
                                 else default_by_slice)

        # Set default method(s) for defining consensus contours.
        self.consensus_types = get_options(
                consensus_types, True, get_consensus_types())
        self.default_consensus_type = get_options(
            default_consensus_type, self.consensus_types,
            get_consensus_types())[-1]
        self.consensus_atlas_ids = set()

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
            for auto_strategy in self.auto_strategies:
                self.segment(strategy=auto_strategy, step=self.auto_step,
                             reg_setup_only=auto_reg_setup_only)

    def segment(self, atlas_ids=None, strategy=None, step=None,
            consensus_types=None, force=False, reg_setup_only=False):
        atlas_ids = get_options(atlas_ids, True, list(self.sass))
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        steps = get_steps(step)

        active_ids = []
        sas_args = {}
        sas_auto = True
        for idx in atlas_ids:
            if ((self.sass[idx] is None) or (force) or
                    any([not self.sass[idx].segmentations[strategy][step]
                        for step in steps])):
                active_ids.append(idx)
                args = (self.im1, self.im2[idx], self.ss1, self.ss2[idx],
                        self.log_level, self.workdir / str(idx), self.overwrite,
                        sas_auto, self.auto_step, self.auto_strategies,
                        reg_setup_only, self.default_step,
                        self.default_strategy, self.roi_names,)
                if self.max_workers == 1:
                    tic()
                    self.logger.info(f"Segmenting with atlas '{idx}'")
                    self.sass[idx] = SingleAtlasSegmentation(
                            *args, **self.sas_kwargs)
                    self.logger.info(f"Segmentation with atlas '{idx}': "
                                     f"{toc():.2f} s")
                else:
                    sas_args[idx] = args

        # Submitting multiple single-atlas segmentations to a ThreadPoolExecutor
        # can give some speed increase compared to using a single thread.
        # The ThreadPoolExecutor can be replaced by ProcessPoolExecutor
        # (import needed, and environment for running registration
        # executables must be set up for each process).  In practice
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

        if reg_setup_only:
            return

        for idx in active_ids:
            self.sass[idx].segment(strategy, step, force)
            for step in steps:
                seg = self.sass[idx].segmentations[strategy][step]
                for reg_step in seg:
                    seg[reg_step].name = f"{idx}_{seg[reg_step].name}"

        if consensus_types is None:
            consensus_types = self.consensus_types
        if consensus_types:
            for step in steps:
                for reg_step in self.get_sas(atlas_ids[0]).get_reg_steps(step):
                    self.set_consensuses(atlas_ids, strategy, step, reg_step,
                                         consensus_types, force)

    def get_consensus(self, atlas_ids=None, strategy=None, step=None,
                      reg_step=None, consensus_type=None, force=False):
        atlas_ids = get_options(atlas_ids, True, list(self.sass))
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        consensus_type = consensus_type or self.default_consensus_type
        if self.consensus_atlas_ids != set(atlas_ids):
            self.segment(atlas_ids, strategy, step, consensus_type, force)
        all_reg_steps = self.get_sas(atlas_ids[0]).get_reg_steps(step)
        reg_step = get_option(reg_step, None, all_reg_steps)
        if ((reg_step not in self.consensuses[strategy][step]) or
            (consensus_type not in self.consensuses[strategy][step][reg_step])):
            self.segment(atlas_ids, strategy, step, consensus_type, force)
        return self.consensuses[strategy][step][reg_step][consensus_type]

    def get_sas(self, atlas_id=None):
        if atlas_id is None:
            atlas_id = list(self.sass)[-1]
        return self.sass[atlas_id]

    def get_sas_segmentations(self, atlas_ids=True, strategy=None,
            step=None, reg_step=None, force=False):

        atlas_ids = get_options(atlas_ids, True, list(self.sass.keys()))

        return sum([self.get_sas(atlas_id).get_segmentation(
            strategy, step, reg_step, force) for atlas_id in atlas_ids],
            StructureSet())

    def set_consensuses(self, atlas_ids=None, strategy=None, step=None,
                        reg_step=None, consensus_types=None, force=False):

        if consensus_types is None:
            consensus_types = self.consensus_types
        if not consensus_types:
            return
        if isinstance(consensus_types, str):
            consensus_types = [consensus_types]

        atlas_ids = get_options(atlas_ids, True, list(self.sass))
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        step = get_option(step, self.auto_step, self.steps)
        all_reg_steps = self.get_sas(atlas_ids[0]).get_reg_steps(step)
        reg_step = get_option(reg_step, None, all_reg_steps)
        if not reg_step in self.consensuses[strategy][step]:
            self.consensuses[strategy][step][reg_step] = {}

        all_rois = {roi_name: [] for roi_name in self.roi_names}
        for atlas_id in atlas_ids:
            sas = self.get_sas(atlas_id)
            ss = sas.get_segmentation(strategy, step, reg_step)
            for roi in ss.get_rois():
                if roi.name in all_rois:
                    all_rois[roi.name].append(roi)

        atlas_ids = set(atlas_ids)
        for consensus_type in consensus_types:
            if (consensus_type in self.consensuses[strategy][step][reg_step]
                and (not force) and (self.consensus_atlas_ids == atlas_ids)):
                continue

            if self.consensus_atlas_ids != atlas_ids:
                for tmp_reg_step in self.consensuses[strategy][step]:
                    self.consensuses[strategy][step][tmp_reg_step] = {}
                self.consensus_atlas_ids = atlas_ids

            ss_consensus = StructureSet(
                    name=f"{strategy}_{step}_{reg_step}_{consensus_type}")
            for roi_name, rois in all_rois.items():
                roi = StructureSet(rois).get_consensus(consensus_type)
                roi.name = roi_name
                ss_consensus.add_roi(roi)
                
            ss_consensus.set_image(self.im1, add_to_image=False)
            ss_consensus.reset_contours(most_points=True)
            self.consensuses[strategy][step][reg_step][consensus_type] = (
                    ss_consensus)

    def get_comparison(
            self, consensus_types=None, atlas_ids=True,
            id1=None, id2=None, to_keep=None, strategies=None,
            steps=None, reg_steps=None, force=False, metrics=None,
            slice_stats=None, default_by_slice=None, voxel_size=None,
            name_as_index=False, atlas_ids_to_compare=False,
            combination_length=None, **kwargs):

        if not hasattr(self, "ss1_filtered"):
            return

        consensus_types = get_options(
                consensus_types, self.consensus_types, get_consensus_types())
        if not (is_list(atlas_ids) and atlas_ids and is_list(atlas_ids[0])):
            atlas_ids = get_options(atlas_ids, None, list(self.sass))
            if (combination_length or 0) > 1:
                atlas_ids = list(combinations(atlas_ids, combination_length))
            else:
                atlas_ids = [atlas_ids]

        start = tuple() if isinstance(atlas_ids[0], tuple) else list()
        all_atlas_ids = list(set(sum(atlas_ids, start)))

        atlas_ids_to_compare = get_options(
            atlas_ids_to_compare, None, all_atlas_ids)
        n_atlas_ids_to_compare = len(atlas_ids_to_compare)

        strategies = get_options(
                strategies, self.default_strategy, self.strategies)
        steps = get_options(
                steps, self.default_step, self.steps)
        metrics = get_options(metrics, self.metrics, get_comparison_metrics())
        slice_stats = get_options(slice_stats, self.slice_stats,
                                  get_stat_functions())
        default_by_slice = get_option(default_by_slice, self.default_by_slice,
                                      get_by_slice_methods())

        sas_id = get_options(atlas_ids_to_compare, True, all_atlas_ids)[0]
        sas = self.get_sas(sas_id)
        step_reg_steps = {}
        for step in steps:
            all_reg_steps = sas.get_reg_steps(step)
            step_reg_steps[step] = get_options(reg_steps, True, all_reg_steps)
        
        df = None
        ss1 = sas.ss1_filtered.filtered_copy(to_keep=to_keep)
        for idx, id2 in enumerate(atlas_ids_to_compare + consensus_types):
            if idx < n_atlas_ids_to_compare:
                sas = self.get_sas(id2)
            for strategy in strategies:
                for step in steps:
                    for reg_step in step_reg_steps[step]:
                        if idx < n_atlas_ids_to_compare:
                            tic()
                            ss2s = {id2: (sas.get_segmentation(
                                strategy, step, reg_step
                                ).filtered_copy(to_keep=to_keep))}
                            self.logger.info(
                                    f"{id2} [{strategy}, {step}, {reg_step}]: "
                                    f"{toc():.2f} s")
                        else:
                            ss2s = {}
                            for tmp_atlas_ids in atlas_ids:
                                label = "_".join([str(atlas_id)
                                                  for atlas_id in tmp_atlas_ids]
                                                 + [id2])
                                ss2s[label] = (self.get_consensus(tmp_atlas_ids,
                                    strategy, step, reg_step, consensus_type=id2
                                    ).filtered_copy(to_keep=to_keep))
                            self.logger.info(
                                    f"{label} "
                                    f"[{strategy}, {step}, {reg_step}]: "
                                    f"{toc():.2f} s")

                        for id2_label, ss2 in ss2s.items():
                            df_tmp = ss1.get_comparison(
                                    ss2, metrics=metrics,
                                    slice_stats=slice_stats,
                                    default_by_slice=default_by_slice,
                                    voxel_size=voxel_size,
                                    name_as_index=name_as_index,
                                    **kwargs)

                            if df_tmp is None:
                                continue

                            for label, value in [
                                    ("id1", id1), ("id2", id2_label),
                                    ("strategy", strategy), ("step", step),
                                    ("reg_step", reg_step)]:
                                if value is not None:
                                    df_tmp[label] = pd.Series(
                                            df_tmp.shape[0] * [value])

                            if df is None:
                                df = df_tmp
                            else:
                                df = pd.concat([df, df_tmp], ignore_index=True)

        return df

    def get_similarity_scores(
            self, atlas_ids=None, strategy=None, step=None, roi_name=None,
            reg_step=None, force=False, max_keep=None, **kwargs):

        atlas_ids = get_options(atlas_ids, True, list(self.sass))
        scores1 = {}
        for atlas_id in atlas_ids:
            self.sass[atlas_id].segment(
                    strategy, step, force, reg_setup_only=True)
            reg = self.sass[atlas_id].get_registration(
                    strategy, step, roi_name, force)
            score = reg.get_mutual_information(reg_step, **kwargs)
            if score not in scores1:
                scores1[score] = []
            scores1[score].append(atlas_id)

        scores2 = {}
        for score, atlas_ids in sorted(scores1.items()):
            for atlas_id in sorted(atlas_ids):
                scores2[atlas_id] = score
        
        if max_keep is not None and len(scores2) > max_keep:
            return dict(list(scores2.items())[: max_keep])

        return scores2


class SingleAtlasSegmentation(Data):
    def __init__(
        self, im1=None, im2=None, ss1=None, ss2=None, log_level=None,
        workdir="segmentation_workdir", overwrite=False,
        auto=False, auto_step=None, auto_strategies=None,
        auto_reg_setup_only=False,
        default_step=-1, default_strategy="pull", roi_names=None,
        ss1_index=-1, ss2_index=-1, ss1_name="Filtered1", ss2_name="Filtered2",
        initial_crop_focus=None, initial_crop_margins=None,
        initial_alignment=None, initial_transform_name=None,
        crop_to_match_size1=True, voxel_size1=None, bands1=None, pfiles1=None, 
        most_points1=True, roi_crop_margins=None, default_roi_crop_margins=10,
        roi_pfiles=None, crop_to_match_size2=True, voxel_size2=None,
        default_roi_bands=None, roi_bands=None, pfiles2=None,
        most_points2=True, capture_output=False,
        keep_tmp_dir=False, metrics=None, slice_stats=None,
        default_by_slice=None):

        # Set images and associated structure sets.
        self.im1 = ensure_image(im1)
        self.im2 = ensure_image(im2)
        self.ss1_index = get_structure_set_index(ss1_index, self.im1)
        self.ss2_index = get_structure_set_index(ss2_index, self.im2)
        self.ss1 = ensure_structure_set(ss1)
        self.ss2 = ensure_structure_set(ss2)
        self.ss1_name = ss1_name
        self.ss2_name = ss2_name

        # Set up event logging.
        self.log_level = (Defaults().log_level if log_level is None
                else log_level)
        self.logger = get_logger(
                name=type(self).__name__, log_level=self.log_level)

        # Set workdir; optionally delete pre-existing workdir.
        self.workdir = Path(workdir)
        if overwrite and self.workdir.exists():
            rmtree(self.workdir)

        # Define recognised contour-propagation strategies
        # and segmentation steps.
        self.strategies = get_contour_propagation_strategies()
        self.steps = get_segmentation_steps()

        # Set step and strategies for automatic segmenation,
        # default step and strategy, and names or ROIs to be segmented.
        self.auto_step = get_option(auto_step, default_step, self.steps)
        self.default_step = get_option(default_step, self.auto_step, self.steps)
        self.auto_strategies = get_options(
                auto_strategies, default_strategy, self.strategies)
        self.default_strategy = default_strategy or self.auto_strategies[-1]
        self.roi_names = roi_names

        # Set parameters for step-1 registration.
        self.initial_alignment = initial_alignment
        self.initial_crop_focus = initial_crop_focus
        self.initial_crop_margins = initial_crop_margins
        self.crop_to_match_size1 = crop_to_match_size1
        self.voxel_size1 = voxel_size1
        self.bands1 = bands1
        self.pfiles1 = {reg_step: pfile
                        for reg_step, pfile in (pfiles1 or {}).items() if pfile}
        self.most_points1 = most_points1

        # Set parameters for step-2 registration.
        self.roi_crop_margins = roi_crop_margins or {}
        self.default_roi_crop_margins = default_roi_crop_margins
        self.crop_to_match_size2 = crop_to_match_size2
        self.voxel_size2 = voxel_size2
        self.roi_bands = roi_bands or {}
        self.default_roi_bands = default_roi_bands
        self.pfiles2 = pfiles2 or pfiles1
        self.pfiles2_non_null = {reg_step: pfile for reg_step, pfile
                                 in (self.pfiles2 or {}).items() if pfile}
        self.roi_pfiles = roi_pfiles or {}
        self.most_points2 = most_points2

        # Set parameters for step-1 and step-2 registration.
        self.initial_transform_name = (initial_transform_name
                                       or "initial_alignment")
        self.capture_output = capture_output
        self.keep_tmp_dir = keep_tmp_dir

        # Set parameters for post-segmentation ROI comparisons.
        self.metrics = metrics
        self.slice_stats = slice_stats
        self.default_by_slice = (Defaults().by_slice
                                 if default_by_slice is None
                                 else default_by_slice)

        # Define dictionaries for storing results.
        self.registrations = {}
        self.segmentations = {}
        for strategy in self.strategies:
            self.registrations[strategy] = {step: {} for step in self.steps}
            self.segmentations[strategy] = {step: {} for step in self.steps}

        # Perform segmentation.
        if auto:
            for auto_strategy in self.auto_strategies:
                self.segment(auto_strategy, self.auto_step,
                             reg_setup_only=auto_reg_setup_only)

    def segment(self, strategy=None, step=None, force=False,
                reg_setup_only=False):
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        steps = get_steps(step)

        if force:
            for step in steps:
                self.registrations[strategy][step] = {}
                self.segmentations[strategy][step] = {}

        if self.steps[0] in steps:
            if not self.registrations[strategy][self.steps[0]]:
                im1, im2 = match_images(
                        im1=self.im1,
                        im2=self.im2,
                        ss1_index = self.ss1_index,
                        ss2_index = self.ss2_index,
                        ss1=self.ss1,
                        ss2=self.ss2,
                        ss1_name=self.ss1_name,
                        ss2_name=self.ss2_name,
                        roi_names=self.roi_names,
                        im2_crop_focus=self.initial_crop_focus,
                        im2_crop_margins=self.initial_crop_margins,
                        alignment=(self.initial_alignment
                                   if self.crop_to_match_size1 else False),
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

            if (not reg_setup_only and
                not self.segmentations[strategy][self.steps[0]]):
                reg = self.registrations[strategy][self.steps[0]]
                im2 = (reg.fixed_source if "push" == strategy
                       else reg.moving_source)
                self.segment_structure_set(im2.structure_sets[0],
                        strategy, self.steps[0], self.most_points1)

        if self.steps[1] in steps and (
                not self.registrations[strategy][self.steps[1]] or
                not self.segmentations[strategy][self.steps[1]]):

            if not self.registrations[strategy][self.steps[1]]:

                for roi_name in self.roi_names:
                    im1, im2 = match_images(
                            im1=self.im1,
                            im2=self.im2,
                            ss1_index = self.ss1_index,
                            ss2_index = self.ss2_index,
                            ss1=self.get_segmentation(strategy, 0, -1),
                            ss2=self.ss2,
                            ss1_name=self.ss1_name,
                            ss2_name=self.ss2_name,
                            roi_names={roi_name: self.roi_names[roi_name]},
                            im2_crop_focus=roi_name,
                            im2_crop_margins=self.roi_crop_margins.get(
                                roi_name, self.default_roi_crop_margins),
                            alignment=(roi_name if self.crop_to_match_size2
                                       else False),
                            voxel_size=self.voxel_size2,
                            bands=self.roi_bands.get(
                                roi_name, self.default_roi_bands)
                            )

                    fixed, moving = get_fixed_and_moving(im1, im2, strategy)

                    self.registrations[strategy][self.steps[1]][roi_name] = (
                            Registration(
                                self.workdir / f"{strategy}2" / roi_name,
                                fixed=fixed,
                                moving=moving,
                                pfiles=self.roi_pfiles.get(
                                    roi_name, self.pfiles2_non_null),
                                initial_alignment=roi_name,
                                initial_transform_name=(
                                    self.initial_transform_name),
                                capture_output=self.capture_output,
                                log_level=self.log_level,
                                keep_tmp_dir=self.keep_tmp_dir,
                                overwrite=force,
                                )
                            )

            if (not reg_setup_only
                and not self.segmentations[strategy][self.steps[1]]):
                if self.initial_alignment:
                    self.segmentations[strategy][self.steps[1]]\
                            [self.initial_transform_name] = {}
                for reg_step, pfile in self.pfiles2.items():
                    self.segmentations[strategy][self.steps[1]][reg_step] = {}

                for roi_name in self.roi_names:
                    reg = self.registrations[strategy][self.steps[1]][roi_name]
                    im2 = (reg.fixed_source if "push" == strategy
                           else reg.moving_source)
                    self.segment_roi(im2.structure_sets[0][roi_name], strategy,
                                     self.steps[1], self.most_points2)

            for reg_step in list(self.segmentations[strategy][self.steps[1]]):
                if self.segmentations[strategy][self.steps[1]][reg_step]:
                    self.segmentations[strategy][self.steps[1]][reg_step].\
                            set_image(self.im1, add_to_image=False)
                else:
                    del self.segmentations[strategy][self.steps[1]][reg_step]

    def segment_roi(self, roi, strategy, step, most_points=True):
        for reg_step in self.registrations[strategy][step][roi.name].steps:
            if (not reg_step in self.segmentations[strategy][step]
                or not self.segmentations[strategy][step][reg_step]):
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

            self.segmentations[strategy][step][reg_step].set_image(
                    self.im1, add_to_image=False)
            self.segmentations[strategy][step][reg_step].name = (
                f"{strategy}_{step}_{reg_step}")

    def get_registration(self, strategy=None, step=None, roi_name=None,
            force=False):
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        step = get_option(step, self.default_step, self.steps)
        self.segment(strategy, step, force, reg_setup_only=True)

        if "local" == step:
            roi_names = sorted([key
                for key in self.registrations[strategy][step]])
            roi_name = get_option(roi_name, None, roi_names)
            return self.registrations[strategy][step][roi_name]
        else:
            return self.registrations[strategy][step]

    def get_segmentation(self, strategy=None, step=None, reg_step=None,
            force=False):
        strategy = get_option(strategy, self.default_strategy, self.strategies)
        step = get_option(step, self.default_step, self.steps)
        self.segment(strategy, step, force)
        segmentations = self.segmentations[strategy][step]
        reg_steps = list(segmentations)

        if is_list(reg_step):
            assert 1 == len(reg_step)
            reg_step = {roi_name: reg_step[0] for roi_name in self.roi_names}

        if isinstance(reg_step, dict):
            reg_step2 = {}
            for roi_name, roi_reg_step in reg_step.items():
                if self.steps[1] == step:
                    if roi_name in self.roi_pfiles:
                        roi_reg_steps = list(self.roi_pfiles[roi_name])
                    else:
                        roi_reg_steps = list(self.pfiles2_non_null)
                else:
                    roi_reg_steps = reg_steps
                roi_reg_step2 = get_options(roi_reg_step, False, roi_reg_steps)
                assert 1 == len(roi_reg_step2)
                reg_step2[roi_name] = roi_reg_step2[0]

            reg_step2_values = list(set(reg_step2.values()))
            if 1 != len(reg_step2_values):
                ss_merge = StructureSet(name=f"{strategy}_{step}")
                for roi_name, roi_reg_step in reg_step2.items():
                    roi = segmentations[roi_reg_step][roi_name].clone()
                    roi.name = f"{roi_reg_step}_{roi.name}"
                    ss_merge.add_roi(roi)
                ss_merge.set_image(self.im1, add_to_image=False)
                return ss_merge
            else:
                reg_step = reg_step2_values[0]

        if isinstance(reg_step, (int, str)) or reg_step is None:
            reg_step = get_option(reg_step, None, reg_steps)
            return segmentations[reg_step]

    def get_comparison_steps(self, steps):
        if steps is None:
            steps = [self.default_step]
        elif isinstance(steps, int):
            steps = [steps]
        else:
            steps = [step if isinstance(step, str)
                     else get_segmentation_steps[step] for step in steps]

        return steps

    def get_comparison(
            self, id1=None, id2=None, to_keep=None, strategies=None,
            steps=None, reg_steps=None, force=False, metrics=None,
            slice_stats=None, default_by_slice=None, 
            voxel_size=None, name_as_index=False, **kwargs):

        if not hasattr(self, "ss1_filtered"):
            return

        strategies = get_options(
                strategies, self.default_strategy, self.strategies)
        steps = get_options(
                steps, self.default_step, self.steps)
        metrics = get_options(metrics, self.metrics, get_comparison_metrics())
        slice_stats = get_options(slice_stats, self.slice_stats,
                                  get_stat_functions())
        default_by_slice = get_option(default_by_slice, self.default_by_slice,
                                      get_by_slice_methods())
        
        df = None
        ss1 = self.ss1_filtered.filtered_copy(to_keep=to_keep)
        for strategy in strategies:
            for step in steps:
                self.segment(strategy, step, force)
                all_reg_steps = list(self.segmentations[strategy][step])
                step_reg_steps = get_options(reg_steps, None, all_reg_steps)
                for reg_step in step_reg_steps:
                    ss2 = (self.get_segmentation(strategy, step, reg_step)
                           .filtered_copy(to_keep=to_keep))
                    df_tmp = ss1.get_comparison(
                            ss2, metrics=metrics, slice_stats=slice_stats,
                            default_by_slice=default_by_slice,
                            voxel_size=voxel_size, name_as_index=name_as_index,
                            **kwargs)

                    if df_tmp is None:
                        continue

                    for label, value in [
                            ("id1", id1), ("id2", id2), ("strategy", strategy),
                            ("step", step), ("reg_step", reg_step)]:
                        if value is not None:
                            df_tmp[label] = pd.Series(
                                    df_tmp.shape[0] * [value])

                    if df is None:
                        df = df_tmp
                    else:
                        df = pd.concat([df, df_tmp], ignore_index=True)

        return df

    def adjust_reg_files(self, strategies=None, step=None, roi_names=None,
                         params_by_reg_step=None):
        adjustments = {}
        if not params_by_reg_step:
            return adjustments
        strategies = get_options(
                strategies, self.default_strategy, self.strategies)
        step = get_option(step, self.default_step, self.steps)
        roi_names = roi_names or [None]

        for strategy in strategies:
            self.segment(strategy, step, True, reg_setup_only=True)

        for reg_step, params in params_by_reg_step.items():
            adjustments = {**adjustments,
                           **{f"{step}_{reg_step}_{key}": value
                              for key, value in params.items()}}
            for strategy in strategies:
                for roi_name in roi_names:
                    self.get_registration(strategy, step, roi_name, False
                                          ).adjust_file(reg_step, params)

        return adjustments

    def get_reg_steps(self, step=None):
        step = get_option(step, None, self.steps)
        initial_step = [self.initial_transform_name]
        if step == self.steps[0]:
            if not self.initial_alignment:
                initial_step = []
            return (initial_step + list(self.pfiles1))
        else:
            return (initial_step + list(self.pfiles2))

def get_option(opt=None, fallback_opt=None, allowed_opts=None):

    if not is_list(allowed_opts) or not len(allowed_opts):
        raise RuntimeError("No allowed options specified")

    for option in [opt, fallback_opt]:
        if option in allowed_opts:
            return option
        if isinstance(option, int):
            return allowed_opts[option]
        if option is False:
            return ""

    return allowed_opts[-1]

def get_options(opts=None, fallback_opts=None, allowed_opts=None):

    if not is_list(allowed_opts) or not allowed_opts:
        raise RuntimeError("No allowed options specified")

    if opts is True:
        return allowed_opts
    elif opts is False:
        return []
    elif opts in allowed_opts:
        return [opts]
    elif isinstance(opts, int):
        return [allowed_opts[opts]]

    options = []
    if is_list(opts):
        for opt in opts:
            new_opts = get_options(opt, False, allowed_opts)
            for new_opt in new_opts:
                if new_opt not in options:
                    options.append(new_opt)

    if not options and fallback_opts is not None:
        options = get_options(fallback_opts, None, allowed_opts)

    return options or [allowed_opts[-1]]

def get_fixed_and_moving(im1, im2, strategy):
    return (im1, im2) if ("pull" == strategy) else (im2, im1)

def ensure_any(in_val):
    return in_val

def ensure_image(im):
    return (im if im is None else Image(im))

def ensure_structure_set(ss):
    return (ss if ss is None else StructureSet(ss))

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

def get_structure_set_index(ss_index, im):
    if (ss_index < 0 and isinstance(im, Image)
        and len(im.structure_sets) <= abs(ss_index)):
        return len(im.structure_sets) + ss_index
    return ss_index


class SASTuner(Data):

    def __init__(self, pfiles1_variations=None, pfiles2_variations=None,
                 keep_segmentations=False, **kwargs):

        self.pfiles1_variations = pfiles1_variations
        self.pfiles2_variations = pfiles2_variations
        self.keep_segmentations = keep_segmentations
        self.kwargs = kwargs

        self.sas_constant_kwargs = {}
        self.sas_variable_kwargs = {}
        self.comparison_kwargs = {}

        if kwargs:
            sas_parameters = list(signature(SingleAtlasSegmentation).parameters)
            for key, value in kwargs.items():
                if key in sas_parameters:
                    if isinstance(value, list) and len(value) > 1:
                        self.sas_variable_kwargs[key] = value
                    else:
                        self.sas_constant_kwargs[key] = value
                else:
                    self.comparison_kwargs[key] = value

            self.sas_constant_kwargs["auto"] = True
            self.sas_constant_kwargs["auto_reg_setup_only"] = True
            self.sas_constant_kwargs["overwrite"] = True

        if pfiles1_variations:
            reg1_permutations = get_dict_permutations(
                    {reg_step:
                     get_dict_permutations(pfiles1_variations[reg_step])
                     for reg_step in pfiles1_variations})
        else:
            reg1_permutations = [{}]

        if pfiles2_variations:
            reg2_permutations = get_dict_permutations(
                    {reg_step:
                     get_dict_permutations(pfiles2_variations[reg_step])
                     for reg_step in pfiles2_variations})
        else:
            reg2_permutations = [{}]

        sas_permutations = get_dict_permutations(self.sas_variable_kwargs)

        self.df = None
        self.sass = []
        self.adjustments = []

        for sas_permutation in sas_permutations:
            sas = SingleAtlasSegmentation(
                    **self.sas_constant_kwargs, **sas_permutation)
            strategies = get_options(
                    sas.auto_strategies, sas.default_strategy, sas.strategies)
            for reg1_permutation in reg1_permutations:
                reg1_adjustments = sas.adjust_reg_files(
                        strategies, sas.steps[0], None, reg1_permutation)
                for reg2_permutation in reg2_permutations:
                    reg2_adjustments = sas.adjust_reg_files(
                            strategies, sas.steps[1],
                            sas.roi_names, reg2_permutation)
                    adjustments = {**sas_permutation, **reg1_adjustments,
                                   **reg2_adjustments}

                    df_comparison = sas.get_comparison(**self.comparison_kwargs)

                    self.adjustments.append(deepcopy(adjustments))
                    if keep_segmentations:
                        self.sass.append(sas.clone())

                    if adjustments:
                        df_permutation = pd.DataFrame(
                                max(1, df_comparison.shape[0]) * [adjustments])
                    else:
                        df_permutation = None

                    if df_permutation is None and df_comparison is None:
                        continue
                    elif df_permutation is None:
                        df_sas = df_comparison
                    elif df_comparison is None:
                        df_sas = df_permutation
                    else:
                        df_sas = pd.concat(
                                [df_permutation, df_comparison], axis=1)

                    if self.df is None:
                        self.df = df_sas
                    else:
                        self.df = pd.concat(
                                [self.df, df_sas], ignore_index=True)
