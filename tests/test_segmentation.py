"""Test the skrt.segmentation module."""

import pytest

from skrt import Image, StructureSet
from skrt.core import Defaults
from skrt.segmentation import (
        ensure_image,
        ensure_structure_set,
        get_contour_propagation_strategies,
        get_fixed_and_moving,
        get_option,
        get_segmentation_steps,
        get_steps,
        get_structure_set_index,
        MultiAtlasSegmentation,
        SingleAtlasSegmentation,
        SasTuner,
        )

# Check for elastix executable
try:
    subprocess.check_output("elastix")
    has_elastix = True
except:
    has_elastix = False

# Decorator for tests requiring elastix functionality
def needs_elastix(func):
    def wrapper():
        if not has_elastix:
            return
        else:
            func()
    return wrapper

global_workdir = "tmp/segmentation_workdir"

def test_sas_instantiation():
    """Test instantiation of SingleAtlasSegmentation."""

    sas = SingleAtlasSegmentation()

    # Check that strategies and steps are defined.
    assert len(sas.strategies) > 0
    assert len(sas.steps) > 0

    # Check that dictionaries for storing registrations and segmentations
    # have been defined.
    for strategy in sas.strategies:
        for step in sas.steps:
            assert isinstance(sas.registrations[strategy][step], dict)
            assert isinstance(sas.segmentations[strategy][step], dict)

def test_mas_instantiation():
    """Test instantiation of MultiAtlasSegmentation."""

    mas = MultiAtlasSegmentation()

    # Check that dictionary for storing single-atlas segmentations
    # and consensus results have been defined.
    assert isinstance(mas.sass, dict)
    for strategy in get_contour_propagation_strategies():
        for step in get_segmentation_steps():
            assert isinstance(mas.consensuses[strategy][step], dict)

def test_sas_tuner_instantiation():
    """Test instantiation of SasTuner."""

    tuner = SasTuner()

    # Check that kwargs and df attributes are null.
    assert tuner.kwargs == {}
    assert tuner.df is None

def test_ensure_image():
    """Test that ensure_image() returns Image or None."""
    assert ensure_image(None) is None
    assert isinstance(ensure_image(Image()), Image)

def test_ensure_structure_set():
    """Test that ensure_structure_set() returns StructureSet or None."""
    assert ensure_structure_set(None) is None
    assert isinstance(ensure_structure_set(StructureSet()), StructureSet)

def test_get_contour_propagation_strategies():
    """
    Test that get_contour_propagation_strategies() returns list of strings.
    """
    local_engines = {"niftyreg": 1, "elastix": 2}
    for engine, n_strategy in local_engines.items():
        for engine_arg in [engine, None]:
            Defaults().registration_engine = engine
            strategies = get_contour_propagation_strategies(engine_arg)
            assert isinstance(strategies, list)
            assert len(strategies) == n_strategy
            for strategy in strategies:
                assert isinstance(strategy, str)

def test_get_fixed_and_moving():
    """Test image assignment as fixed or moving, depending on strategy"""
    # There's no check of object type, so test with strings.
    images = ["im1", "im2"]
    assert get_fixed_and_moving(*images, "pull") == tuple(images)
    assert get_fixed_and_moving(*images, "push") == tuple(sorted(
        images, reverse=True))

def test_get_option():
    """Test option retrieval."""

    # Check that exception is raised if no allowed options specified.
    with pytest.raises(RuntimeError) as error_info:
        get_option()
    assert "No allowed options specified" in str(error_info.value)

    # Check that returned option defaults to last-listed allowed option.
    allowed_opts = ["a", "b", "c"]
    assert get_option(allowed_opts=allowed_opts) == allowed_opts[-1]

    # Check that allowed input option, specified by value or index, is returned.
    for idx1, allowed in enumerate(allowed_opts):
        assert get_option(opt=allowed, allowed_opts=allowed_opts) == allowed
        assert get_option(opt=idx1, allowed_opts=allowed_opts) == allowed
        idx2 = idx1 - len(allowed_opts)
        assert get_option(opt=idx2, allowed_opts=allowed_opts) == allowed

    # Check that out-of-range index raises exception.
    with pytest.raises(IndexError) as error_info:
        get_option(opt=len(allowed_opts), allowed_opts=allowed_opts)
        get_option()
    assert "index out of range" in str(error_info.value)

    # Check that fallback option is returned when input option is invalid.
    invalid_opts = ["d", "e", "f"]
    for invalid, allowed in zip(invalid_opts, allowed_opts):
        assert get_option(opt=invalid, fallback_opt=allowed,
                allowed_opts=allowed_opts) == allowed

def test_get_segmentation_steps():
    """Test retrieval of segmentation steps."""
    seg_steps = get_segmentation_steps()
    assert isinstance(seg_steps, list)
    assert len(seg_steps) > 0

def test_get_steps():
    """Test retrieval of steps to be run."""
    seg_steps = get_segmentation_steps()
    assert get_steps(None) == seg_steps
    for idx, name in enumerate(seg_steps):
        n_step = idx + 1

        for step in [idx, name, [idx], [name]]:
            steps = get_steps(step)
            assert len(steps) == n_step
            for idx2 in range(n_step):
                assert seg_steps[idx2] in steps
