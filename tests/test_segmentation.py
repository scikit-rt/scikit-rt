"""Test the skrt.segmentation module."""

import pytest

from skrt.segmentation import (
        get_contour_propagation_strategies,
        get_fixed_and_moving,
        get_option,
        get_segmentation_steps,
        MultiAtlasSegmentation,
        SingleAtlasSegmentation,
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
