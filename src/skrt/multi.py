"""Classes for creating, and iterating over, multi-patient datasets."""

import glob
import os
from pathlib import Path

from skrt.core import Data
from skrt.patient import Patient


class PatientDataset(Data):
    """
    Class for storing, and iterating over, paths to patient directories.
    """
    def __init__(self, paths=None, **kwargs):
        """
        Create container for paths.

        **Parameters:**

        paths: list, default=None
            List of paths to patient directories - may contain wildcards.

        kwargs: dict
            Dictionary of keyword arguments to be passed to Patient
            constructor, called when iterating over paths.
        """
        # Store paths.
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        self.paths = []
        if paths:
            for p in paths:
                self.paths.extend(glob.glob(str(p)))

        # Store keyword arguments for Patient constructor.
        self.kwargs = kwargs

    def add_path(self, path):
        """Add to list of paths to patient directories."""
        self.paths.extend(glob.glob(str(path)))

    def __iter__(self):
        """Enable iteration over dataset."""
        return PatientIterator(self)


class PatientIterator:
    """
    Class for iterating over PatientDataset object.
    """

    def __init__(self, dataset):
        """Initialise counter and dataset."""
        self.idx = -1
        self.dataset = dataset

    def __next__(self):
        """Instantiate and return next Patient."""
        self.idx += 1
        if self.idx < len(self.dataset.paths):
            return Patient(self.dataset.paths[self.idx], **self.dataset.kwargs)
        raise StopIteration
