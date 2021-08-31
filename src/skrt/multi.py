'''Class for iterating over multiple patients.'''

import glob
import os

from skrt.patient import Patient


class PatientDataset:

    def __init__(self, paths=[]):

        # Find paths
        if isinstance(paths, str):
            paths = [paths]
        self.paths = []
        for p in paths:
            self.paths.extend(glob.glob(p))

    def add_path(self, path):
        self.paths.extend(glob.glob(path))

    def __iter__(self):
        return PatientIterator(self)

    def __getitem__(self, idx):
        return PatientDataset(self.paths[idx])


class PatientIterator:

    def __init__(self, dataset):
        self.idx = -1
        self.dataset = dataset

    def __next__(self):
        self.idx += 1
        if self.idx < len(self.dataset.paths):
            return Patient(self.dataset.paths[self.idx])
        raise StopIteration

