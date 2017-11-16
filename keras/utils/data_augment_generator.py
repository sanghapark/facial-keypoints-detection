import numpy as np
import skimage.transform as skt
from utils.enum import dataset_type


class DataAugmentGenerator(object):

    def __init__(self,
                 X_train,
                 Y_train,
                 batchsize,
                 flip_indices,
                 flit_ratio,      # 0.5
                 rotate_ratio,    # 0.5
                 contrast_ratio): # 0.5
        self.X_train = X_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.flip_indices = flip_indices
        self.rotate_ratio = rotate_ratio
        self.constrast_ratio = constrast_ratio
    
    def _random_indices(self, ratio):
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    def flip(self):
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices, :, :, ::-1]
        self.targets[indices, ::2] = self.targets[indices, ::2] * -1
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = self.targets[indices, b], self.targets[indices, a]

    def rotate(self):
        indices = self._random_indices(self.rotate_ratio)
        angle = np.random.randint(-10, 10)
        for i in indices:
            self.inputs[i] = sk.rotate(self.inputs[i, 0, :, :], angle)
        angle = np.radians(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.targets = self.targets.reshape(len(self.targets), self.Y_train.shape[1] / 2, 2)
        self.targets[indices] = np.dot(self.targets[indices], R)
        self.targets = self.targets.reshape(len(self.targets), self.Y_train.shape[1])
        self.targets = np.clip(self.targets, -1, 1)

    def contrast(self):
        indices = self._random_indices(self.constrast_ratio)
        delta = np.random.uniform(0.8, 1.2)
        self.inputs[indices] = (delta*self.inputs[indices, :, :, :]) + (1-delta)*np.mean(self.inputs[indices, :, :, :])


    def generate(self, batchsize, flip=True, rotate=True, contrast=True):
        while True:
            batches = [(start, min(start+self.batchsize, self.size_train)) for start in range(0, self.size_train, self.batchsize)]
            for start, end in batches:
                self.inputs = self.X_train[start:end].copy()
                self.targets = self.Y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]
                self.flip() if flip else None
                self.rotate() if rotate else None
                self.contrast() if contrast else None
                yield (self.inputs, self.targets)