import numpy as np
import skimage.transform as skt
from imgaug import augmenters as iaa
import imgaug as ia


class DataAugmentGenerator(object):

    def __init__(self,
                 X_train,
                 Y_train,
                 batchsize,
                 flip_indices,
                 flip_ratio,      # 0.5
                 rotate_ratio,    # 0.5
                 contrast_ratio,
                 perspective_transform_ratio,
                 elastic_transform_ratio): # 0.5
        self.X_train = X_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.flip_indices = flip_indices
        self.rotate_ratio = rotate_ratio
        self.contrast_ratio = contrast_ratio
        self.perspective_transform_ratio = perspective_transform_ratio
        self.elastic_transform_ratio = elastic_transform_ratio
    
    def _random_indices(self, ratio):
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    def flip(self):
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices, :, ::-1, :]
        self.targets[indices, ::2] = self.targets[indices, ::2] * -1
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = self.targets[indices, b], self.targets[indices, a]

    def rotate(self):
        indices = self._random_indices(self.rotate_ratio)
        angle = np.random.randint(-10, 10)
        for i in indices:
            self.inputs[i] = skt.rotate(self.inputs[i, :, :, :1], angle)
        angle = np.radians(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.targets = self.targets.reshape(len(self.targets), int(self.Y_train.shape[1]/2), 2)
        self.targets[indices] = np.dot(self.targets[indices], R)
        self.targets = self.targets.reshape(len(self.targets), self.Y_train.shape[1])
        self.targets = np.clip(self.targets, -1, 1)

    def contrast(self):
        indices = self._random_indices(self.contrast_ratio)
        delta = np.random.uniform(0.8, 1.2)
        self.inputs[indices] = (delta*self.inputs[indices, :, :, :]) + (1-delta)*np.mean(self.inputs[indices, :, :, :])

    def perspective_transform(self):
        indices = self._random_indices(self.perspective_transform_ratio)
        keypoints_on_images = []
        for target in self.targets[indices]:
            keypoints = []
            for i in range(0, len(target), 2):
                x = target[i]*48 + 48
                y = target[i+1]*48 + 48
                keypoints.append(ia.Keypoint(x=x, y=y))
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=(96, 96, 1)))
        seq = iaa.Sequential([
            iaa.PerspectiveTransform(scale=(0.01, 0.1))
        ])
        seq_det = seq.to_deterministic()
        self.inputs[indices] = (seq_det.augment_images(self.inputs[indices, :, :, :]*48.0+48.0)-48.0)/48.0
        keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
        normalized_keypoints_on_images = []
        for keypoints_on_image in keypoints_aug:
            normalized_keypoints = []
            for keypoint in keypoints_on_image.get_coords_array():
                normalized_x, normalized_y = (keypoint[0]-48)/48, (keypoint[1]-48)/48
                normalized_keypoints.append(normalized_x)
                normalized_keypoints.append(normalized_y)
            normalized_keypoints_on_images.append(normalized_keypoints)
        self.targets[indices] = np.array(normalized_keypoints_on_images)



    def elastic_transform(self):
        indices = self._random_indices(self.elastic_transform_ratio)
        seq = iaa.Sequential([
            iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.2)
        ])
        self.inputs[indices] = seq.augment_images(self.inputs[indices, :, :, :]*255)/255


    def generate(self, batchsize, flip=True, rotate=True, contrast=True, perspective_transform=True, elastic_transform=True):
        while True:
            batches = [(start, min(start+self.batchsize, self.size_train)) for start in range(0, self.size_train, self.batchsize)]
            for start, end in batches:
                self.inputs = self.X_train[start:end].copy()
                self.targets = self.Y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]
                self.flip() if flip else None
                self.rotate() if rotate else None
                self.contrast() if contrast else None
                self.perspective_transform() if perspective_transform else None
                self.elastic_transform() if elastic_transform else None
                yield (self.inputs, self.targets)
                