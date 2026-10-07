import glob
import os
import pickle

import numpy as np
import torch
from numpy.random import randint
from PIL import Image
from torch.utils import data

from dataloader.video_transform import (
    GroupRandomSizedCrop, GroupMildFaceCrop, GroupRandomHorizontalFlip, GroupResize,
    Stack, ToTorchFormatTensor,
)
import torchvision.transforms


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def DB_index(self):
        return self._data[3]


class AIDE_Dataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,
                 contrast=False, DBtransform=None, t=3, db_pkl_path=None):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.DBtransform = DBtransform
        self.image_size = image_size
        self.mode = mode
        self.contrast = contrast
        self._parse_list()
        self.DB_feature = pickle.load(open(db_pkl_path, 'rb'))
        first_val = next(iter(self.DB_feature.values()))
        self._db_shape = first_val.shape  # e.g. (210, 45)

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % len(self.video_list))

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = (np.multiply(list(range(self.num_segments)), average_duration)
                       + randint(average_duration, size=self.num_segments))
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1,
                                      size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def get_DB(self, record):
        feature = self.DB_feature[record.DB_index]
        ts = torch.tensor(feature).float()
        if self.mode == 'train' and self.DBtransform is not None:
            ts = self.DBtransform(ts)
        return ts

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        weight = torch.tensor(1.0)
        if self.contrast == 'V-DB':
            return self.get_DB(record), self.get(record, segment_indices), weight
        elif self.contrast == 'V':
            return torch.zeros(self._db_shape), self.get(record, segment_indices), weight
        elif self.contrast == 'DB':
            return self.get_DB(record), (torch.zeros((1, 3, 1, 1)), record.label), weight

    def get(self, record, indices):
        # AIDE frames are named 0.jpg, 1.jpg, ..., 44.jpg — sort numerically
        video_frames_path = sorted(
            glob.glob(os.path.join(record.path, '*.jpg')),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.duration):
                images.append(Image.open(video_frames_path[p]).convert('RGB'))
                if p < record.num_frames - 1:
                    p += 1
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        return images, record.label

    def __len__(self):
        return len(self.video_list)


class TimeSeriesAugmentation(object):
    def __init__(self, noise_A=0.1, origin_rate=4):
        self.noise_A = noise_A
        self.origin_rate = origin_rate
        self.i = 0

    def __call__(self, DB_ts):
        self.i += 1
        if self.i % self.origin_rate == 0:
            return DB_ts
        return DB_ts + torch.randn_like(DB_ts) * self.noise_A


def DECNet_train_data_loader(data_set, txt_path, contrast=False, noise_A=0.1,
                              origin_rate=4, t=3, db_pkl_path=None, mild_face_crop=False):
    image_size = 112
    train_transforms = torchvision.transforms.Compose([
        GroupMildFaceCrop(image_size) if mild_face_crop else GroupRandomSizedCrop(image_size),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor(),
    ])
    return AIDE_Dataset(
        list_file=txt_path,
        num_segments=8,
        duration=2,
        mode='train',
        transform=train_transforms,
        image_size=image_size,
        contrast=contrast,
        DBtransform=TimeSeriesAugmentation(noise_A=noise_A, origin_rate=origin_rate),
        t=t,
        db_pkl_path=db_pkl_path,
    )


def DECNet_test_data_loader(data_set, txt_path, contrast=False, t=3, db_pkl_path=None):
    image_size = 112
    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor(),
    ])
    return AIDE_Dataset(
        list_file=txt_path,
        num_segments=8,
        duration=2,
        mode='test',
        transform=test_transform,
        image_size=image_size,
        contrast=contrast,
        t=t,
        db_pkl_path=db_pkl_path,
    )
