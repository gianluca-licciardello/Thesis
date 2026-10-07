import os.path
from typing import Any
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
import pickle


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

    @property
    def weight(self):
        return float(self._data[4]) if len(self._data) > 4 else 1.0


class PPB_Dataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, contrast=False, DBtransform=None, t=3, db_pkl_path=None):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.DBtransform = DBtransform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        self.contrast = contrast
        if db_pkl_path is None:
            raise ValueError("Provide --db_pkl_path for the prepared feature pickle")
        self.DB_feature = pickle.load(open(db_pkl_path, 'rb'))
        first_val = next(iter(self.DB_feature.values()))
        self._db_shape = first_val.shape  # e.g. (8, T) for CAN-bus, (18, T) for keypoints

        # Drop samples whose DB key is absent from the pickle (e.g. MeTRAbs
        # produced fewer frames for a clip than the annotation expects).
        if contrast in ('V-DB', 'DB'):
            before = len(self.video_list)
            self.video_list = [r for r in self.video_list
                               if r.DB_index in self.DB_feature]
            dropped = before - len(self.video_list)
            if dropped:
                print(f'[WARN] Dropped {dropped} sample(s) with no matching DB key in pickle.')

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1]) >= 16]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def get_DB(self, record):
        tsfresh_feature_filtered_df = self.DB_feature[record.DB_index]
        tsfresh_feature = torch.tensor(tsfresh_feature_filtered_df).float()
        if self.mode == 'train':
            tsfresh_feature = self.DBtransform(tsfresh_feature)   #数据增强
        return tsfresh_feature
    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)


        weight = torch.tensor(record.weight, dtype=torch.float32)
        if self.contrast == "V-DB":
            DB_feature = self.get_DB(record)
            return DB_feature, self.get(record, segment_indices), weight
        elif self.contrast == "V":
            DB_feature = torch.zeros(self._db_shape)
            return DB_feature, self.get(record, segment_indices), weight
        elif self.contrast == 'DB':
            DB_feature = self.get_DB(record)
            return DB_feature, (torch.zeros((1, 3, 1, 1)), record.label), weight

    def get(self, record, indices):
        video_name = record.path.split('/')[-1]
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path.sort()

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        return images, record.label

    def __len__(self):
        return len(self.video_list)



class TimeSeriesAugmentation(object):
    def __init__(self, noise_A=11, origin_rate=4):
        self.noise_A = noise_A
        self.origin_rate = origin_rate
        self.i = 0
    def __call__(self, DB_ts):
        noise = torch.randn_like(DB_ts) * self.noise_A  # Gaussian (paper §IV.A)
        self.i += 1
        if self.i % self.origin_rate == 0:
            return DB_ts
        else:
            return DB_ts + noise


def DECNet_train_data_loader(data_set, txt_path, contrast=False, noise_A=5, origin_rate=4, t=3, db_pkl_path=None, mild_face_crop=False):
    image_size = 112
    train_transforms = torchvision.transforms.Compose([GroupMildFaceCrop(image_size) if mild_face_crop else GroupRandomSizedCrop(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])

    train_data = PPB_Dataset(list_file=txt_path,
                              num_segments=8,
                              duration=2,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              contrast=contrast,
                              DBtransform=TimeSeriesAugmentation(noise_A=noise_A, origin_rate=origin_rate),
                              t=t,
                              db_pkl_path=db_pkl_path)
    return train_data


def DECNet_test_data_loader(data_set, txt_path, contrast=False, t=3, db_pkl_path=None):
    image_size = 112
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = PPB_Dataset(list_file=txt_path,
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             contrast=contrast,
                             t=t,
                             db_pkl_path=db_pkl_path)
    return test_data
