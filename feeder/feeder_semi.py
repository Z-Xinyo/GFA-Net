# sys
import pickle

# torch
import torch

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import ceil

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.input_representation = input_representation
        self.l_ratio = l_ratio

        self.load_data(mmap)

        labels_arr = np.array(self.label)
        unique_classes = np.unique(labels_arr)
        keep_indices = []
        for cls in unique_classes:
            idx = np.where(labels_arr == cls)[0]

            # 1%
            k = max(1, ceil(len(idx) * 0.01))
            if k >= len(idx):
                chosen = idx.copy()
            else:
                chosen = np.random.choice(idx, size=k, replace=False)
            keep_indices.extend(chosen.tolist())

            #     #10%
            # if len(idx) < 10:
            #     chosen = idx
            # else:
            #     k = max(1, int(len(idx) * 0.1))
            #     chosen = np.random.choice(idx, size=k, replace=False)
            # keep_indices.extend(chosen.tolist())

        keep_indices = sorted(keep_indices)
        if len(keep_indices) == 0:
            raise ValueError("筛选后没有样本，请检查数据或调整保留比例。")


        self.data = np.array(self.data)[keep_indices]
        self.number_of_frames = np.array(self.number_of_frames)[keep_indices]

        if isinstance(self.sample_name, (list, tuple, np.ndarray)):
            self.sample_name = [self.sample_name[i] for i in keep_indices]

        self.label = [labels_arr[i] for i in keep_indices]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        print(self.data.shape, len(self.number_of_frames), len(self.label))
        print("l_ratio", self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)

        self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

        #load label
        if '.pkl' in self.label_path:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        elif '.npy' in self.label_path:
            self.label = np.load(self.label_path).tolist()

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]
        label = self.label[index]

        # # crop a sub-sequnce  //todo
        data_numpy = augmentations.crop_subsequence(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        if self.input_representation == "motion":
            # motion
            motion = np.zeros_like(data_numpy)
            motion[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]

            data_numpy = motion

        elif self.input_representation == "bone":
            # bone
            bone = np.zeros_like(data_numpy)
            for v1, v2 in self.Bone:
                bone[:, :, v1 - 1, :] = data_numpy[:, :, v1 - 1, :] - data_numpy[:, :, v2 - 1, :]

            data_numpy = bone

        return data_numpy, label