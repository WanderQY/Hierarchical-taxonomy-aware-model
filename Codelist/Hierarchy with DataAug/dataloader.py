import os
import random
import torch
import numpy as np
from config import *
from class_labels import RARE, RARE0
from utils import get_feature
from torch.utils.data import Dataset
import sys
import kaldiio
from audiomentations import TanhDistortion
sys.path.append('E:/Work/BirdCLEF2017/')
random.seed(1234)

### time domain ###

def distortion_augment(wave_data, p=0.5, sr=22050):
    distortion = TanhDistortion(min_distortion=0.3, max_distortion=0.5, p=p)
    return distortion(samples=wave_data, sample_rate=sr)

def noise_augment(signal_feat, noise_file_list, p=0.5, sr=22050):
    if random.random() <= p:
        wave_data_mix = kaldiio.load_mat(random.choice(noise_file_list))
        duration = len(wave_data_mix) / sr
        start = max(random.uniform(0, duration - 5), 0.0)
        end = min(start + 5.0, duration)
        wave_data_mix = wave_data_mix[int(start * sr): int(end * sr)]
        noise_spec = get_feature(wave_data_mix, sr, frame_len=1024, win_step=1 / 4, n_mels=256)
        lam = np.random.beta(0.5, 0.5)
        mix_feat = lam * noise_spec + (1-lam) * signal_feat
        return mix_feat
    else:
        return signal_feat

def cut_mix(wave_data, candidate_list, p=0.5, sr=22050):
    if random.random() <= p:
        new_wave_data = wave_data.copy()
        wave_data_mix = kaldiio.load_mat(random.choice(candidate_list))
        duration1 = len(wave_data) / sr
        start1 = max(random.uniform(0, duration1 - 2), 0.0)
        end1 = min(start1 + 2.0, duration1)
        duration2 = min(len(wave_data_mix) / sr, end1 - start1)
        start2 = max(random.uniform(0, duration2 - 2), 0.0)
        end2 = min(start1 + 2.0, duration2)
        duration = int(min(end2 - start2, end1 - start1))
        new_wave_data[int(start1 * sr): int((start1+duration) * sr)] = wave_data_mix[int(start2 * sr): int((start2+duration) * sr)]
        return new_wave_data
    else:
        return wave_data

class BirdsoundData(Dataset):
    """
    鸟语识别任务的鸟类音频数据
    :param dataset: {文件名: {'ark_path': wave_data, 'label': class_name}}
    """
    def __init__(self, dataset, class_list, option='train', augment=[]):
        self.option = option
        self.id = list(dataset.keys())
        #self.path = [dataset[file]['path'] for file in self.id]
        self.path = [dataset[file]['ark_path'] for file in self.id]
        self.duration = [dataset[file]['duration'] for file in self.id]
        species_list, genus_list, family_list, order_list = class_list
        self.label = torch.as_tensor(np.array([[species_list.index(dataset[spe]['label'][0]),
                                                genus_list.index(dataset[spe]['label'][1]),
                                                family_list.index(dataset[spe]['label'][2]),
                                                order_list.index(dataset[spe]['label'][3])] for spe in self.id]),
                                     dtype=torch.int64)
        self.sr = SAMPLE_RATE
        self.augment = augment
        if self.augment != []:
            self.p = []
            for spe in self.id:
                if dataset[spe]['label'] in RARE0:
                    self.p.append(0.75)
                elif dataset[spe]['label'] in RARE:
                    self.p.append(0.5)
                else:
                    self.p.append(0)
        if 'noise' in self.augment:
            lines = []
            with open('E:/Work/BirdCLEF2017/Feature/noise_datas.scp', 'r') as T:
                lines += T.readlines()
            path_dict = {}
            for line in lines:
                idx, path = line.split()
                path_dict[idx] = path
            self.noise_file_list = list(path_dict.values())
        if 'cut_mix' in self.augment:
            self.candidate_list = {}
            for i in range(len(self.id)):
                if str(self.label[i][0]) not in self.candidate_list:
                    self.candidate_list[str(self.label[i][0])] = []
                self.candidate_list[str(self.label[i][0])].append(self.path[i])

    def __getitem__(self, idx):
        path_idx = self.path[idx]
        if self.option == 'train':
            wave_data = kaldiio.load_mat(path_idx)
            start = max(random.uniform(0, self.duration[idx]-5), 0.0)
            end = min(start + 5.0, self.duration[idx])
            wave_data = wave_data[int(start*self.sr): int(end*self.sr)]
            if 'distortion' in self.augment:
                wave_data = distortion_augment(wave_data, p=0.25)

            if 'cut_mix' in self.augment:
                wave_data = cut_mix(wave_data, self.candidate_list[str(self.label[idx][0])], p=self.p[idx], sr=self.sr)

            feat = get_feature(wave_data, self.sr, frame_len=FRAME_LEN, win_step=1/4, n_mels=N_MELS)
            if 'noise' in self.augment:
                feat = noise_augment(feat, self.noise_file_list, p=self.p[idx])

        elif self.option == 'test':
            #feat, _ = librosa.load(path_idx, None)
            feat = kaldiio.load_mat(path_idx)
        else:
            raise ValueError("Invalid option!")
        return self.id[idx], feat, self.label[idx], self.duration[idx]

    def __len__(self):
        num_spe = len(self.id)
        return num_spe