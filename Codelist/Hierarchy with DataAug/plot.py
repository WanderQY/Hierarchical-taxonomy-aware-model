import os.path
import numpy as np
import librosa
import pylab
import torch
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def get_feature(wave_data, sr=22050, frame_len=1024, n_fft=None, win_step=1/4, window="hamming", preemph=0.97, n_mels=256,
                replace_energy=True):
    wave_data = librosa.effects.preemphasis(wave_data, coef=preemph)
    window_len = frame_len 
    if n_fft is None:
        fft_num = window_len 
    else:
        fft_num = n_fft
    hop_length = round(window_len * win_step) 
    mag_spec = np.abs(librosa.stft(wave_data, n_fft=fft_num, hop_length=hop_length,
                                    win_length=window_len, window=window))
   
    pow_spec = np.square(mag_spec)
    energy = np.sum(pow_spec, axis=0)
    energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  
    mel_spec = librosa.feature.melspectrogram(wave_data, sr, n_fft=fft_num, hop_length=hop_length,
                                              win_length=window_len, window=window, n_mels=n_mels)
    fbank = librosa.power_to_db(mel_spec)
    if replace_energy:
        fbank[0, :] = np.log(energy) 
    fbank = min_max_scaler.fit_transform(fbank)
    fbank = np.pad(fbank, pad_width=((0, 0), (0, 431 - np.shape(fbank)[1])), mode="constant")

    return fbank

def speech_seg(wave_data, duration, sr=22050, max_len=5, min_len=5, overlap=0):
    voice_seg_list = []

    while duration < min_len:
        wave_data = np.append(wave_data, wave_data)
        duration = duration * 2

    start_index = 0
    end_index = start_index + max_len
    while start_index + max_len <= duration:
        seg_i = wave_data[start_index * sr: end_index * sr]
        voice_seg_list.append(seg_i)
        start_index = end_index - overlap
        end_index = start_index + max_len
    res = wave_data[start_index * sr:]
    if duration-start_index >= min_len:  # save more than 3s
        voice_seg_list.append(res)
    return voice_seg_list

import librosa.display
import sys
sys.path.append('../../BirdCLEF/')
select = ['3_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN11809', '5_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN11309', '3_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN1762', '5_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN18164', '5_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN24948', '3_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN76', '2_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN28282', '4_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN27030', '3_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN11520', '1_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN20136', '1_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN12994', '1_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN26985', '3_LIFECLEF2017_BIRD_XC_WAV_RN43962', '3_LIFECLEF2017_BIRD_XC_WAV_RN35400', '1_LIFECLEF2017_BIRD_XC_WAV_RN37132', '1_LIFECLEF2017_BIRD_XC_WAV_RN49447', '2_LIFECLEF2017_BIRD_XC_WAV_RN34744', '2_LIFECLEF2017_BIRD_XC_WAV_RN34304', '1_LIFECLEF2017_BIRD_XC_WAV_RN39357']
wave_dir = sys.path[-1] + 'SortedData/Song_22050'
save_dir = sys.path[-1] + 'SortedData/img'
for f in select:
    path = os.path.join(wave_dir, f + '.wav')
    wave_data, _ = librosa.load(path, 22050)
    duration = librosa.get_duration(filename=path)
    voice_seg_list = speech_seg(wave_data, duration)
    for n, x in enumerate(voice_seg_list):
        feat = get_feature(x)
        save_path = os.path.join(save_dir, f + '-{}.png'.format(n))
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        librosa.display.specshow(feat)
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
