import os.path

import numpy as np
import librosa
import pylab
import torch
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def get_feature(wave_data, sr=22050, frame_len=1024, n_fft=None, win_step=1/4, window="hamming", preemph=0.97, n_mels=256,
                replace_energy=True):
    """
    获取声谱图（语谱图）特征，fbank系数
    :param wave_data: 输入音频数据
    :param sr: 所输入音频文件的采样率，默认为 32kHz
    :param frame_len: 帧长，默认2048个采样点(64ms,32kHz),与窗长相同
    :param n_fft: FFT窗口的长度，默认与窗长相同
    :param win_step: 窗移，默认移动 2/3，2048*2/3=1365个采样点 (42.7ms,32kHz)
    :param window: 窗类型，默认汉明窗
    :param preemph: 预加重系数，默认0.97
    :param n_mels: Mel滤波器组的滤波器数量，默认64
    :param replace_energy: 是否将第0阶倒谱系数替换成帧能量的对数，默认替换
    :return: n_fbank*3维特征，每一列为一个fbank特征向量 np.ndarray[shape=(n_fbank*3, n_frames), dtype=float32]
    """
    wave_data = librosa.effects.preemphasis(wave_data, coef=preemph)  # 预加重，系数0.97
    window_len = frame_len  # 窗长2048
    if n_fft is None:
        fft_num = window_len  # 设置NFFT点数与窗长相等
    else:
        fft_num = n_fft
    hop_length = round(window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2），即帧移(窗移)2/3
    mag_spec = np.abs(librosa.stft(wave_data, n_fft=fft_num, hop_length=hop_length,
                                    win_length=window_len, window=window))
    # 每帧内所有采样点的幅值平方和作为能量值，np.ndarray[shape = (1，n_frames), dtype = float64]
    pow_spec = np.square(mag_spec)
    energy = np.sum(pow_spec, axis=0)
    energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
    # 频谱矩阵：行数=n_mels=64，列数=帧数n_frames=全部采样点数/(2048*2/3)+1（向上取整）
    # 快速傅里叶变化+汉明窗, Mel滤波器组的滤波器数量 = 64
    mel_spec = librosa.feature.melspectrogram(wave_data, sr, n_fft=fft_num, hop_length=hop_length,
                                              win_length=window_len, window=window, n_mels=n_mels)
    fbank = librosa.power_to_db(mel_spec)  # 转换为log尺度
    if replace_energy:
        fbank[0, :] = np.log(energy)  # 将第0个系数替换成对数能量值
    fbank = min_max_scaler.fit_transform(fbank)
    fbank = np.pad(fbank, pad_width=((0, 0), (0, 431 - np.shape(fbank)[1])), mode="constant")

    return fbank

def speech_seg(wave_data, duration, sr=22050, max_len=5, min_len=5, overlap=0):
    """
    分割规则：原始音频不足 3s 进行自我拼接，按 5s 进行音频切割，剩余部分不足 3s 丢弃。
    :param voice_path: 输入音频信号
    :param sr: 所输入音频文件的采样率，默认为 32kHz
    :param max_len: 最大切割长度
    :param min_len: 最小切割长度
    :param overlap: 重叠时间长度
    :return: voice_seg_list, 切割后音频数据 list
    """

    # 音频分割
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
sys.path.append('E:/Work/BirdCLEF2017/')
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
