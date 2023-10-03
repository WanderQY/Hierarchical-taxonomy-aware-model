import sys
import subprocess
import wave
import shutil
import scipy
from scipy.io import wavfile
from skimage import morphology
import skimage.filters as filters
import os

def get_basename_without_ext(filepath):
    basename = os.path.basename(filepath).split(os.extsep)[0]
    return basename

def replace_path(split_datas, replace_dir):
    for k in split_datas.keys():
        for spe in split_datas[k].keys():
            split_datas[k][spe]['path'] = split_datas[k][spe]['path'].replace('E:/Work/BirdCLEF2017/', replace_dir)
    return split_datas

def parse_datasets(split_datas, add_birdsonly=True, replace_dir=False):
    if replace_dir:
        for k in split_datas.keys():
            for d in split_datas[k].keys():
                for f in split_datas[k][d].keys():
                    split_datas[k][d][f]['ark_path'] = split_datas[k][d][f]['ark_path'].replace('E:/Work/BirdCLEF2017/',
                                                                                                replace_dir)
                    split_datas[k][d][f]['path'] = split_datas[k][d][f]['path'].replace('E:/Work/BirdCLEF2017/',
                                                                                        replace_dir)
    train_datas, valid_datas = split_datas['origin']['train'],  split_datas['origin']['valid']
    if add_birdsonly:
        for k, v in split_datas['birdsonly']['train'].items():
            train_datas[k+'_birdsonly'] = v

    return train_datas, valid_datas

def play_wave_file(filename):
    """ Play a wave file
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")
    else:
        if (sys.platform == "linux" or sys.playform == "linux2"):
            subprocess.call(["aplay", filename])
        else:
            print("Platform not supported")

def write_wave_to_file(filename, rate, wave):
    wavfile.write(filename, rate, wave)


def read_wave_file(filename, normalize=True):
    """ Read a wave file from disk
    # Arguments
        filename : the name of the wave file
    # Returns
        (fs, x)  : (sampling frequency, signal)
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")

    s = wave.open(filename, 'rb')

    if (s.getnchannels() != 1):
        raise ValueError("Wave file should be mono")

    strsig = s.readframes(s.getnframes())
    x = np.fromstring(strsig, np.short)
    fs = s.getframerate()
    s.close()
    if normalize:
        x = x/32768.0

    return fs, x

def copy_subset(root_dir, classes, subset_dir):
    # create directories
    if not os.path.exists(subset_dir):
        print("os.makedirs("+subset_dir+")")
        os.makedirs(subset_dir)
    subset_dir_valid = os.path.join(subset_dir, "valid")
    subset_dir_train = os.path.join(subset_dir, "train")
    if not os.path.exists(subset_dir_valid):
        print("os.makedirs("+subset_dir_valid+")")
        os.makedirs(subset_dir_valid)
    if not os.path.exists(subset_dir_train):
        print("os.makedirs("+subset_dir_train+")")
        os.makedirs(subset_dir_train)

    for c in classes:
        valid_source_dir = os.path.join(root_dir, "valid", c)
        train_source_dir = os.path.join(root_dir, "train", c)
        valid_dest_dir = os.path.join(subset_dir_valid, c)
        train_dest_dir = os.path.join(subset_dir_train, c)

        print("shutil.copytree(" + valid_source_dir + "," + valid_dest_dir + ")")
        shutil.copytree(valid_source_dir, valid_dest_dir)
        print("shutil.copytree(" + train_source_dir + "," + train_dest_dir + ")")
        shutil.copytree(train_source_dir, train_dest_dir)




def preprocess_wave(wave):
    """ Preprocess a signal by computing the noise and signal mask of the
    signal, and extracting each part from the signal
    """
    Sxx = wave_to_amplitude_spectrogram(wave)

    n_mask = compute_noise_mask(Sxx)
    s_mask = compute_signal_mask(Sxx)

    n_mask_scaled = reshape_binary_mask(n_mask, wave.shape[0])
    s_mask_scaled = reshape_binary_mask(s_mask, wave.shape[0])

    signal_wave = extract_masked_part_from_wave(s_mask_scaled, wave)
    noise_wave = extract_masked_part_from_wave(n_mask_scaled, wave)

    return signal_wave, noise_wave


def extract_noise_part(spectrogram):
    """ Extract the noise part of a spectrogram
    """
    mask = compute_noise_mask(spectrogram)
    noise_part = extract_masked_part_from_spectrogram(mask, spectrogram)
    return noise_part


def extract_signal_part(spectrogram):
    """ Extract the signal part of a spectrogram
    """
    mask = compute_signal_mask(spectrogram)
    signal_part = extract_masked_part_from_spectrogram(mask, spectrogram)
    return signal_part


def extract_masked_part_from_spectrogram(mask, spectrogram):
    """ Extract the masked part of the spectrogram
    """
    return spectrogram[:, mask]


def extract_masked_part_from_wave(mask, wave):
    return wave[mask]


def compute_signal_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
    # Returns
        binary_mask : the binary signal mask
    """
    threshold = 3
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    return mask


def compute_noise_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
    # Returns
        binary_mask : the binary noise mask
    """
    threshold = 2.5
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    # invert mask
    return np.logical_not(mask)


def compute_binary_mask_sprengel(spectrogram, threshold):
    """ Computes a binary mask for the spectrogram
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
        threshold   : a threshold for times larger than the median
    # Returns
        binary_mask : the binary mask
    """
    # normalize to [0, 1)
    norm_spectrogram = normalize(spectrogram)

    # median clipping
    binary_image = median_clipping(norm_spectrogram, threshold)

    # erosion
    binary_image = morphology.binary_erosion(binary_image, footprint=np.ones((4, 4)))

    # dilation
    binary_image = morphology.binary_dilation(binary_image, footprint=np.ones((4, 4)))

    # extract mask
    mask = np.array([np.max(col) for col in binary_image.T])
    mask = smooth_mask(mask)

    return mask


def compute_binary_mask_lasseck(spectrogram, threshold):
    # normalize to [0, 1)
    norm_spectrogram = normalize(spectrogram)

    # median clipping
    binary_image = median_clipping(norm_spectrogram, threshold)

    # closing binary image (dilation followed by erosion)
    binary_image = morphology.binary_closing(binary_image, selem=np.ones((4, 4)))

    # dialate binary image
    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    # apply median filter
    binary_image = filters.median(binary_image, selem=np.ones((2, 2)))

    # remove small objects
    binary_image = morphology.remove_small_objects(binary_image, min_size=32, connectivity=1)

    mask = np.array([np.max(col) for col in binary_image.T])
    mask = smooth_mask(mask)

    return mask


# TODO: This method needs some real testing
def reshape_binary_mask(mask, size):
    """ Reshape a binary mask to a new larger size
    """
    reshaped_mask = np.zeros(size, dtype=bool)

    x_size_mask = mask.shape[0]
    scale_fact = int(np.floor(size / x_size_mask))
    rest_fact = float(size) / x_size_mask - scale_fact

    rest = rest_fact
    i_begin = 0
    i_end = int(scale_fact + np.floor(rest))
    for i in mask:
        reshaped_mask[i_begin:i_end] = i
        rest += rest_fact
        i_begin = i_end
        i_end = i_end + scale_fact + int(np.floor(rest))
        if rest >= 1:
            rest -= 1.

    if not (i_end - scale_fact) == size:
        raise ValueError("there seems to be a scaling error in reshape_binary_mask")

    return reshaped_mask


def smooth_mask(mask):
    """ Smooths a binary mask using 4x4 dilation
        # Arguments
            mask : the binary mask
        # Returns
            mask : a smoother binary mask
    """
    n_hood = np.ones(4)
    mask = morphology.binary_dilation(mask, n_hood)
    mask = morphology.binary_dilation(mask, n_hood)

    # type casting is a bitch
    return mask


def median_clipping(spectrogram, number_times_larger):
    """ Compute binary image from spectrogram where cells are marked as 1 if
    number_times_larger than the row AND column median, otherwise 0
    """
    row_medians = np.median(spectrogram, axis=1)
    col_medians = np.median(spectrogram, axis=0)

    # create 2-d array where each cell contains row median
    row_medians_cond = np.tile(row_medians, (spectrogram.shape[1], 1)).transpose()
    # create 2-d array where each cell contains column median
    col_medians_cond = np.tile(col_medians, (spectrogram.shape[0], 1))

    # find cells number_times_larger than row and column median
    larger_row_median = spectrogram >= row_medians_cond * number_times_larger
    larger_col_median = spectrogram >= col_medians_cond * number_times_larger

    # create binary image with cells number_times_larger row AND col median
    binary_image = np.logical_and(larger_row_median, larger_col_median)
    return binary_image


def normalize(X):
    """ Normalize numpy array to interval [0, 1]
    """
    mi = np.min(X)
    ma = np.max(X)

    X = (X - mi) / (ma - mi)
    return X

#################### 谱图处理 ####################

def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n, i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

def wave_to_complex_spectrogram(wave):
    return librosa.stft(wave, n_fft=512, hop_length=128, win_length=512)

def wave_to_amplitude_spectrogram(wave):
    X = wave_to_complex_spectrogram(wave)
    X = np.abs(X) ** 2
    return X[4:232]

def wave_to_log_amplitude_spectrogram(wave, fs):
    return np.log(wave_to_amplitude_spectrogram(wave, fs))

def wave_to_sample_spectrogram(wave, fs):
    # Han window of size 512, and hop size 128 (75% overlap)
    return wave_to_log_amplitude_spectrogram(wave, fs)

def wave_to_tempogram(wave, fs):
    tempogram = librosa.feature.tempogram(wave, fs)
    return tempogram


from scipy import ndimage
import numpy as np
import librosa
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

#### feature extract ####
def filter_isolated_cells(array, struct):
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array

def feat_norm(feat):
    feat = filter_isolated_cells(feat, struct=np.ones((3, 3)))
    return feat

def get_feature(wave_data, sr, frame_len=1024, n_fft=None, win_step=1/4, window="hamming", preemph=0.97, n_mels=256,
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
    fbank_delta = librosa.feature.delta(fbank, width=3)  # 一阶差分
    fbank_delta2 = librosa.feature.delta(fbank, width=3, order=2)  # 二阶差分
    #fbank = feat_norm(fbank)
    #fbank_delta = feat_norm(fbank_delta)
    #fbank_delta2 = feat_norm(fbank_delta2)
    feat = np.stack((fbank, fbank_delta, fbank_delta2))
    return feat
