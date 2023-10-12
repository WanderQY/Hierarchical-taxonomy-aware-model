from kaldiio import WriteHelper
from utils import *
import sys
sys.path.append('../../BirdCLEF2017/')

#################### signal to noise ####################
def preprocess_sound_file(filename, signal_dir, noise_dir):
    """ Preprocess sound file. Loads sound file from filename, downsampels,
    extracts signal/noise parts from sound file, splits the signal/noise parts
    into equally length segments of size segment size seconds.
    # Arguments
        filename : the sound file to preprocess
        class_dir : the directory to save the extracted signal in
        noise_dir : the directory to save the extracted noise in
    # Returns
        nothing, simply saves the preprocessed sound segments
    """
    samplerate, wave = read_wave_file(filename)
    print(wave)

    if len(wave) == 0:
        print("An empty sound file..")

    wave = wave.astype(float)

    signal_wave, noise_wave = preprocess_wave(wave)

    basename = get_basename_without_ext(filename)

    if signal_wave.shape[0] > 0:
        # signal_segments = split_into_segments(signal_wave, samplerate, segment_size_seconds)
        # save_segments_to_file(class_dir, signal_segments, basename, samplerate)
        filepath = os.path.join(signal_dir, basename + ".wav")
        write_wave_to_file(filepath, samplerate, signal_wave)

    if noise_wave.shape[0] > 0:
        # noise_segments = split_into_segments(noise_wave, samplerate, segment_size_seconds)
        # save_segments_to_file(noise_dir, noise_segments, basename, samplerate)
        filepath = os.path.join(noise_dir, basename + ".wav")
        write_wave_to_file(filepath, samplerate, noise_wave)




filedir = ['SplitDatas/train/src', 'SplitDatas/valid/src']
# {'id':{'path':,'label':}}
dataset = {}
birdsonly = {}
save_dir = sys.path[-1] + 'SortedData/Song_22050'
signal_dir = sys.path[-1] + 'SortedData/BirdsOnly'
noise_dir = sys.path[-1] + 'SortedData/NoiseOnly'
from pydub import AudioSegment
#filename = sys.path[-1] + 'SplitDatas/train/src/Aburria aburri efzmgo/0_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN27309.wav'



for d in filedir:
    key = d.split('/')[1]
    dataset[key] = {}
    birdsonly[key] = {}
    dir = sys.path[-1] + d
    class_list = os.listdir(dir)
    for c in class_list:
        class_dir = os.path.join(dir, c)
        wav_list = os.listdir(class_dir)
        for wav in wav_list:
            path = os.path.join(save_dir, wav)
            sound = AudioSegment.from_wav(path)
            sound = sound.set_channels(1).set_frame_rate(22050).set_sample_width(2)
            basename = get_basename_without_ext(path)
            save_path = os.path.join(save_dir, basename + ".wav")
            sound.export(save_path, format="wav")
            dataset[key][basename] = {}
            birdsonly[key][basename] = {}
            preprocess_sound_file(path, signal_dir, noise_dir)
            dataset[key][basename]['path'] = os.path.join(save_dir, basename + ".wav")
            dataset[key][basename]['label'] = c
            birdsonly[key][basename]['path'] = os.path.join(signal_dir, basename + ".wav")
            birdsonly[key][basename]['label'] = c




import json
import librosa
split_dataset = {}
split_dataset['origin']=dataset
split_dataset['birdsonly']=birdsonly
miss = []
for k in split_dataset:
    for d in split_dataset[k]:
        for file in split_dataset[k][d]:
            if not 'duration' in split_dataset[k][d][file].keys():
                path = split_dataset[k][d][file]['path']
                try:
                    duration = librosa.get_duration(filename=path)
                    split_dataset[k][d][file]['duration'] = duration
                except:
                    miss.append(path)

for d in split_dataset['birdsonly']:
    for file in split_dataset['birdsonly'][d]:
        if split_dataset['birdsonly'][d][file]['path'] in miss:
            del split_dataset['birdsonly'][d][file]

json_file = sys.path[-1] + 'SplitDatas/split_dataset1.json'
with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(split_dataset, fp)

### save noise datas ###
import json
noise_datas = {}
noise_file = os.listdir(noise_dir)
for file in noise_file:
    path = os.path.join(noise_dir, file)
    basename = get_basename_without_ext(path)
    noise_datas[basename] = {}
    noise_datas[basename]['path'] = path
    noise_datas[basename]['duration'] = librosa.get_duration(filename=path)

json_file = sys.path[-1] + 'SplitDatas/noiseonly.json'
with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(noise_datas, fp)


dataset = json.load(open(sys.path[-1] + 'SplitDatas/noiseonly.json'))
ark_path = sys.path[-1] + 'Feature/noise_datas.ark'
scp_path = sys.path[-1] + 'Feature/noise_datas.scp'
missed = []
with WriteHelper('ark,scp:{},{}'.format(ark_path, scp_path)) as writer:
    for file in dataset.keys():
        try:
            wave_data, _ = librosa.load(dataset[file]['path'], None)
            idx = dataset[file]['path']
            writer(idx, wave_data)
        except:
            missed.append(file)
print(missed)

lines = []
with open(scp_path, 'r') as T:
    lines += T.readlines()
print('data.scp done')
print(len(lines))

path_dict = {}
for line in lines:
    idx, path = line.split()
    path_dict[idx] = path
for file in dataset.keys():
    idx = dataset[file]['path']
    dataset[file]['ark_path'] = path_dict[idx]

json_file = sys.path[-1] + 'SplitDatas/noiseonly.json'
with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(dataset, fp)

### save wave datas ###
import json
dataset = json.load(open(sys.path[-1] + 'SplitDatas/all_dataset.json'))
ark_path = sys.path[-1] + 'Feature/wave_datas.ark'
scp_path = sys.path[-1] + 'Feature/wave_datas.scp'
missed = []
with WriteHelper('ark,scp:{},{}'.format(ark_path, scp_path)) as writer:
    for k in dataset.keys():
        for file in dataset[k].keys():
            try:
                feat, _ = librosa.load(dataset[k][file]['path'], None)
                idx = dataset[k][file]['path']
                writer(idx, feat)
            except:
                missed.append(file)
print(missed)

lines = []
with open(scp_path, 'r') as T:
    lines += T.readlines()
print('data.scp done')
print(len(lines))

path_dict = {}
for line in lines:
    idx, path = line.split()
    path_dict[idx] = path

for k in dataset.keys():
    for file in dataset[k].keys():
        idx = dataset[k][file]['path']
        dataset[k][file]['ark_path'] = path_dict[idx]
