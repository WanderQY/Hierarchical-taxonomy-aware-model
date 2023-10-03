from sklearn.utils import shuffle
import sys
sys.path.append('E:/Work/BirdCLEF2017/')


#### split dataset ####
import json
dataset = json.load(open(sys.path[-1] + 'SplitDatas/all_dataset.json'))
split_dataset = {}

spe_list = {}
for d in dataset.keys():
    split_dataset[d] = {}
    spe_list[d] = {}
    split_dataset[d]['train'] = {}
    split_dataset[d]['valid'] = {}
    split_dataset[d]['test'] = {}
    for file in dataset[d].keys():
        if dataset[d][file]['label'] not in spe_list[d].keys():
            spe_list[d][dataset[d][file]['label']] = []
        spe_list[d][dataset[d][file]['label']].append([file, dataset[d][file]['duration'], dataset[d][file]['path'], dataset[d][file]['ark_path']])
for d in dataset.keys():
    for species in spe_list[d].keys():
        train, valid, test = [], [], []
        species_list = spe_list[d][species]
        species_list_sorted = sorted(species_list, key=lambda x: int(x[1]), reverse=True)
        num = int(len(species_list_sorted) // 10)
        for n in range(num):
            samples = species_list_sorted[n * 10:(n + 1) * 10]
            samples = shuffle(samples, random_state=1337)
            train += samples[:8]
            valid.append(samples[8])
            test.append(samples[9])
        res = species_list_sorted[num * 10:]
        train += res
        for t in train:
            split_dataset[d]['train'][t[0]] = {}
            split_dataset[d]['train'][t[0]]['label'] = species
            split_dataset[d]['train'][t[0]]['duration'] = t[1]
            split_dataset[d]['train'][t[0]]['path'] = t[2]
            split_dataset[d]['train'][t[0]]['ark_path'] = t[3]
        for t in valid:
            split_dataset[d]['valid'][t[0]] = {}
            split_dataset[d]['valid'][t[0]]['label'] = species
            split_dataset[d]['valid'][t[0]]['duration'] = t[1]
            split_dataset[d]['valid'][t[0]]['path'] = t[2]
            split_dataset[d]['valid'][t[0]]['ark_path'] = t[3]
        for t in test:
            split_dataset[d]['test'][t[0]] = {}
            split_dataset[d]['test'][t[0]]['label'] = species
            split_dataset[d]['test'][t[0]]['duration'] = t[1]
            split_dataset[d]['test'][t[0]]['path'] = t[2]
            split_dataset[d]['test'][t[0]]['ark_path'] = t[3]

json_file = sys.path[-1] + 'SplitDatas/split_dataset1.json'
with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(split_dataset, fp)

### small dataset ###

import json
dataset = json.load(open(sys.path[-1] + 'SplitDatas/all_dataset.json'))
split_dataset = {}

spe_list = {}
for d in dataset.keys():
    split_dataset[d] = {}
    split_dataset[d]['train'] = {}
    split_dataset[d]['valid'] = {}
    split_dataset[d]['test'] = {}
    for file in dataset[d].keys():
        if dataset[d][file]['label'] not in spe_list.keys():
            spe_list[dataset[d][file]['label']] = 0
        spe_list[dataset[d][file]['label']] += 1

list = []
for k, v in spe_list.items():
    list.append([k,v])
list= sorted(list, key=lambda x: int(x[1]), reverse=True)
select_list = []
for i, v in enumerate(list):
    if i % 3 == 0:
        select_list.append(v)

SELECT_CLASS = []
for i in select_list:
    SELECT_CLASS.append(i[0])

dataset = json.load(open(sys.path[-1] + 'SplitDatas/split_dataset1.json'))
for d in dataset.keys():
    for k in dataset[d].keys():
        for file in dataset[d][k].keys():
            if dataset[d][k][file]['label'] in SELECT_CLASS:
                split_dataset[d][k][file] = dataset[d][k][file]
json_file = sys.path[-1] + 'SplitDatas/small_dataset2.json'
with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(split_dataset, fp)