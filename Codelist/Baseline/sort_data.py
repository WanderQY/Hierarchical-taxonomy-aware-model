import os
import time
from shutil import copyfile
import traceback
import xmltodict as x2d
import sys
sys.path.append('E:/Work/BirdCLEF2017/')

# Specify all folders containing wav or xml files
data_dirs = [sys.path[-1] + 'RawData/BirdCLEF2017TrainingSetPart1/TrainingSet/xml/',
             sys.path[-1] + 'RawData/BirdCLEF2017TrainingSetPart1/TrainingSet/wav/',
             sys.path[-1] + 'RawData/BirdCLEF2017TrainingSetPart2/data/']

# Specify target folder
# Files will be sorted into folders according to genus, species and class id
class_dir = sys.path[-1] + 'SplitDatas/train/src/'


##############################################
def getMetaData(mfile, copy=False):
    metadata = {}
    classids = {}

    # collect all xml-file paths
    xmlfiles = []
    for d in data_dirs:
        xmlfiles += [d + xmlpath for xmlpath in sorted(os.listdir(d)) if xmlpath.split('.')[-1].lower() in ['xml']]

    print("XML-FILES:", len(xmlfiles))

    # collect all wav-file paths
    wavfiles = {}
    for d in data_dirs:
        for w in [d + wavpath for wavpath in sorted(os.listdir(d)) if wavpath.split('.')[-1].lower() in ['wav']]:
            wavfiles[w.split('/')[-1]] = w

    print("WAV-FILES:", len(wavfiles))

    # open files and extract metadata
    print("EXTRACTING METADATA...")
    start = time.time()
    miss = []
    xml_datas = {}  # save xml file datas
    for i in range(len(xmlfiles)):
        try:
            if i % 100 == 0 and i > 1:
                end = time.time()
                time_left = (((end - start) / 100) * (len(xmlfiles) - i)) // 60
                print("...", i, "TIME LEFT:", time_left, "min ...")
                start = time.time()

            xml = open(xmlfiles[i], 'r', encoding='UTF-8').read()
            xmldata = x2d.parse(xml)  # 将xml数据转为python中的dict字典数据

            # reference src file path
            src_path = wavfiles[xmldata['Audio']['FileName']]

            # compose new file path of class id, quality and species name
            # new path name
            dst_path = class_dir + xmldata['Audio']['Genus'] + ' ' + xmldata['Audio']['Species'] + ' ' + \
                       xmldata['Audio']['ClassId'] + '/' + xmldata['Audio']['Quality'] + '_' + src_path.split('/')[-1]

            # add to metadata
            metadata[src_path] = dst_path

            # add to class ids
            cid = xmldata['Audio']['ClassId']
            c = xmldata['Audio']['Genus'] + ' ' + xmldata['Audio']['Species']

            if not cid in classids:
                classids[cid] = {c: 1}
            else:
                if not c in classids[cid]:
                    classids[cid][c] = 1
                else:
                    classids[cid][c] += 1

            # write to file (raw file path; copy file path)
            mfile.write(src_path + ";" + dst_path + "\n")
            mfile.flush()

            # copy files?
            if copy:
                if not os.path.exists(dst_path.rsplit('/', 1)[0]):
                    os.makedirs(dst_path.rsplit('/', 1)[0])
                try:
                    copyfile(src_path, dst_path)
                except:
                    miss.append(xmlfiles[i])

        except KeyboardInterrupt:
            break

        except:
            traceback.print_exc()
            miss.append(xmlfiles[i])
            continue

    print("DONE!")

    return wavfiles, metadata, classids, miss

##############################################

# Results will be saved in a txt file
# You can copy all wav files to their specific folder by setting copy=True
mfile = open(sys.path[-1]+'Info/filepaths.txt', 'w')
wavfiles, mdata, cids, miss = getMetaData(mfile, copy=True)
mfile.close()
missfile = open(sys.path[-1]+'Info/missed.txt', 'w')
for i in miss:
    missfile.write(i + "\n")
missfile.close()
# Stats
print("CLASS IDS:", cids)