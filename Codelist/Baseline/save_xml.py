import os
import time
from openpyxl import Workbook
import traceback
import xmltodict as x2d
import sys
sys.path.append('E:/Work/BirdCLEF2017/')

# Specify all folders containing wav or xml files
data_dirs = [sys.path[-1] + 'BirdCLEF2017TrainingSetPart1/TrainingSet/xml/',
             sys.path[-1] + 'BirdCLEF2017TrainingSetPart1/TrainingSet/wav/',
             sys.path[-1] + 'BirdCLEF2017TrainingSetPart2/data/']
class_dir = sys.path[-1]+'BirdCLEF2017train/'


##############################################

metadata = []
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
        sub_species, Order, BackgroundSpecies, Time, Elevation, Content, Comments = "", "", "", "", "", "", ""
        try:
            if xmldata['Audio']['Sub-species']:
                sub_species = xmldata['Audio']['Sub-species'] + ' '
            if xmldata['Audio']['Order']:
                Order = xmldata['Audio']['Order'] + ' '
            if xmldata['Audio']['BackgroundSpecies']:
                BackgroundSpecies = xmldata['Audio']['BackgroundSpecies'] + ' '
            if xmldata['Audio']['Time']:
                Time = xmldata['Audio']['Time'] + ' '
            if xmldata['Audio']['Elevation']:
                Elevation = xmldata['Audio']['Elevation'] + ' '
            if xmldata['Audio']['Content']:
                Content = xmldata['Audio']['Content'] + ' '
            if xmldata['Audio']['Comments']:
                Comments = xmldata['Audio']['Comments'] + ' '
        except:
            traceback.print_exc()


        # new path name
        dst_path = class_dir + xmldata['Audio']['Genus'] + '-' + xmldata['Audio']['Species'] + '-' + \
                    xmldata['Audio']['ClassId'] + '-' + src_path.split('/')[-1].split('_')[-1]

        # add to class ids
        cid = xmldata['Audio']['ClassId']
        c = xmldata['Audio']['Genus'] + ' ' + xmldata['Audio']['Species'] + ' ' + sub_species
        # add to metadata
        metadata.append([xmldata['Audio']['ClassId'],
                         xmldata['Audio']['FileName'],
                         Order,
                         xmldata['Audio']['Family'],
                         xmldata['Audio']['Genus'],
                         xmldata['Audio']['Species'],
                         sub_species,
                         BackgroundSpecies,
                         xmldata['Audio']['VernacularNames'],
                         xmldata['Audio']['Year'],
                         xmldata['Audio']['Date'],
                         Time,
                         xmldata['Audio']['Locality'],
                         xmldata['Audio']['Latitude'],
                         xmldata['Audio']['Longitude'],
                         Elevation,
                         xmldata['Audio']['Quality'],
                         xmldata['Audio']['MediaId'],
                         Content,
                         Comments,
                         xmldata['Audio']['Author'],
                         xmldata['Audio']['AuthorID'],
                         src_path, dst_path])
        wb = Workbook()
        ws = wb.active
        ws.title = 'metadata'
        ws.append(['ClassId', 'FileName', 'Order', 'Family', 'Genus', 'Species', 'Sub-species', 'BackgroundSpecies',
                   'VernacularNames', 'Year', 'Date', 'Time', 'Locality', 'Latitude', 'Longitude', 'Elevation',
                   'Quality', 'MediaId', 'Content', 'Comments', 'Author', 'AuthorID'])
        for item in metadata:
            ws.append(item)

    except KeyboardInterrupt:
        break

    except:
        traceback.print_exc()
        miss.append(xmlfiles[i])
        continue

    print("DONE!")

wb.save(sys.path[-1] + 'metadata1.xlsx')



