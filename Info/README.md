# Species information

It includes species lists of the datasets and their subsets used in the experiments.

<div align=center>
   <img src="https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/blob/main/images/dataset_table.png" width="700px">
</div>

Hierarchical taxonomy in the [IOC WORLD BIRD LIST (v13.1)](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/blob/main/Info/master_ioc_list_v13.1.xlsx) can be downloaded.

## Metadata

### Lishui-Zhejiang Birdsdata
This dataset is calculated by the orthoritists
The list comprises 597 species along with their scientific names, English names, and taxonomy, which can be found in the Excel file [./Info/Lishui-Zhejiang Birdsdata/Lishui-Zhejiang Birdsdata.xlsx](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info/Lishui-Zhejiang%20Birdsdata/Lishui-Zhejiang%20Birdsdata.xlsx). Additionally, the total number of files and recordings has been calculated.
<div align=center>
   <img src="images/LS metadata.png" width="800px">
</div>

We utilized "stratified sampling" to randomly select 100 bird species, denoted as "LS", for expedited model training. The species list for LS is saved in the TXT file [./Info/Lishui-Zhejiang Birdsdata/LS species list.txt](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info/Lishui-Zhejiang%20Birdsdata/LS%20species%20list.txt). 
<div align=center>
   <img src="images/stat of LS1.png" width="800px">
</div>

### [BirdCLEF2018]
The datails of the metadata can be found in the Excel file [./Info/BirdCLEF2018/metadata.xlsx](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info/BirdCLEF2018/metadata.xlsx). This information includes class IDs, file names, scientific names, taxonomy of species, and other details about the recordings.
<div align=center>
   <img src="images/BC metadata.png" width="800px">
</div>

We selected 150 and 500 bird species respectively to construct a small (BC-S) dataset and a large (BC-L) dataset to evaluate our model. The selected species list are saved in [./Info/BirdCLEF2018/BC-S species list.txt](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info/BirdCLEF2018/BC-S%20species%20list.txt) and [./Info/BirdCLEF2018/BC-L species list.txt](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info/BirdCLEF2018/BC-L%20species%20list.txt). 
<div align=center>
   <img src="images/stat of BC-S.png" width="800px">
</div>
<div align=center>
   <img src="images/stat of BC-L.png" width="800px">
</div>


## Open source data
[Our training data is open source and can be accessed here.](http://gofile.me/5Erwh/OlgtdIeul)
We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding metadata files (.xml).
You can download the zipped files or select specific portions of the data to create your own datasets.

An example of (.xml) format matedata.
```
<?xml version="1.0" encoding="UTF-8"?>
<Audio>
	<MediaId>49844</MediaId>
	<FileName>LIFECLEF2017_BIRD_XC_WAV_RN49844.wav</FileName>
	<ClassId>rfstup</ClassId>
	<Date>2010-01-23</Date>
	<Time>13:00</Time>
	<Country>Peru</Country>
	<Locality>Abra Portuchuelo, Ancash</Locality>
	<Latitude>-8.977</Latitude>
	<Longitude>-77.5998</Longitude>
	<Elevation>4200</Elevation>
	<Author>Andrew Spencer</Author>
	<AuthorID>CDTGHVBGZP</AuthorID>
	<Content>call, song</Content>
	<Quality>1</Quality>
	<Year>BirdCLEF2017</Year>
	<Family>Furnariidae</Family>
	<Genus>Geocerthia</Genus>
	<Species>serrana</Species>
	<VernacularNames>Striated Earthcreeper</VernacularNames>
</Audio>
```

