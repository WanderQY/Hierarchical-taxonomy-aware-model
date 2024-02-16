# Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species: A phylogenetic perspective
Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species

# Authors
Qingyu Wang [1] & Yanzhi Song [1] & Yeqian Du [1] & Zhouwang Yang [1] & Peng Cui [2] & Binnan Luo [3]

[1] University of Science and Technology of China, School of Data Science, China

[2] Nanjing Institute of Environmental Sciences, China

[3] Jiangsu Tianning Ecological Group Co., China

# Abstract
The study of bird populations is crucial for biodiversity research and conservation. Deep artificial neural networks have revolutionized bird acoustic recognition, but most methods overlook hierarchical relationships among bird populations, resulting in the loss of biological information. To address this concern, we propose the concept of Phylogenetic Perspective Neural Networks (PPNN), a novel approach that incorporates hierarchical multilevel labels for each bird. This approach uses a hierarchical semantic embedding framework to capture feature information at different levels. Attention mechanisms are employed to extract and select common and distinguishing features, thereby improving classification accuracy. We also propose a path correction strategy to rectify inconsistent predictions. Experimental results on bird acoustic datasets demonstrate that PPNN outperforms current methods, achieving classification accuracies of 90.450\%, 91.883\%, and 89.950\% on the Lishui-Zhejiang birdsdata (100 species), BirdCLEF2018-Small (150 species), and BirdCLEF2018-Large (500 species) datasets respectively, with the lowest hierarchical distance of a mistake across all datasets. This approach is applicable to any bird acoustic dataset, and the method presents significant advantages as the number of categories increases.

<div align=center>
   <img src="images/abstract.png" width="700px">
</div>
   
# Model architecture
## Audio preprocessing
<div align=center>
   <img src="images/preprocessing.png" width="700px">
</div>

Signal-to-noise separation + Spectrogram transformation + Data augmentation

## Phylogenetic Perspective Neural Network
<div align=center>
   <img src="images/model architecture.PNG" width="1000px">
</div>

# Results and findings
<div align=center>
   <img src="images/BC-S result.png" width="800px">
</div>

<div align=center>
   <img src="images/BC-L result.png" width="800px">
</div>

<div align=center>
   <img src="images/LS result.png" width="800px">
</div>

<div align=center>
   <img src="images/LS1 result.png" width="800px">
</div>

* The PRNN model consistently outperformed all comparison methods on the BC-S
and BC-L datasets for each class hierarchy.

* The PRNN model showed minimal parameter changes (+6.44M), highlighting its applicability.

* On the LS dataset, PRNN exhibits superior generalization performance as the training set size decreases.

* The PRNN model had the lowest HDM values on all datasets, implying that the application of hierarchical
  constraints can mitigate prediction errors, thereby enhancing the reliability of prediction.

## Grad-cam
Visualization of the activation achieved by four distinct network branches, each corresponding to a different hierarchy. 
<div align=center>
   <img src="images/grad-cam.png" width="700px">
</div>

# Get Started
## Open source data
[Our training data is open source and can be accessed here.](http://gofile.me/5Erwh/OlgtdIeul)
We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding metadata files (.xml).
You can download the zipped files or select specific portions of the data to create your own datasets.

### [BirdCLEF2018](https://www.imageclef.org/LifeCLEF2018)

The dataset is the official bird sound recognition competition dataset released by LifeCLEF for 2018. Sourced primarily from
the [Xeno-Canto Archive](xeno-canto.org), it contains songs of 1500 bird species from Central and South America, making it the
most comprehensive bird acoustics dataset in the literature. In total, the database containes 36,446 occurrences of bird songs recorded in files of various lengths.
<div align=center>
   <img src="images/stat of BC.png" width="800px">
</div>

### Lishui-Zhejiang Birdsdata

The dataset is a large collection of bird sounds gathered by the Lishui Ecological Environment Bureau from the natural environment of Lishui City, Zhejiang Province, China. 
It comprises live recordings of 597 distinct bird species spanning 20 orders and 68 families. In total, the database contains 123,109 occurrences of bird songs recorded in files of various lengths.
<div align=center>
   <img src="images/stat of LS.png" width="800px">
</div>


**You can find the species list and their information in the [./Info folder](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Info).**

We have separated three subsets (LS, BC-S, BC-L) with hierarchical labels for training and evaluating the model. The datasets were divided into three exclusive groups: 80% for training, 10% for validation, and 10% for testing to compare the experimental results. The prepared datasets were constructed to json files in [./SplitDatas/ folder](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/SplitDatas).

'origin': The infomation of raw audio file.

'birdsonly' and 'noiseonly': The two different parts of the raw audio after signal-to-noise seperation.

## Code description
### Libraries
python==3.8.8

torch==1.9.1+cu111

torchvision==0.10.1+cu111

librosa==0.8.1

scikit-learn==1.0

numpy==1.22.3

matplotlib==3.7.1

kaldiio==2.17.2

audiomentations==0.27.0

pandas==1.5.3

openpyxl==3.0.9

tqdm==4.62.3

scikit-skimage==0.19.3

### Begin to train
#### 1. Audio preprocessing.
Execute the Python script
```audio_preprocessing.py```
in the folder `./Codelist/`. This will generate processed audio files in three folders: `./SortedData/Song_22050`, `./SortedData/BirdsOnly`, and `./SortedData/NoiseOnly`.

#### 2. Partitioning the dataset.
Utilize the Python script 
```split_dataset.py```
in the folder `./Codelist/` to partition the dataset into training, validation, and testing sets. The result will be saved as `./SplitDatas/split_dataset1_with_hier.json`.

#### 3. Model training.
Run the Python script ```train.py```.

#### 4. Model testing.
Execute the Python script ```evaluation.py```.

**For more details, refer to the [./Codelist/ folder](https://github.com/WanderQY/Hierarchical-taxonomy-aware-model/tree/main/Codelist).**

# Further discussion
It should be noted that on-site sound analysis entails a more intricate environmental noise assessment, as a variety of factors such as different birds, insects, mammals, artificial sounds, wind, rain, and thunder can be associated with the target species in the collected recordings. Actually, the number of bird sounds collected is not very large, with non-bird sounds accounting for more than 70% of the total recordings. Compared to the two-stage model constructed using pre-determined bird / noise discrimination (a), we used an improved hierarchical relationship tree (b) (c), where an additional “s2n (signal-to-noise)” level was added before the first level to facilitate the training of a more comprehensive classification model. 
<div align=center>
   <img src="images/hier with noise.png" width="800px">
</div>
In this manner, non avian sounds are less likely to be recognized as flight calls, and vice versa. Due to the split of the training and testing sets at a ratio of 8: 2, we employed the extracted target bird chirps from the training set to fine-tune the trained GINN model. Evaluation results on test sets are shown in Table 9. Evaluate the effectiveness of the model from three different perspectives: recall, precision, and accuracy. Furthermore, it is feasible to enlarge the scope of the unknown species Table 9 by incorporating a “other” sibling node to each layer. This scenario will not be explored further in this article, but it demonstrates the expandability of hierarchical relationships.
<div align=center>
   <img src="images/soundscape result.png" width="800px">
</div>


# Project
Our Biodiversity Intelligent Identification System is about to be launched.

http://180.101.130.43:10081/

# Contact
If you have any questions or want to use the code, please contact wangqingyu@mail.ustc.edu.cn.
