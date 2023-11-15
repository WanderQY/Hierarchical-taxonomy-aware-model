# Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species: A genetics-informed approach
Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species

# Authors
Qingyu Wang [1] & Yanzhi Song [1] & Yeqian Du [1] & Zhouwang Yang [1] & Peng Cui [2] & Binnan Luo [3]

[1] University of Science and Technology of China, School of Data Science, China

[2] Nanjing Institute of Environmental Sciences, China

[3] Jiangsu Tianning Ecological Group Co., China

# Abstract
The study of bird populations is crucial for biodiversity research and conservation. Deep artificial neural networks have revolutionized bird acoustic recognition, but most methods overlook hierarchical relationships among bird populations, resulting in the loss of ecological information. To address this concern, we propose the concept of Genetics-informed Neural Networks (GINN), a novel approach that incorporates hierarchical multilevel labels for each bird. This approach uses a hierarchical semantic embedding framework to capture feature information at different levels. Attention mechanisms are employed to extract and select common and distinguishing features, thereby improving classification accuracy. We also propose a path correction strategy to rectify inconsistent predictions. Experimental results on bird acoustic datasets demonstrate that GINN outperforms current methods, achieving classification accuracies of 90.450\%, 91.883\%, and 89.950\% on the Lishui-Zhejiang birdsdata (100 species), BirdCLEF2018-Small (150 species), and BirdCLEF2018-Large (500 species) datasets respectively, with the lowest hierarchical distance of a mistake across all datasets. This approach is applicable to any bird acoustic dataset, and the method presents significant advantages as the number of categories increases.

<div align=center>
   <img src="images/abstract.png" width="700px">
</div>
   
# Model architecture
## Audio preprocessing
<div align=center>
   <img src="images/preprocessing.png" width="700px">
</div>

'.' signal-noise separation

'.' spectrogram transformation

'.' data augmentation

## Genetics-informed Neural Network
<div align=center>
   <img src="images/model architecture.PNG" width="700px">
</div>

# Results
<div align=center>
   <img src="images/grad-cam.png" width="700px">
</div>

# Get Started
## Open source data
[Our training data is open source and can be accessed here.]([https://www.openai.com](http://gofile.me/5Erwh/OlgtdIeul)) 
We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding matadata files (.xml).
1. BirdCLEF
The dataset contains songs of 22 bird species from 5 families and genera differents. The recordings were downloaded from the Xeno-canto database in .wav format and each recording was manually annotated by labelling the start and stop time for every vocalisation occurrence using Sonic Visualiser. In total, database contained 6537 occurrences of bird songs of various length from 967 file recordings. A precise description of the distribution by species and country can be found in the associated article. See below an example of an annotated file.
3. Zhejiang-lishui birdsdata

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

1. Audio preprocessing. Use python audio_preprocessing.py in folder ./Codelist/. Generate processed audio files in three folders ./SortedData/Song_22050 ./SortedData/BirdsOnly ./SortedData/NoiseOnly.

2. Partition training set, validation set, testing set. Use python split_dataset.py in folder ./Codelist/. Obtain ./SplitDatas/split_dataset1_with_hier.json

3. Train the model. Use python train.py. 

4. Test the model. Use python evaluation.py.

# Project
Our Biodiversity Intelligent Identification System is about to be launched.

http://180.101.130.43:10081/

# Contact
If you have any questions or want to use the code, please contact wangqingyu@mail.ustc.edu.cn.
