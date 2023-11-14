# Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species: A genetics-informed approach
Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species

# Abstract
The study of bird populations is crucial for biodiversity research and conservation. Deep artificial neural networks have revolutionized bird acoustic recognition, but most methods overlook hierarchical relationships among bird populations, resulting in the loss of ecological information. To address this concern, we propose the concept of Genetics-informed Neural Networks (GINN), a novel approach that incorporates hierarchical multilevel labels for each bird. This approach uses a hierarchical semantic embedding framework to capture feature information at different levels. Attention mechanisms are employed to extract and select common and distinguishing features, thereby improving classification accuracy. We also propose a path correction strategy to rectify inconsistent predictions. Experimental results on bird acoustic datasets demonstrate that GINN outperforms current methods, achieving classification accuracies of 90.450\%, 91.883\%, and 89.950\% on the Lishui-Zhejiang birdsdata (100 species), BirdCLEF2018-Small (150 species), and BirdCLEF2018-Large (500 species) datasets respectively, with the lowest hierarchical distance of a mistake across all datasets. This approach is applicable to any bird acoustic dataset, and the method presents significant advantages as the number of categories increases.
![Local Image](images/abstract.png){:height="200px" width="200px"}

# Model architecture
![Local Image](images/preprocessing.png){:height="200px" width="200px"}
![Local Image](images/model architecture.PNG){:height="200px" width="200px"}

# Results
![Local Image](images/grad-cam.png){:height="200px" width="200px"}

# Get Started
1. Install Python 3.8, PyTorch 1.11.0.
2. Download data. You can obtain all the trained models and subset of acoustic recordings from Google Drive and can be accessed via https://drive.google.com/drive/folders/1rORMGIrZKOCLPsvSj0vsGs2Iu5A4QnJE?usp=sharing. All the datasets are well pre-processed and can be used easily.
3. Audio preprocessing. Use python audio_preprocessing.py in folder ./Codelist/. Generate processed audio files in three folders ./SortedData/Song_22050 ./SortedData/BirdsOnly ./SortedData/NoiseOnly.
4. Partition training set, validation set, testing set. Use python split_dataset.py in folder ./Codelist/. Obtain ./SplitDatas/split_dataset1_with_hier.json
5. Train the model. Use python train.py. 
6. Test the model. Use python evaluation.py.
   
# Contact
If you have any questions or want to use the code, please contact wangqingyu@mail.ustc.edu.cn.
