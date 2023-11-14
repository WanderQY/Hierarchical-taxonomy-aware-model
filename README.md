# Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species: A genetics-informed approach
Hierarchical-taxonomy-aware and attentional convolutional neural networks for acoustic identification of bird species

# Get Started
1. Install Python 3.8, PyTorch 1.11.0.
2. Download data. You can obtain all the trained models and subset of acoustic recordings from Google Drive and can be accessed via https://drive.google.com/drive/folders/1rORMGIrZKOCLPsvSj0vsGs2Iu5A4QnJE?usp=sharing. All the datasets are well pre-processed and can be used easily.
3. Audio preprocessing. Use python audio_preprocessing.py in folder ./Codelist/. Generate processed audio files in three folders ./SortedData/Song_22050 ./SortedData/BirdsOnly ./SortedData/NoiseOnly.
4. Partition training set, validation set, testing set. Use python split_dataset.py in folder ./Codelist/. Obtain ./SplitDatas/split_dataset1_with_hier.json
5. Train the model. Use python train.py. 
6. Test the model. Use python evaluation.py.
   
# Contact
If you have any questions or want to use the code, please contact wangqingyu@mail.ustc.edu.cn.
