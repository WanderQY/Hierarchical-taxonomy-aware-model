## Folder Organization
The main folder contains scripts essential for training both the Baseline and GINN models.

## Data Pre-processing
1. **sort_data.py:** Sorts files (wav or xml) into folders based on attributes such as "genus," "species," and "class id." Results are saved in a text file.
2. **read_metadata.py:** Reads metadata from an Excel file to obtain detailed bird species information.
3. **audio_processing.py:** Preprocesses and organizes audio data. The `preprocess_sound_file()` function separates raw audio into signal and noise parts, saving them into separate directories. Processed data is serialized into a JSON file (split_dataset.json). Features (e.g., wave data) from the noise-only and bird-only datasets are extracted and saved into Kaldi-compatible ark files. Related information about noise-only and bird-only datasets is saved in additional JSON files (noiseonly.json, birdsonly.json).
4. **utils.py:** Contains key audio processing functions used in other scripts.
5. **class_label.py:** Saves default species used in experiments.
6. **split_dataset.py:** Splits datasets into training, validation, and test sets with an 8:1:1 ratio.

## Training of the Baseline Model
Related codes are saved in the "Baseline" directory.
1. **model.py:** Defines the architecture of the baseline model.
2. **dataloader.py:** Creates datasets for each epoch.
3. **mixup.py:** Implements data augmentation through Mixup.
4. **train.py:** Trains the baseline model.
5. **train_aug.py:** Trains the baseline model using data augmentations.
6. **evaluation.py:** Evaluates model performance by calculating the confusion matrix, plotting it, and computing accuracy, recall, precision, and specificity metrics.

## Training of the GINN Model
Related codes are saved in the "Hierarchy with DataAug" directory.
1. **model.py:** Defines the architecture of the GINN model.
2. **attention_modules.py:** Contains details of position attention.
3. **dataloader.py:** Creates datasets for each epoch.
4. **mixup.py:** Implements data augmentation through Mixup.
5. **loss.py:** Defines the hierarchical loss used in experiments.
6. **train.py:** Trains the GINN model.
7. **train_aug.py:** Trains the GINN model using data augmentations.
8. **path_corr.py:** Implements path correction strategy.
9. **evaluation.py:** Evaluates model performance by calculating the confusion matrix, plotting it, and computing accuracy, recall, precision, and specificity metrics.
10. **grad_cam.py:** Visualizes the activation map of the model.
