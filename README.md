# GitSense
Artifact of a paper “Commit Artifact Preserving Build Prediction” (ISSTA'24)


## Dataset
You can obtain our training and testing data through the following link. 
The textual features and statistical features are stored separately (.json and .csv). We also provide the Github commit extraction tool in the /extract_data directory. 
If you want to extract data by yourself, you can use it following the ReadMe in the /extract_data directory.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12820030.svg)](https://zenodo.org/records/12820030)

The original data and code to extract important features from the original features are in the /im_features directory. 
The extracted important features and their descriptions are listed in the feature.csv file.

## Code
The ChatGPT prompt and code are in the /gpt_prompt directory. 
If you want to run it in our scenario, you can simply replace the API Key with your API Key.

The results and trained models are in the /result and /model, respectively.

If you want to reproduce our results, you can directly run "python Main.py" with our data. 

！！！Remember to change the directory in your environment. 

All the training configs are listed in the trainer.py. You can change the parameters according to your device.

## Use the model on your dataset
Additionally, if you want to use GitSense in your scenario, you can change the code in the data.py to replace the dataset or the processing method of the dataset.

The GPAResNet.py is the model part of GitSense. If you want to use it in your task, you may change the model code to fit your task.

## Others
Transformer_origin.py is the model of ablation evaluation of GitSense without CNN and sliding windows. It is the original Transformer.

The predict.py in the /TF directory is the reimplementation of the TF approach.

