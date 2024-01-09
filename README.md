# CRISPR-M
CRISPR-M is a novel multi-view deep learning model with a new feature encoding scheme, regarding sgRNA off-target effect prediction for target sites containing indels and mismatches. CRISPR-M takes advantage of convolutional neural networks and bidirectional long short-term memory recurrent neural networks to construct a three-branch network towards multi-views. Compared to existing methods, CRISPR-M demonstrates significant performance advantages running on real-world datasets. Furthermore, experimental analysis of CRISPR-M under multiple metrics reveals its capability to extract features and validates its superiority on sgRNA off-target effect predictions (in /test).

## /codes
Here is the introduction of files in ```/codes```. The ```/codes``` directory encompasses essential components for data processing, encoding, and modeling. Functions for encoding data, computing metrics, and preprocessing raw data are provided. Additionally, there are backup files for code preservation, though they are not actively used in the current project. Notably, the main program for this project is located in the test directory.

|file|content|
|----|----|
|```encoding.py```|* Content: Contains data encoding functions. <br/> * Purpose: Likely involved in encoding or converting data for further processing in the project.|
|```metrics_utils.py```|* Content: Contains functions for computing metrics. <br/> * Purpose: Used for evaluating and measuring the performance of the implemented models or algorithms.|
|```data_preprocessing_utils.py```|* Content: Contains data preprocessing functions. <br/> * Purpose: Involved in preparing and cleaning the raw data for use in the project.|
|```positional_encoding.py```|* Content: Contains the PositionalEncoding class. <br/> * Purpose: Likely related to adding positional information to the data, which is crucial in sequence-based tasks like natural language processing.|
|```transformer_utils.py```|* Content: Contains Transformer classes. <br/> * Purpose: Central to the implementation of Transformer-based models, indicating that the project might involve sequence-to-sequence tasks or attention mechanisms.|
|other files|* Content: Used for backing up code but not actually used. <br/> * Purpose: These files seem to be reserved for backup purposes and are not actively utilized in the current project.|

## /datasets
The associations between dataset names in github and dataset names in paper. We collect two categories of datasets for model learning and validation. One category contains mismatches and indels, i.e., datasets CIRCLE and GUIDE_I in Table 1, and the other category contains mismatches only, i.e., other datasets. The CIRCLE dataset identifies 340 active off-target loci samples containing indels and 7031 active off-target loci samples containing mismatch only using the CIRCLE-seq technique. Note that the CIRCLE dataset is derived from the experimental data of 10 gRNAs and contains sufficient off-target samples for each gRNA, which is suitable for ten-fold cross validation. The GUIDE_I dataset also contains indel samples, but contains only 60 active off-target loci samples. For the rest datasets, we use PKD, SITE, GUIDE_II, and GUIDE_III for the mismatch-only experiments, and HEK293T and K562 for the experiments regarding epigenetic features. PKD has sufficient data for active off-target sites, but insufficient data for inactivated off-target sites. SITE has sufficient active off-target sites and inactivated off-target sites. GUIDE_II and GUIDE_III have sufficient data for inactive off-target loci samples, but only a small number of active off-target loci samples.

|dataset name in github ```/datasets```|dataset name in paper|
| ---- | ---- |
| CIRCLE(mismatch&insertion&deletion) | CIRCLE |
| dataset_I-2 | GUIDE_I |
| PKD | Protein knockout detection (PKD) |
| SITE | SITE |
| GUIDE_II | GUIDE_II |
| GUIDE_III | GUIDE_III |
| HEK293T in "epigenetic_data" | HEK293T |
| K562 in "epigenetic_data" | K562 |

## /test
Here are the experiments corresponding to each folder in ```/test```. The ```/test``` directory comprises experiments designed for specific purposes, each housed in a dedicated folder. Here's an overview of the experiments:

|folder|usage|main program of CRISPR-M|
|----|----|----|
| 1indel | Comparisons on Target Sites Containing Both Mismatches and Indels | \1indel\CRISPR-M\encoding_test.py |
| 2encoding_test | Comparisons of Encoding Schemes | \2encoding_test\mine\encoding_test.py |
| 3mismatch | Comparisons on Mismatches-only sgRNA-Target Prediction | \3mismatch\mine\encoding_test.py |
| 4multidataset | Comparisons with Complex Off-Target Site Datasets | \4multidataset\CRISPR-M\encoding_test.py |
| 6epigenetic | Comparisons with Epigenetic Features | \6epigenetic\CRISPR-M\encoding_test.py |
| 7visualization | Visual Analysis of CRISPR-M on the Off-Target Effect Prediction | \7visualization\encoding_test.py |
| other folders | discard |

Take folder-2encoding_test as an example, ```encoding_test.py``` in the folder-mine is the main program of the test, one could run ```python encoding_test.py``` for run it. The ```test_model.py``` contains the model architecture used for the test. The model in function 'm81212_n13' of ```test_model.py``` is final model of CRISPR-M. ```fig2.py``` in folder-fig2 is the visualization program that visualizes the results of several experiments.


Here is an output example of running the main program. The program will print the training process and the evaluation results of the model. 

> [INFO] ===== Start Loading dataset CIRCLE =====
[INFO] use 0-th-grna-fold grna  (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 1-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 2-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 3-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 4-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 5-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for validation  
[INFO] use 6-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 7-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 8-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] use 9-th-grna-fold grna (GTTGCCCCACAGGGCAGTAANGG) for train  
[INFO] train_features.shape = (560515, 24)  
[INFO] train_feature_ont.shape = (560515, 24)  
[INFO] train_feature_offt.shape = (560515, 24)  
[INFO] train_labels.shape = (560515,), and positive samples number = 7185  
[INFO] validation_features.shape = (24434, 24)  
[INFO] validation_feature_ont.shape = (24434, 24)  
[INFO] validation_feature_offt.shape = (24434, 24)  
[INFO] validation_labels.shape = (24434,), and positive samples number = 186  
[INFO] ===== End Loading dataset CIRCLE =====  
[INFO] ===== Start train =====  
Model: "model_n"  
Total params: 1,706,040  
Trainable params: 1,704,824  
Non-trainable params: 1,216  
Epoch 1/500  
548/548 [==============================] - ETA: 0s - loss: 0.3811 - acc: 0.8585 - auroc: 0.5045 - auprc: 0.0130       
Epoch 1: val_auprc improved from -inf to 0.00561, saving model to tcrispr_model.h5  
548/548 [==============================] - 56s 77ms/step - loss: 0.3811 - acc: 0.8585 - auroc: 0.5045 - auprc: 0.0130 - val_loss: 0.0759 - val_acc: 0.9924 - val_auroc: 0.3886 - val_auprc: 0.0056 - lr: 0.0010  
Epoch 2/500  
547/548 [============================>.] - ETA: 0s - loss: 0.0846 - acc: 0.9866 - auroc: 0.5736 - auprc: 0.0191  
Epoch 2: val_auprc improved from 0.00561 to 0.00761, saving model to tcrispr_model.h5  
548/548 [==============================] - 39s 70ms/step - loss: 0.0846 - acc: 0.9866 - auroc: 0.5741 - auprc: 0.0192 - val_loss: 0.1405 - val_acc: 0.9924 - val_auroc: 0.5000 - val_auprc: 0.0076 - lr: 0.0010  
...  
Epoch 96/500  
547/548 [============================>.] - ETA: 0s - loss: 0.0193 - acc: 0.9933 - auroc: 0.9905 - auprc: 0.7845  
Epoch 96: val_auprc did not improve from 0.28008  
548/548 [==============================] - 39s 71ms/step - loss: 0.0193 - acc: 0.9933 - auroc: 0.9905 - auprc: 0.7843 - val_loss: 0.0448 - val_acc: 0.9928 - val_auroc: 0.8442 - val_auprc: 0.2332 - lr: 7.3787e-06  
Epoch 96: early stopping  
[INFO] ===== End train =====  
764/764 [==============================] - 18s 19ms/step - loss: 0.0419 - acc: 0.9925 - auroc: 0.9369 - auprc: 0.2801         
764/764 [==============================] - 16s 16ms/step  
accuracy=0.9925104362773185, precision=1.0, recall=0.016129032258064516, f1=0.031746031746031744, fbeta=0.02008032128514056 auroc=0.9468733481621808, auprc=0.28261133474683825, auroc_by_auc=0.9468733481621808, auprc_by_auc=0.2809769808292004, spearman_corr_by_pred_score=0.13454728586392598, spearman_corr_by_pred_labels=0.12652358677195874  

## experiment environment
### run model dependencies
- Python 3.8
- tensorflow 2.9
- keras 2.9
- pandas 1.4
- numpy 1.22
- scikit-learn 1.1
#### visualization dependencies
- matploblib 3.5
- seaborn 0.11

## visualization
Here are the relations between visualization programs and pictures in experiments.

| visualization program | experiment name | figure name |
|----|----|----|
| ```/test/1indel/mean_roc_prc.py``` | Comparisons on Target Sites Containing Both Mismatches and Indels | Fig. 1 |
| ```/test/2encoding_test/fig2/fig2.py``` | Comparisons on Mismatches-only sgRNA-Target Prediction | Fig. 2 |
| ```/test/2encoding_test/fig2/fig2.py``` | Comparisons with Complex Off-Target Site Datasets | Fig. 2 |
| ```/test/2encoding_test/fig2/fig2.py``` | Comparisons of Encoding Schemes | Fig. 3 |
| ```/test/2encoding_test/fig2/fig2.py``` | Comparisons with Epigenetic Features | Fig. 3 |
| ```/test/7visualization/visual.py``` | Visual Analysis of CRISPR-M on the Off-Target Effect Prediction | Fig. 4 |

## CopyRight
This project is licensed under the terms of the MIT license.  
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

## Contact
If you have any questions, please contact us by  
Tel: (86) 22-85358850;  
Fax: (86) 22-85358850;  
Email: jianliu@nankai.edu.cn
