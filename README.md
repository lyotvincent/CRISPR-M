# CRISPR-M
CRISPR-M is a novel multi-view deep learning model with a new feature encoding scheme, regarding sgRNA off-target effect prediction for target sites containing indels and mismatches.

## /codes
Here is the introduction of files in ```/codes```.

|file|content|
|----|----|
|```encoding.py```|contains the data encoding functions|
|```metrics_utils.py```|contains the functions computing metrics|
|```data_preprocessing_utils.py```|contains the data preprocessing functions|
|```positional_encoding.py```|contains the PositionalEncoding class|
|```transformer_utils.py```|contains the Transformer classes|
|other files|used for backing up code and are not actually used|

## /datasets
The associations between dataset names in github and dataset names in paper.  

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
Here are the experiments corresponding to each folder in ```test```.

|folder|usage|
|----|----|
| 1indel | Comparisons on Target Sites Containing Both Mismatches and Indels |
| 2encoding_test | Comparisons on Mismatches-only sgRNA-Target Prediction |
| 3mismatch | Comparisons with Complex Off-Target Site Datasets |
| 4multidataset | Comparisons of Encoding Schemes |
| 6epigenetic | Comparisons with Epigenetic Features |
| 7visualization | Visual Analysis of CRISPR-M on the Off-Target Effect Prediction |
| other folders | discard |

Take folder-2encoding_test as an example, ```encoding_test.py``` in the folder-mine is the main program of the test, one could run ```python encoding_test.py``` for run it. The ```test_model.py``` contains the model architecture used for the test. The model in function 'm81212_n13' of ```test_model.py``` is final model of CRISPR-M. ```fig2.py``` in folder-fig2 is the visualization program that visualizes the results of several experiments

## experiment environment
#### run model
- python3.8
- tensorflow-2.9
- keras-2.9
- pandas-1.4
- numpy-1.22
- scikit-learn-1.1
#### visualization
- matploblib-3.5
- seaborn-0.11

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