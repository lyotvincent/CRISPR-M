# CRISPR-M
CRISPR-M is a novel multi-view deep learning model with a new feature encoding scheme, regarding sgRNA off-target effect prediction for target sites containing indels and mismatches.

## codes
Here is the introduction of files in ```/codes```.

|file|content|
|----|----|
|```encoding.py```|contains the data encoding functions|
|```metrics_utils.py```|contains the functions computing metrics|
|```data_preprocessing_utils.py```|contains the data preprocessing functions|
|```positional_encoding.py```|contains the PositionalEncoding class|
|```transformer_utils.py```|contains the Transformer classes|
|other files|used for backing up code and are not actually used|

## datasets
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

## test
|folder|usage|
|----|----|
| 1indel | Comparisons on Target Sites Containing Both Mismatches and Indels |
| 2encoding_test | Comparisons on Mismatches-only sgRNA-Target Prediction |
| 3mismatch | Comparisons with Complex Off-Target Site Datasets |
| 4multidataset | Comparisons of Encoding Schemes |
| 6epigenetic | Comparisons with Epigenetic Features |
| 7visualization | Visual Analysis of CRISPR-M on the Off-Target Effect Prediction |
