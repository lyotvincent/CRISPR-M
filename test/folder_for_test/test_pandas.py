import numpy, pickle
import pandas as pd

x = [
    [[1,2,3], [4,5,6]],
    [[11,22,33], [44,55,66]]
]


# x = pd.DataFrame(x, columns=["tpr", "xr"])
# x.to_csv("test_pandas.csv", index=False)

# test_pandas_list = pd.read_csv("test_pandas.csv")
# print(test_pandas_list)
# print(test_pandas_list.shape)
# test_pandas_list = numpy.array(test_pandas_list)
# print(test_pandas_list)
# print(test_pandas_list.shape)

# x = numpy.array(x)
# with open("test_save.csv", "wb") as f:
#     pickle.dump(x, f)
# with open("test_save.csv", "rb") as f:
#     x = pickle.load(f)
# print(x)
# print(x.shape)

import os
data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+r"/datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv"
circle_dataset = pd.read_csv(data_path)
n0, n1 = 0, 0
for i, row in circle_dataset.iterrows():
    gRNA_seq = row['sgRNA_seq']
    target_seq = row['off_seq']
    label = row['label']
    if "_" in gRNA_seq or "_" in target_seq:
        if label >0:
            n0 += 1
        else:
            n1 += 1
print(n0, n1)