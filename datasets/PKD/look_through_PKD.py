
import pandas as pd


pkd_dataset = pd.read_pickle("./PKD.pkl")
print(pkd_dataset)
pkd_dataset = pkd_dataset[0]
print(pkd_dataset)