import pandas as pd

x = [1, 0.4, 0.5, 0.1, 0.1]
y = [0.9, 0.1, 0.7, 0.9, 0.3]

df = pd.DataFrame({"X": x, "Y":y})
print(df)
print(df.corr("spearman"))
print(df.corr("spearman")["X"]["Y"])
print(type(df.corr("spearman")["X"]["Y"]))
