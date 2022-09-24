import numpy as np
import pickle
def txt2pkl(n):
    f = open("./rocprc2.7/"+n+".txt", 'r')
    lines = f.readlines()
    f.close()
    l1 = lines[0].strip().strip(",").split(",")
    l1 = [i for i in l1]
    l1 = np.array(l1, np.float32)
    print(len(l1))
    l2 = lines[1].strip().strip(",").split(",")
    l2 = [i for i in l2]
    l2 = np.array(l2, np.float32)
    print(len(l2))
    l = list()
    l.append(l1)
    l.append(l2)
    with open("./"+n+".csv", "wb") as f:
        pickle.dump(l, f)

txt2pkl("tpr")
txt2pkl("fpr")
txt2pkl("precision_point")
txt2pkl("recall_point")
