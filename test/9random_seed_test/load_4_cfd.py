import os, sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_PKD(out_dim=1):
    print("[INFO] ===== Start Loading dataset PKD =====")
    pkd_features_on = []
    pkd_features_off = []
    pkd_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/PKD/PKD.csv"
    pkd_dataset = pd.read_csv(data_path)
    # print(pkd_dataset)
    ## shuffle
    pkd_dataset = shuffle(pkd_dataset, random_state=42)
    ## encode
    for i, row in pkd_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['Day21-ETP-binarized']
        pkd_features_on.append(on_target_seq)
        pkd_features_off.append(off_target_seq)
        pkd_labels.append(label)
    pkd_features_on = np.array(pkd_features_on)
    pkd_features_off = np.array(pkd_features_off)
    pkd_labels = np.array(pkd_labels)
    print("[INFO] Encoded dataset PKD pkd_features_on with size of", pkd_features_on.shape)
    print("[INFO] Encoded dataset PKD pkd_features_off with size of", pkd_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset PKD is {0}, the active+inactive is {1}.".format(len(pkd_labels[pkd_labels>0]), len(pkd_labels)))
    return pkd_features_on, pkd_features_off, pkd_labels

def load_PDH(out_dim=1):
    print("[INFO] ===== Start Loading dataset PDH =====")
    pdh_features_on = []
    pdh_features_off = []
    pdh_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/PDH/PDH.csv"
    pdh_dataset = pd.read_csv(data_path)
    # print(pdh_dataset)
    ## shuffle
    pdh_dataset = shuffle(pdh_dataset, random_state=42)
    ## encode
    for i, row in pdh_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['readFraction']
        pdh_features_on.append(on_target_seq)
        pdh_features_off.append(off_target_seq)
        pdh_labels.append(label)
    pdh_features_on = np.array(pdh_features_on)
    pdh_features_off = np.array(pdh_features_off)
    pdh_labels = np.array(pdh_labels)
    print("[INFO] Encoded dataset PDH pdh_features_on with size of", pdh_features_on.shape)
    print("[INFO] Encoded dataset PDH pdh_features_off with size of", pdh_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset PDH is {0}, the active+inactive is {1}.".format(len(pdh_labels[pdh_labels>0]), len(pdh_labels)))
    return pdh_features_on, pdh_features_off, pdh_labels

def load_SITE(out_dim=1):
    print("[INFO] ===== Start Loading dataset SITE =====")
    site_features_on = []
    site_features_off = []
    site_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/SITE/SITE.csv"
    site_dataset = pd.read_csv(data_path)
    # print(site_dataset)
    ## shuffle
    site_dataset = shuffle(site_dataset, random_state=42)
    ## encode
    for i, row in site_dataset.iterrows():
        on_target_seq = row['on_seq']
        off_target_seq = row['off_seq']
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        on_target_seq = on_target_seq[:-3]+off_target_seq[-3]+on_target_seq[-2:]
        label = row['reads']
        if label > 0:
            label = 1.0
        site_features_on.append(on_target_seq)
        site_features_off.append(off_target_seq)
        site_labels.append(label)
    site_features_on = np.array(site_features_on)
    site_features_off = np.array(site_features_off)
    site_labels = np.array(site_labels)
    print("[INFO] Encoded dataset site site_features_on with size of", site_features_on.shape)
    print("[INFO] Encoded dataset site site_features_off with size of", site_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset site is {0}, the active+inactive is {1}.".format(len(site_labels[site_labels>0]), len(site_labels)))
    return site_features_on, site_features_off, site_labels

def load_GUIDE_I(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_I =====")
    guide_i_features_on = []
    guide_i_features_off = []
    guide_i_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/GUIDE_I/GUIDE_I.csv"
    guide_i_dataset = pd.read_csv(data_path)
    # print(guide_i_dataset)
    ## shuffle
    guide_i_dataset = shuffle(guide_i_dataset, random_state=42)
    ## encode
    for i, row in guide_i_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['GUIDE-SEQ Reads']
        guide_i_features_on.append(on_target_seq)
        guide_i_features_off.append(off_target_seq)
        guide_i_labels.append(label)
    guide_i_features_on = np.array(guide_i_features_on)
    guide_i_features_off = np.array(guide_i_features_off)
    guide_i_labels = np.array(guide_i_labels)
    print("[INFO] Encoded dataset guide_i guide_i_features_on with size of", guide_i_features_on.shape)
    print("[INFO] Encoded dataset guide_i guide_i_features_off with size of", guide_i_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_i is {0}, the active+inactive is {1}.".format(len(guide_i_labels[guide_i_labels>0]), len(guide_i_labels)))
    return guide_i_features_on, guide_i_features_off, guide_i_labels

def load_GUIDE_II(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_II =====")
    guide_ii_features_on = []
    guide_ii_features_off = []
    guide_ii_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/GUIDE_II/GUIDE_II.csv"
    guide_ii_dataset = pd.read_csv(data_path)
    # print(guide_ii_dataset)
    ## shuffle
    guide_ii_dataset = shuffle(guide_ii_dataset, random_state=42)
    ## encode
    for i, row in guide_ii_dataset.iterrows():
        on_target_seq = row['sgRNA_seq'].upper()
        off_target_seq = row['off_seq'].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        on_target_seq = on_target_seq[:-3]+off_target_seq[-3]+on_target_seq[-2:]
        label = row['label']
        guide_ii_features_on.append(on_target_seq)
        guide_ii_features_off.append(off_target_seq)
        guide_ii_labels.append(label)
    guide_ii_features_on = np.array(guide_ii_features_on)
    guide_ii_features_off = np.array(guide_ii_features_off)
    guide_ii_labels = np.array(guide_ii_labels)
    print("[INFO] Encoded dataset guide_ii guide_ii_features_on with size of", guide_ii_features_on.shape)
    print("[INFO] Encoded dataset guide_ii guide_ii_features_off with size of", guide_ii_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_ii is {0}, the active+inactive is {1}.".format(len(guide_ii_labels[guide_ii_labels>0]), len(guide_ii_labels)))
    return guide_ii_features_on, guide_ii_features_off, guide_ii_labels

def load_GUIDE_III(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_III =====")
    guide_iii_features_on = []
    guide_iii_features_off = []
    guide_iii_labels = []
    ## load
    data_path = SRC_DIR+r"/datasets/GUIDE_III/GUIDE_III.csv"
    guide_iii_dataset = pd.read_csv(data_path)
    # print(guide_iii_dataset)
    ## shuffle
    guide_iii_dataset = shuffle(guide_iii_dataset, random_state=42)
    ## encode
    for i, row in guide_iii_dataset.iterrows():
        on_target_seq = row['sgRNA_seq'].upper()
        off_target_seq = row['off_seq'].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        on_target_seq = on_target_seq[:-3]+off_target_seq[-3]+on_target_seq[-2:]
        label = row['label']
        guide_iii_features_on.append(on_target_seq)
        guide_iii_features_off.append(off_target_seq)
        guide_iii_labels.append(label)
    guide_iii_features_on = np.array(guide_iii_features_on)
    guide_iii_features_off = np.array(guide_iii_features_off)
    guide_iii_labels = np.array(guide_iii_labels)
    print("[INFO] Encoded dataset guide_iii guide_iii_features_on with size of", guide_iii_features_on.shape)
    print("[INFO] Encoded dataset guide_iii guide_iii_features_off with size of", guide_iii_features_off.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_iii is {0}, the active+inactive is {1}.".format(len(guide_iii_labels[guide_iii_labels>0]), len(guide_iii_labels)))
    return guide_iii_features_on, guide_iii_features_off, guide_iii_labels

if __name__=="__main__":
    features, features2, labels=load_GUIDE_III()
    print(max(labels))
    print(min(labels))
    is_reg = False
    for i in labels:
        if 0<i<1:
            is_reg = True
            break
    print(is_reg)
