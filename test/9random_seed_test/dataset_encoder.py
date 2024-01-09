import numpy as np
import pandas as pd

pd.options.display.max_columns = None


# DATASETS: CIRCLE, PKD, PDH, SITE, GUIDE_I, GUIDE_II, GUIDE_III


class gRNA_pair_encoder:
    # encoded gRNA and target sequence into seven dimensions binary matrix
    def __init__(self, gRNA_on, target_off, base_dict = None, direct = None, SITE_dataset = False) :

        self.gRNA_on = "-" * (24-len(gRNA_on)) + gRNA_on
        self.target_off = "-" * (24-len(target_off)) + target_off
        self.base_dict = {'A':[1,0,0,0,0],
                          'C':[0,1,0,0,0],
                          'G':[0,0,1,0,0],
                          'T':[0,0,0,1,0],
                          '_':[0,0,0,0,1],
                          '-':[0,0,0,0,0]}
        self.site_base_dict = {'A': [1, 0, 0, 0, 0],
                               'C': [0, 1, 0, 0, 0],
                               'G': [0, 0, 1, 0, 0],
                               'T': [0, 0, 0, 1, 0],
                               'N': [0, 0, 0, 0, 1],
                               '-': [0, 0, 0, 0, 0]}
        self.direct = {'A':1, 'C':2, 'G':3, 'T':4, '_':5}
        self.site_direct = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5,'-':0}

        if SITE_dataset:
            self.encoder_site_pair()
        else:
            self.encoder_pair()


    def encoder_pair(self):

        # on_seq encoder
        seq_on = []
        on_base_list = list(self.gRNA_on)
        for i in range(len(on_base_list)):
            if on_base_list[i] == "N":
                on_base_list[i] = list(self.target_off)[i]
            seq_on.append(self.base_dict[on_base_list[i]])
        self.on_seq = np.array(seq_on)

        # off_seq_encoder
        seq_off = []
        off_base_list = list(self.target_off)
        for i in range(len(off_base_list)):
            seq_off.append(self.base_dict[off_base_list[i]])
        self.off_seq = np.array(seq_off)

        # pair_encoder
        pair_encode = []

        for i in range(len(on_base_list)):
            base_code = np.bitwise_or(self.on_seq[i],self.off_seq[i])
            direct_code = np.zeros(2)
            if on_base_list[i] == "N":
                on_base_list[i] = off_base_list[i]
            if on_base_list[i] == "-" or off_base_list[i] == "-" or self.direct[on_base_list[i]] == self.direct[off_base_list[i]]:
                pass
            else:
                if self.direct[on_base_list[i]] < self.direct[off_base_list[i]]:
                    direct_code[0] = 1
                else:
                    direct_code[1] = 1
            pair_encode.append(np.concatenate((base_code,direct_code)))
        self.pair_code = np.array(pair_encode)

    def encoder_site_pair(self):
        seq_on = []
        on_base_list = list(self.gRNA_on)
        seq_off = []
        off_base_list = list(self.target_off)
        on_base_list[-3] = off_base_list[-3]
        for i in range(len(on_base_list)):
            seq_on.append(self.site_base_dict[on_base_list[i]])
        for i in range(len(off_base_list)):
            seq_off.append(self.site_base_dict[off_base_list[i]])
        self.on_seq = np.array(seq_on)
        self.off_seq = np.array(seq_off)

        pair_encode = []
        for i in range(len(on_base_list)):
            base_code = np.bitwise_or(self.on_seq[i],self.off_seq[i])
            direct_code = np.zeros(2)
            if self.site_direct[on_base_list[i]] == self.site_direct[off_base_list[i]]:
                pass
            else:
                if self.site_direct[on_base_list[i]] < self.site_direct[off_base_list[i]]:
                    direct_code[0] = 1
                else:
                    direct_code[1] = 1
            pair_encode.append(np.concatenate((base_code,direct_code)))
        self.pair_code = np.array(pair_encode)




def encode_CIRCLE_dataset():
    print("Encoding dataset CIRCLE")
    circle_matrix = []
    circle_labels = []
    circle_dataset = pd.read_csv("../datasets/CIRCLE(mismatch:insertion:deletion)/CIRCLE_seq_data.csv")
    for idx, row in circle_dataset.iterrows():
        gRNA_seq = row['sgRNA_seq']
        target_seq = row['off_seq']
        label = row['label']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                      target_off=target_seq)
        circle_matrix.append(pair_codes.pair_code)
        circle_labels.append(label)
    circle_matrix = np.array(circle_matrix)
    circle_labels = np.array(circle_labels)
    print("Encoding dataset CIRCLE into a binary matrix with size of", circle_matrix.shape)
    print("The number of off-target sites in dataset CIRCLE is", len(circle_labels[circle_labels>0]))
    # Encoding dataset CIRCLE into a binary matrix with size of (584949, 24, 7)
    # The number of off-target sites in dataset CIRCLE is 7371
    return circle_matrix, circle_labels


def encode_PKD_dataset():
    print("Encoding dataset PKD")
    pkd_matrix = []
    pkd_label = []
    pkd_dataset = pd.read_pickle("../datasets/PKD/PKD.pkl")
    pkd_dataset = pkd_dataset[0]
    for idx, row in pkd_dataset.iterrows():
        gRNA_seq = row['30mer']
        target_seq = row['30mer_mut']
        label = row['Day21-ETP-binarized']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                      target_off=target_seq)
        pkd_matrix.append(pair_codes.pair_code)
        pkd_label.append(label)
    pkd_label = np.array(pkd_label)
    pkd_matrix = np.array(pkd_matrix)
    print("Encoding dataset PKD into a binary matrix with size of", pkd_matrix.shape)
    print("The number of off-target sites in dataset PKD is", len(pkd_label[pkd_label > 0]))
    # Encoding dataset PKD into a binary matrix with size of (4853, 24, 7)
    # The number of off-target sites in dataset PKD is 2273
    return pkd_matrix, pkd_label


def encode_PDH_dataset():
    print("Encoding dataset PDH")
    pdh_matrix = []
    pdh_count = []
    pdh_dataset = pd.read_pickle("../datasets/PDH/PDH.pkl")
    for idx, row in pdh_dataset.iterrows():
        gRNA_seq = row['30mer']
        target_seq = row['30mer_mut']
        count = row['readFraction']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                       target_off=target_seq)
        pdh_matrix.append(pair_codes.pair_code)
        pdh_count.append(count)
    pdh_count = np.array(pdh_count)
    pdh_label = np.zeros(len(pdh_count))
    pdh_label[pdh_count>0] = 1
    pdh_matrix = np.array(pdh_matrix)
    print("Encoding dataset PDH into a binary matrix with size of", pdh_matrix.shape)
    print("The number of off-target sites in dataset PDH is", len(pdh_label[pdh_label > 0]))
    # Encoding dataset PDH into a binary matrix with size of (10129, 24, 7)
    # The number of off-target sites in dataset PDH is 52
    return pdh_matrix, pdh_label


def encode_SITE_dataset():
    print("Encoding dataset SITE")
    site_matrix = []
    count = []
    site_dataset = pd.read_csv("../datasets/SITE/SITE.csv")
    for idx, row in site_dataset.iterrows():
        gRNA_seq = '-' + row['on_seq'].upper()
        target_seq = '-' + row['off_seq'].upper()
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                       target_off=target_seq,
                                       SITE_dataset = True)
        site_matrix.append(pair_codes.pair_code)
        count.append(row['reads'])
    count = np.array(count)
    site_label = np.zeros(len(count))
    site_label[count > 0] = 1
    site_matrix = np.array(site_matrix)
    print("Encoding dataset SITE into a binary matrix with size of", site_matrix.shape)
    print("The number of off-target sites in dataset SITE is", len(site_label[site_label > 0]))
    # Encoding dataset SITE into a binary matrix with size of (217733, 24, 7)
    # The number of off-target sites in dataset SITE is 3767
    return site_matrix, site_label


def encode_GUIDE_I_dataset():
    print("Encoding dataset GUIDE_I")
    guide_matrix = []
    guide_count = []
    guide_dataset = pd.read_pickle("../datasets/GUIDE_I/GUIDE_I.pkl")
    for idx, row in guide_dataset.iterrows():
        gRNA_seq = row['30mer']
        target_seq = row['30mer_mut']
        count = row['GUIDE-SEQ Reads']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                       target_off=target_seq)
        guide_matrix.append(pair_codes.pair_code)
        guide_count.append(count)
    guide_count = np.array(guide_count)
    guide_label = np.zeros(len(guide_count))
    guide_label[guide_count>0] = 1
    guide_matrix = np.array(guide_matrix)
    print("Encoding dataset GUIDE_I into a binary matrix with size of", guide_matrix.shape)
    print("The number of off-target sites in dataset GUIDE_I is", len(guide_label[guide_label > 0]))
    # Encoding dataset GUIDE_I into a binary matrix with size of (294534, 24, 7)
    # The number of off-target sites in dataset GUIDE_I is 354
    return guide_matrix, guide_label


def encode_GUIDE_II_dataset():
    print("Encoding dataset GUIDE_II")
    guide_matrix = []
    guide_label = []
    guide_dataset = pd.read_csv("../datasets/GUIDE_II/GUIDE_II.csv")
    for idx, row in guide_dataset.iterrows():
        gRNA_seq = row['sgRNA_seq'].upper()
        target_seq = row['off_seq'].upper()
        label = row['label']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                       target_off=target_seq)
        guide_matrix.append(pair_codes.pair_code)
        guide_label.append(label)
    guide_label = np.array(guide_label)
    guide_matrix = np.array(guide_matrix)
    print("Encoding dataset GUIDE_II into a binary matrix with size of", guide_matrix.shape)
    print("The number of off-target sites in dataset GUIDE_II is", len(guide_label[guide_label > 0]))
    # Encoding dataset GUIDE_II into a binary matrix with size of (95829, 24, 7)
    # The number of off-target sites in dataset GUIDE_II is 54
    return guide_matrix, guide_label


def encode_GUIDE_III_dataset():
    print("Encoding dataset GUIDE_III")
    guide_matrix = []
    guide_label = []
    guide_dataset = pd.read_csv("../datasets/GUIDE_III/GUIDE_III.csv")
    for idx, row in guide_dataset.iterrows():
        gRNA_seq = row['sgRNA_seq'].upper()
        target_seq = row['off_seq'].upper()
        label = row['label']
        pair_codes = gRNA_pair_encoder(gRNA_on=gRNA_seq,
                                       target_off=target_seq)
        guide_matrix.append(pair_codes.pair_code)
        guide_label.append(label)
    guide_label = np.array(guide_label)
    guide_matrix = np.array(guide_matrix)
    print("Encoding dataset GUIDE_III into a binary matrix with size of", guide_matrix.shape)
    print("The number of off-target sites in dataset GUIDE_III is", len(guide_label[guide_label > 0]))
    # Encoding dataset GUIDE_III into a binary matrix with size of (383463, 24, 7)
    # The number of off-target sites in dataset GUIDE_III is 56
    return guide_matrix, guide_label







