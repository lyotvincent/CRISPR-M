import os, pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from encoding import encode_by_crispr_net_method, encode_in_6_dimensions, encode_by_base_pair_vocabulary, encode_by_base_vocabulary

def load_CIRCLE_dataset(encoding_method=encode_by_base_pair_vocabulary, sgRNA_type_index=None, out_dim=1, ten_grna_fold_CIRLCE=None, MAX_STEPS=24, EMBED_SIZE=6, is_split=False):
    print("[INFO] ===== Start Loading dataset CIRCLE =====")
    circle_feature = []
    circle_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv"
    circle_dataset = pd.read_csv(data_path)
    # 判断是否按照sgRNA分组
    if sgRNA_type_index == None: # 不按照sgRNA分组，随机分
        ## shuffle
        circle_dataset = shuffle(circle_dataset, random_state=42)
        ## encode
        for i, row in circle_dataset.iterrows():
            gRNA_seq = row['sgRNA_seq']
            target_seq = row['off_seq']
            label = row['label']
            pair_code = encoding_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
            circle_feature.append(pair_code)
            circle_labels.append(label)
        circle_feature = np.array(circle_feature)
        #if encoding_method == encode_in_6_dimensions:
        # circle_feature = circle_feature.reshape(circle_feature.shape[0], 1, MAX_STEPS, EMBED_SIZE)
        # print(circle_feature[-3:])
        if out_dim == 2:
            circle_labels = to_categorical(circle_labels)
        circle_labels = np.array(circle_labels)
        # circle_labels = to_categorical(circle_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
        # print(circle_labels[-3:])
        print("[INFO] Encoded dataset CIRCLE feature with size of", circle_feature.shape)
        print("[INFO] The labels number of active off-target sites in dataset CIRCLE is {0}, the active+inactive is {1}.".format(len(circle_labels[circle_labels>0]), len(circle_labels)))
        ## split
        if is_split:
            train_features, validation_features, train_labels, validation_labels = train_test_split(circle_feature, circle_labels, test_size=0.2, random_state=42, stratify=circle_labels)
            return train_features, validation_features, train_labels, validation_labels
        else:
            return circle_feature, circle_labels
    else: # 按照sgRNA分组
        SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
        train_features = list()
        train_labels = list()
        validation_features = list()
        validation_labels = list()
        if len(ten_grna_fold_CIRLCE[0]["features"]) != 0: # 如果已经分好了就直接从里面取
            for i in range(10):
                if i != sgRNA_type_index:
                    print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[sgRNA_type_index]))
                    train_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
                else:
                    print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[sgRNA_type_index]))
                    validation_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
        else: # 如果是none就新分配十份
            print("[INFO] ===== Start Encoding 10-grna-fold dataset CIRCLE =====")
            # CIRCLE dataset里一共十种sgRNA，可以用来作10-fold交叉验证
            for i in range(10):
                print("[INFO] generating %s-th grna-fold (%s) features & labels"%(i, SGRNA_TYPE[i]))
                # extract i-th fold
                one_fold_dataset = circle_dataset[circle_dataset['sgRNA_type']==SGRNA_TYPE[i]]
                # encode i-th fold
                for _, row in one_fold_dataset.iterrows():
                    gRNA_seq = row['sgRNA_seq']
                    target_seq = row['off_seq']
                    label = row['label']
                    pair_code = encoding_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
                    ten_grna_fold_CIRLCE[i]["features"].append(pair_code)
                    ten_grna_fold_CIRLCE[i]["labels"].append(label)
                ten_grna_fold_CIRLCE[i]["features"] = np.array(ten_grna_fold_CIRLCE[i]["features"])
                if encoding_method == encode_in_6_dimensions:
                    ten_grna_fold_CIRLCE[i]["features"] = ten_grna_fold_CIRLCE[i]["features"].reshape(ten_grna_fold_CIRLCE[i]["features"].shape[0], 1, MAX_STEPS, EMBED_SIZE)
                if out_dim == 2:
                    ten_grna_fold_CIRLCE[i]["labels"] = to_categorical(ten_grna_fold_CIRLCE[i]["labels"])
                ten_grna_fold_CIRLCE[i]["labels"] = np.array(ten_grna_fold_CIRLCE[i]["labels"])
                print("[INFO] %s-th-grna-fold set, features shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["features"].shape)))
                print("[INFO] %s-th-grna-fold set, labels shape=%s, and positive samples number = %s"%(i, str(ten_grna_fold_CIRLCE[i]["labels"].shape), len(ten_grna_fold_CIRLCE[i]["labels"][ten_grna_fold_CIRLCE[i]["labels"]>0])))
                if i != sgRNA_type_index:
                    print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[i]))
                    train_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
                else:
                    print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[i]))
                    validation_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
            print("[INFO] ===== End Encoding 10-grna-fold dataset CIRCLE =====")
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        validation_features = np.array(validation_features)
        validation_labels = np.array(validation_labels)
        print("[INFO] train_feature.shape = %s"%str(train_features.shape))
        print("[INFO] train_labels.shape = %s, and positive samples number = %s"%(train_labels.shape, len(train_labels[train_labels>0])))
        print("[INFO] validation_features.shape = %s"%str(validation_features.shape))
        print("[INFO] validation_labels.shape = %s, and positive samples number = %s"%(validation_labels.shape, len(validation_labels[validation_labels>0])))
        print("[INFO] ===== End Loading dataset CIRCLE =====")
        return ten_grna_fold_CIRLCE, train_features, train_labels, validation_features, validation_labels

def load_I_2_dataset(encoding_method=encode_by_base_pair_vocabulary, out_dim=1, MAX_STEPS=24, EMBED_SIZE=6, is_split=False):
    print("[INFO] ===== Start Loading dataset I_2 =====")
    I_2_feature = []
    I_2_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/dataset_I-2/elevation_6gRNA_wholeDataset.csv"
    I_2_dataset = pd.read_csv(data_path)
    ## shuffle
    I_2_dataset = shuffle(I_2_dataset, random_state=42)
    ## encode
    for i, row in I_2_dataset.iterrows():
        gRNA_seq = row['crRNA']
        target_seq = row['DNA']
        label = row['label']
        pair_code = encoding_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
        I_2_feature.append(pair_code)
        I_2_labels.append(label)
    I_2_feature = np.array(I_2_feature)
    #if encoding_method == encode_in_6_dimensions:
    # I_2_feature = I_2_feature.reshape(I_2_feature.shape[0], 1, MAX_STEPS, EMBED_SIZE)
    # print(I_2_feature[-3:])
    if out_dim == 2:
        I_2_labels = to_categorical(I_2_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    I_2_labels = np.array(I_2_labels)
    # print(I_2_labels[-3:])
    print("[INFO] Encoded dataset I_2 feature with size of", I_2_feature.shape)
    print("[INFO] The labels number of active off-target sites in dataset I_2 is {0}, the active+inactive is {1}.".format(len(I_2_labels[I_2_labels>0]), len(I_2_labels)))
    ## split
    if is_split:
        train_features, validation_features, train_labels, validation_labels = train_test_split(I_2_feature, I_2_labels, test_size=0.2, random_state=42, stratify=I_2_labels)
        return train_features, validation_features, train_labels, validation_labels
    else:
        return I_2_feature, I_2_labels

def load_CIRCLE_dataset_encoded_by_base_vocabulary(sgRNA_type_index=None, out_dim=1, ten_grna_fold_CIRLCE=None):
    print("[INFO] ===== Start Loading dataset CIRCLE =====")
    circle_feature = []
    circle_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv"
    circle_dataset = pd.read_csv(data_path)
    SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
    train_feature_ont = list()
    train_feature_offt = list()
    train_labels = list()
    validation_feature_ont = list()
    validation_feature_offt = list()
    validation_labels = list()
    if len(ten_grna_fold_CIRLCE[0]["feature_ont"]) != 0: # 如果已经分好了就直接从里面取
        for i in range(10):
            if i != sgRNA_type_index:
                print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[sgRNA_type_index]))
                train_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                train_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
            else:
                print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[sgRNA_type_index]))
                validation_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                validation_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
    else: # 如果是none就新分配十份
        print("[INFO] ===== Start Encoding 10-grna-fold dataset CIRCLE =====")
        # CIRCLE dataset里一共十种sgRNA，可以用来作10-fold交叉验证
        for i in range(10):
            print("[INFO] generating %s-th grna-fold (%s) features & labels"%(i, SGRNA_TYPE[i]))
            # extract i-th fold
            one_fold_dataset = circle_dataset[circle_dataset['sgRNA_type']==SGRNA_TYPE[i]]
            # encode i-th fold
            for _, row in one_fold_dataset.iterrows():
                on_target_seq = row['sgRNA_seq']
                off_target_seq = row['off_seq']
                label = row['label']
                ten_grna_fold_CIRLCE[i]["feature_ont"].append(encode_by_base_vocabulary(seq=on_target_seq))
                ten_grna_fold_CIRLCE[i]["feature_offt"].append(encode_by_base_vocabulary(seq=off_target_seq))
                ten_grna_fold_CIRLCE[i]["labels"].append(label)
            ten_grna_fold_CIRLCE[i]["feature_ont"] = np.array(ten_grna_fold_CIRLCE[i]["feature_ont"])
            ten_grna_fold_CIRLCE[i]["feature_offt"] = np.array(ten_grna_fold_CIRLCE[i]["feature_offt"])
            if out_dim == 2:
                ten_grna_fold_CIRLCE[i]["labels"] = to_categorical(ten_grna_fold_CIRLCE[i]["labels"])
            ten_grna_fold_CIRLCE[i]["labels"] = np.array(ten_grna_fold_CIRLCE[i]["labels"])
            print("[INFO] %s-th-grna-fold set, feature_ont shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["feature_ont"].shape)))
            print("[INFO] %s-th-grna-fold set, feature_offt shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["feature_offt"].shape)))
            print("[INFO] %s-th-grna-fold set, labels shape=%s, and positive samples number = %s"%(i, str(ten_grna_fold_CIRLCE[i]["labels"].shape), len(ten_grna_fold_CIRLCE[i]["labels"][ten_grna_fold_CIRLCE[i]["labels"]>0])))
            if i != sgRNA_type_index:
                print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[i]))
                train_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                train_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
            else:
                print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[i]))
                validation_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                validation_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
        print("[INFO] ===== End Encoding 10-grna-fold dataset CIRCLE =====")
    train_feature_ont = np.array(train_feature_ont)
    train_feature_offt = np.array(train_feature_offt)
    train_labels = np.array(train_labels)
    validation_feature_ont = np.array(validation_feature_ont)
    validation_feature_offt = np.array(validation_feature_offt)
    validation_labels = np.array(validation_labels)
    print("[INFO] train_feature_ont.shape = %s"%str(train_feature_ont.shape))
    print("[INFO] train_feature_offt.shape = %s"%str(train_feature_offt.shape))
    print("[INFO] train_labels.shape = %s, and positive samples number = %s"%(train_labels.shape, len(train_labels[train_labels>0])))
    print("[INFO] validation_feature_ont.shape = %s"%str(validation_feature_ont.shape))
    print("[INFO] validation_feature_offt.shape = %s"%str(validation_feature_offt.shape))
    print("[INFO] validation_labels.shape = %s, and positive samples number = %s"%(validation_labels.shape, len(validation_labels[validation_labels>0])))
    print("[INFO] ===== End Loading dataset CIRCLE =====")
    return ten_grna_fold_CIRLCE, train_feature_ont, train_feature_offt, train_labels, validation_feature_ont, validation_feature_offt, validation_labels

def load_CIRCLE_dataset_encoded_by_both_base_and_base_pair(sgRNA_type_index=None, out_dim=1, ten_grna_fold_CIRLCE=None):
    print("[INFO] ===== Start Loading dataset CIRCLE =====")
    circle_features = []
    circle_feature_ont = []
    circle_feature_offt = []
    circle_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv"
    circle_dataset = pd.read_csv(data_path)
    if sgRNA_type_index == None: # 不按照sgRNA分组，随机分
        ## shuffle
        circle_dataset = shuffle(circle_dataset, random_state=42)
        ## encode
        for i, row in circle_dataset.iterrows():
            on_target_seq = row['sgRNA_seq']
            off_target_seq = row['off_seq']
            label = row['label']
            circle_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
            circle_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
            circle_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
            circle_labels.append(label)
        circle_features = np.array(circle_features)
        circle_feature_ont = np.array(circle_feature_ont)
        circle_feature_offt = np.array(circle_feature_offt)
        if out_dim == 2:
            circle_labels = to_categorical(circle_labels)
        circle_labels = np.array(circle_labels)
        print("[INFO] Encoded dataset CIRCLE features with size of", circle_features.shape)
        print("[INFO] Encoded dataset CIRCLE feature ont with size of", circle_feature_ont.shape)
        print("[INFO] Encoded dataset CIRCLE feature offt with size of", circle_feature_offt.shape)
        print("[INFO] The labels number of active off-target sites in dataset CIRCLE is {0}, the active+inactive is {1}.".format(len(circle_labels[circle_labels>0]), len(circle_labels)))
        return circle_features, circle_feature_ont, circle_feature_offt, circle_labels
    else: # 按照sgRNA分组
        SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
        train_features = list()
        train_feature_ont = list()
        train_feature_offt = list()
        train_labels = list()
        validation_features = list()
        validation_feature_ont = list()
        validation_feature_offt = list()
        validation_labels = list()
        if len(ten_grna_fold_CIRLCE[0]["feature_ont"]) != 0: # 如果已经分好了就直接从里面取
            for i in range(10):
                if i != sgRNA_type_index:
                    print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[sgRNA_type_index]))
                    train_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    train_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                    train_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                    train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
                else:
                    print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[sgRNA_type_index]))
                    validation_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    validation_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                    validation_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                    validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
        else: # 如果是none就新分配十份
            print("[INFO] ===== Start Encoding 10-grna-fold dataset CIRCLE =====")
            # CIRCLE dataset里一共十种sgRNA，可以用来作10-fold交叉验证
            for i in range(10):
                print("[INFO] generating %s-th grna-fold (%s) features & labels"%(i, SGRNA_TYPE[i]))
                # extract i-th fold
                one_fold_dataset = circle_dataset[circle_dataset['sgRNA_type']==SGRNA_TYPE[i]]
                # encode i-th fold
                for _, row in one_fold_dataset.iterrows():
                    on_target_seq = row['sgRNA_seq']
                    off_target_seq = row['off_seq']
                    label = row['label']
                    ten_grna_fold_CIRLCE[i]["features"].append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
                    ten_grna_fold_CIRLCE[i]["feature_ont"].append(encode_by_base_vocabulary(seq=on_target_seq))
                    ten_grna_fold_CIRLCE[i]["feature_offt"].append(encode_by_base_vocabulary(seq=off_target_seq))
                    ten_grna_fold_CIRLCE[i]["labels"].append(label)
                ten_grna_fold_CIRLCE[i]["features"] = np.array(ten_grna_fold_CIRLCE[i]["features"])
                ten_grna_fold_CIRLCE[i]["feature_ont"] = np.array(ten_grna_fold_CIRLCE[i]["feature_ont"])
                ten_grna_fold_CIRLCE[i]["feature_offt"] = np.array(ten_grna_fold_CIRLCE[i]["feature_offt"])
                if out_dim == 2:
                    ten_grna_fold_CIRLCE[i]["labels"] = to_categorical(ten_grna_fold_CIRLCE[i]["labels"])
                ten_grna_fold_CIRLCE[i]["labels"] = np.array(ten_grna_fold_CIRLCE[i]["labels"])
                print("[INFO] %s-th-grna-fold set, features shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["features"].shape)))
                print("[INFO] %s-th-grna-fold set, feature_ont shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["feature_ont"].shape)))
                print("[INFO] %s-th-grna-fold set, feature_offt shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["feature_offt"].shape)))
                print("[INFO] %s-th-grna-fold set, labels shape=%s, and positive samples number = %s"%(i, str(ten_grna_fold_CIRLCE[i]["labels"].shape), len(ten_grna_fold_CIRLCE[i]["labels"][ten_grna_fold_CIRLCE[i]["labels"]>0])))
                if i != sgRNA_type_index:
                    print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[i]))
                    train_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    train_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                    train_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                    train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
                else:
                    print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[i]))
                    validation_features.extend(ten_grna_fold_CIRLCE[i]["features"])
                    validation_feature_ont.extend(ten_grna_fold_CIRLCE[i]["feature_ont"])
                    validation_feature_offt.extend(ten_grna_fold_CIRLCE[i]["feature_offt"])
                    validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
            print("[INFO] ===== End Encoding 10-grna-fold dataset CIRCLE =====")
        train_features = np.array(train_features)
        train_feature_ont = np.array(train_feature_ont)
        train_feature_offt = np.array(train_feature_offt)
        train_labels = np.array(train_labels)
        validation_features = np.array(validation_features)
        validation_feature_ont = np.array(validation_feature_ont)
        validation_feature_offt = np.array(validation_feature_offt)
        validation_labels = np.array(validation_labels)
        print("[INFO] train_features.shape = %s"%str(train_features.shape))
        print("[INFO] train_feature_ont.shape = %s"%str(train_feature_ont.shape))
        print("[INFO] train_feature_offt.shape = %s"%str(train_feature_offt.shape))
        print("[INFO] train_labels.shape = %s, and positive samples number = %s"%(train_labels.shape, len(train_labels[train_labels>0])))
        print("[INFO] validation_features.shape = %s"%str(validation_features.shape))
        print("[INFO] validation_feature_ont.shape = %s"%str(validation_feature_ont.shape))
        print("[INFO] validation_feature_offt.shape = %s"%str(validation_feature_offt.shape))
        print("[INFO] validation_labels.shape = %s, and positive samples number = %s"%(validation_labels.shape, len(validation_labels[validation_labels>0])))
        print("[INFO] ===== End Loading dataset CIRCLE =====")
        return ten_grna_fold_CIRLCE, train_features, train_feature_ont, train_feature_offt, train_labels, validation_features, validation_feature_ont, validation_feature_offt, validation_labels

def load_I_2_dataset_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset I_2 =====")
    I_2_features = []
    I_2_feature_ont = []
    I_2_feature_offt = []
    I_2_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/dataset_I-2/elevation_6gRNA_wholeDataset.csv"
    I_2_dataset = pd.read_csv(data_path)
    ## shuffle
    I_2_dataset = shuffle(I_2_dataset, random_state=42)
    ## encode
    for i, row in I_2_dataset.iterrows():
        on_target_seq = row['crRNA']
        off_target_seq = row['DNA']
        label = row['label']
        I_2_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        I_2_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        I_2_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        I_2_labels.append(label)
    I_2_features = np.array(I_2_features)
    I_2_feature_ont = np.array(I_2_feature_ont)
    I_2_feature_offt = np.array(I_2_feature_offt)
    if out_dim == 2:
        I_2_labels = to_categorical(I_2_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    I_2_labels = np.array(I_2_labels)
    print("[INFO] Encoded dataset I_2 features with size of", I_2_features.shape)
    print("[INFO] Encoded dataset I_2 feature ont with size of", I_2_feature_ont.shape)
    print("[INFO] Encoded dataset I_2 feature offt with size of", I_2_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset I_2 is {0}, the active+inactive is {1}.".format(len(I_2_labels[I_2_labels>0]), len(I_2_labels)))
    return I_2_features, I_2_feature_ont, I_2_feature_offt, I_2_labels

def load_PKD(encoding_method, out_dim=1, is_binarized=True):
    print("[INFO] ===== Start Loading dataset PKD =====")
    pkd_features = []
    pkd_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/PKD/PKD.csv"
    pkd_dataset = pd.read_csv(data_path)
    # print(pkd_dataset)
    ## shuffle
    pkd_dataset = shuffle(pkd_dataset, random_state=42)
    ## encode
    if is_binarized:
        label_name = 'Day21-ETP-binarized'
    else:
        label_name = 'Day21-ETP'
    for i, row in pkd_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row[label_name]
        pkd_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        pkd_labels.append(label)
    pkd_features = np.array(pkd_features)
    if out_dim == 2:
        pkd_labels = to_categorical(pkd_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    pkd_labels = np.array(pkd_labels)
    print("[INFO] Encoded dataset PKD features with size of", pkd_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset PKD is {0}, the active+inactive is {1}.".format(len(pkd_labels[pkd_labels>0]), len(pkd_labels)))
    return pkd_features, pkd_labels

def load_PKD_encoded_by_both_base_and_base_pair(out_dim=1, is_binarized=True):
    print("[INFO] ===== Start Loading dataset PKD =====")
    pkd_features = []
    pkd_feature_ont = []
    pkd_feature_offt = []
    pkd_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/PKD/PKD.csv"
    pkd_dataset = pd.read_csv(data_path)
    # print(pkd_dataset)
    ## shuffle
    pkd_dataset = shuffle(pkd_dataset, random_state=42)
    ## encode
    if is_binarized:
        label_name = 'Day21-ETP-binarized'
    else:
        label_name = 'Day21-ETP'
    for i, row in pkd_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row[label_name]
        pkd_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        pkd_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        pkd_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        pkd_labels.append(label)
    pkd_features = np.array(pkd_features)
    pkd_feature_ont = np.array(pkd_feature_ont)
    pkd_feature_offt = np.array(pkd_feature_offt)
    if out_dim == 2:
        pkd_labels = to_categorical(pkd_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    pkd_labels = np.array(pkd_labels)
    print("[INFO] Encoded dataset PKD features with size of", pkd_features.shape)
    print("[INFO] Encoded dataset PKD feature ont with size of", pkd_feature_ont.shape)
    print("[INFO] Encoded dataset PKD feature offt with size of", pkd_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset PKD is {0}, the active+inactive is {1}.".format(len(pkd_labels[pkd_labels>0]), len(pkd_labels)))
    return pkd_features, pkd_feature_ont, pkd_feature_offt, pkd_labels

def load_PDH(encoding_method, out_dim=1):
    print("[INFO] ===== Start Loading dataset PDH =====")
    pdh_features = []
    pdh_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/PDH/PDH.csv"
    pdh_dataset = pd.read_csv(data_path)
    # print(pdh_dataset)
    ## shuffle
    pdh_dataset = shuffle(pdh_dataset, random_state=42)
    ## encode
    for i, row in pdh_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['readFraction']
        pdh_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        pdh_labels.append(label)
    pdh_features = np.array(pdh_features)
    if out_dim == 2:
        pdh_labels = to_categorical(pdh_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    pdh_labels = np.array(pdh_labels)
    print("[INFO] Encoded dataset PDH features with size of", pdh_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset PDH is {0}, the active+inactive is {1}.".format(len(pdh_labels[pdh_labels>0]), len(pdh_labels)))
    return pdh_features, pdh_labels

def load_PDH_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset PDH =====")
    pdh_features = []
    pdh_feature_ont = []
    pdh_feature_offt = []
    pdh_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/PDH/PDH.csv"
    pdh_dataset = pd.read_csv(data_path)
    # print(pdh_dataset)
    ## shuffle
    pdh_dataset = shuffle(pdh_dataset, random_state=42)
    ## encode
    for i, row in pdh_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['readFraction']
        pdh_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        pdh_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        pdh_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        pdh_labels.append(label)
    pdh_features = np.array(pdh_features)
    pdh_feature_ont = np.array(pdh_feature_ont)
    pdh_feature_offt = np.array(pdh_feature_offt)
    if out_dim == 2:
        pdh_labels = to_categorical(pdh_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    pdh_labels = np.array(pdh_labels)
    print("[INFO] Encoded dataset PDH features with size of", pdh_features.shape)
    print("[INFO] Encoded dataset PDH feature ont with size of", pdh_feature_ont.shape)
    print("[INFO] Encoded dataset PDH feature offt with size of", pdh_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset PDH is {0}, the active+inactive is {1}.".format(len(pdh_labels[pdh_labels>0]), len(pdh_labels)))
    return pdh_features, pdh_feature_ont, pdh_feature_offt, pdh_labels

def load_SITE(encoding_method, out_dim=1, is_binarized=True):
    print("[INFO] ===== Start Loading dataset SITE =====")
    site_features = []
    site_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/SITE/SITE.csv"
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
        if is_binarized and label > 0:
            label = 1.0
        site_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        site_labels.append(label)
    site_features = np.array(site_features)
    if out_dim == 2:
        site_labels = to_categorical(site_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    if is_binarized == False:
        site_labels = np.reshape(site_labels, (len(site_labels), 1))
        site_labels = MinMaxScaler().fit_transform(site_labels)
        site_labels = np.reshape(site_labels, (len(site_labels)))
    site_labels = np.array(site_labels)
    print("[INFO] Encoded dataset site features with size of", site_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset site is {0}, the active+inactive is {1}.".format(len(site_labels[site_labels>0]), len(site_labels)))
    return site_features, site_labels

def load_SITE_encoded_by_both_base_and_base_pair(out_dim=1, is_binarized=True):
    print("[INFO] ===== Start Loading dataset SITE =====")
    site_features = []
    site_feature_ont = []
    site_feature_offt = []
    site_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/SITE/SITE.csv"
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
        if is_binarized and label > 0:
            label = 1.0
        site_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        site_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        site_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        site_labels.append(label)
    site_features = np.array(site_features)
    site_feature_ont = np.array(site_feature_ont)
    site_feature_offt = np.array(site_feature_offt)
    if out_dim == 2:
        site_labels = to_categorical(site_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    if is_binarized == False:
        site_labels = np.reshape(site_labels, (len(site_labels), 1))
        site_labels = MinMaxScaler().fit_transform(site_labels)
        site_labels = np.reshape(site_labels, (len(site_labels)))
    site_labels = np.array(site_labels)
    print("[INFO] Encoded dataset site features with size of", site_features.shape)
    print("[INFO] Encoded dataset site feature ont with size of", site_feature_ont.shape)
    print("[INFO] Encoded dataset site feature offt with size of", site_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset site is {0}, the active+inactive is {1}.".format(len(site_labels[site_labels>0]), len(site_labels)))
    return site_features, site_feature_ont, site_feature_offt, site_labels

def load_GUIDE_I(encoding_method, out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_I =====")
    guide_i_features = []
    guide_i_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_I/GUIDE_I.csv"
    guide_i_dataset = pd.read_csv(data_path)
    # print(guide_i_dataset)
    ## shuffle
    guide_i_dataset = shuffle(guide_i_dataset, random_state=42)
    ## encode
    for i, row in guide_i_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['GUIDE-SEQ Reads']
        guide_i_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_i_labels.append(label)
    guide_i_features = np.array(guide_i_features)
    if out_dim == 2:
        guide_i_labels = to_categorical(guide_i_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_i_labels = np.array(guide_i_labels)
    print("[INFO] Encoded dataset guide_i features with size of", guide_i_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_i is {0}, the active+inactive is {1}.".format(len(guide_i_labels[guide_i_labels>0]), len(guide_i_labels)))
    return guide_i_features, guide_i_labels

def load_GUIDE_I_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_I =====")
    guide_i_features = []
    guide_i_feature_ont = []
    guide_i_feature_offt = []
    guide_i_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_I/GUIDE_I.csv"
    guide_i_dataset = pd.read_csv(data_path)
    # print(guide_i_dataset)
    ## shuffle
    guide_i_dataset = shuffle(guide_i_dataset, random_state=42)
    ## encode
    for i, row in guide_i_dataset.iterrows():
        on_target_seq = row['30mer']
        off_target_seq = row['30mer_mut']
        label = row['GUIDE-SEQ Reads']
        guide_i_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_i_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        guide_i_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        guide_i_labels.append(label)
    guide_i_features = np.array(guide_i_features)
    guide_i_feature_ont = np.array(guide_i_feature_ont)
    guide_i_feature_offt = np.array(guide_i_feature_offt)
    if out_dim == 2:
        guide_i_labels = to_categorical(guide_i_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_i_labels = np.array(guide_i_labels)
    print("[INFO] Encoded dataset guide_i features with size of", guide_i_features.shape)
    print("[INFO] Encoded dataset guide_i feature ont with size of", guide_i_feature_ont.shape)
    print("[INFO] Encoded dataset guide_i feature offt with size of", guide_i_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_i is {0}, the active+inactive is {1}.".format(len(guide_i_labels[guide_i_labels>0]), len(guide_i_labels)))
    return guide_i_features, guide_i_feature_ont, guide_i_feature_offt, guide_i_labels

def load_GUIDE_II(encoding_method, out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_II =====")
    guide_ii_features = []
    guide_ii_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_II/GUIDE_II.csv"
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
        guide_ii_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_ii_labels.append(label)
    guide_ii_features = np.array(guide_ii_features)
    if out_dim == 2:
        guide_ii_labels = to_categorical(guide_ii_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_ii_labels = np.array(guide_ii_labels)
    print("[INFO] Encoded dataset guide_ii features with size of", guide_ii_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_ii is {0}, the active+inactive is {1}.".format(len(guide_ii_labels[guide_ii_labels>0]), len(guide_ii_labels)))
    return guide_ii_features, guide_ii_labels

def load_GUIDE_II_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_II =====")
    guide_ii_features = []
    guide_ii_feature_ont = []
    guide_ii_feature_offt = []
    guide_ii_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_II/GUIDE_II.csv"
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
        guide_ii_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_ii_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        guide_ii_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        guide_ii_labels.append(label)
    guide_ii_features = np.array(guide_ii_features)
    guide_ii_feature_ont = np.array(guide_ii_feature_ont)
    guide_ii_feature_offt = np.array(guide_ii_feature_offt)
    if out_dim == 2:
        guide_ii_labels = to_categorical(guide_ii_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_ii_labels = np.array(guide_ii_labels)
    print("[INFO] Encoded dataset guide_ii features with size of", guide_ii_features.shape)
    print("[INFO] Encoded dataset guide_ii feature ont with size of", guide_ii_feature_ont.shape)
    print("[INFO] Encoded dataset guide_ii feature offt with size of", guide_ii_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_ii is {0}, the active+inactive is {1}.".format(len(guide_ii_labels[guide_ii_labels>0]), len(guide_ii_labels)))
    return guide_ii_features, guide_ii_feature_ont, guide_ii_feature_offt, guide_ii_labels

def load_GUIDE_III(encoding_method, out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_III =====")
    guide_iii_features = []
    guide_iii_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_III/GUIDE_III.csv"
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
        guide_iii_features.append(encoding_method(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_iii_labels.append(label)
    guide_iii_features = np.array(guide_iii_features)
    if out_dim == 2:
        guide_iii_labels = to_categorical(guide_iii_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_iii_labels = np.array(guide_iii_labels)
    print("[INFO] Encoded dataset guide_iii features with size of", guide_iii_features.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_iii is {0}, the active+inactive is {1}.".format(len(guide_iii_labels[guide_iii_labels>0]), len(guide_iii_labels)))
    return guide_iii_features, guide_iii_labels

def load_GUIDE_III_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset GUIDE_III =====")
    guide_iii_features = []
    guide_iii_feature_ont = []
    guide_iii_feature_offt = []
    guide_iii_labels = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/GUIDE_III/GUIDE_III.csv"
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
        guide_iii_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        guide_iii_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        guide_iii_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        guide_iii_labels.append(label)
    guide_iii_features = np.array(guide_iii_features)
    guide_iii_feature_ont = np.array(guide_iii_feature_ont)
    guide_iii_feature_offt = np.array(guide_iii_feature_offt)
    if out_dim == 2:
        guide_iii_labels = to_categorical(guide_iii_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    guide_iii_labels = np.array(guide_iii_labels)
    print("[INFO] Encoded dataset guide_iii features with size of", guide_iii_features.shape)
    print("[INFO] Encoded dataset guide_iii feature ont with size of", guide_iii_feature_ont.shape)
    print("[INFO] Encoded dataset guide_iii feature offt with size of", guide_iii_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset guide_iii is {0}, the active+inactive is {1}.".format(len(guide_iii_labels[guide_iii_labels>0]), len(guide_iii_labels)))
    return guide_iii_features, guide_iii_feature_ont, guide_iii_feature_offt, guide_iii_labels

def get_epigenetic_code(epigenetic_1, epigenetic_2, epigenetic_3, epigenetic_4):
    epimap = {'A': 1, 'N': 0}
    tlen = 24
    epigenetic_1 = epigenetic_1.upper()
    epigenetic_1 = "N"*(tlen-len(epigenetic_1)) + epigenetic_1
    epigenetic_2 = epigenetic_2.upper()
    epigenetic_2 = "N"*(tlen-len(epigenetic_2)) + epigenetic_2
    epigenetic_3 = epigenetic_3.upper()
    epigenetic_3 = "N"*(tlen-len(epigenetic_3)) + epigenetic_3
    epigenetic_4 = epigenetic_4.upper()
    epigenetic_4 = "N"*(tlen-len(epigenetic_4)) + epigenetic_4
    epi_code = list()
    for i in range(len(epigenetic_1)):
        t = [epimap[epigenetic_1[i]], epimap[epigenetic_2[i]], epimap[epigenetic_3[i]], epimap[epigenetic_4[i]]]
        epi_code.append(t)
    return epi_code

def load_K562_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset K562 =====")
    k562_features = []
    k562_feature_ont = []
    k562_feature_offt = []
    k562_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/k562.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    # on_seqs = _ori_df[1].tolist()
    # off_seqs = _ori_df[6].tolist()
    # labels = _ori_df[11].tolist()
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        k562_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        k562_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        k562_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        k562_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
    k562_features = np.array(k562_features)
    k562_feature_ont = np.array(k562_feature_ont)
    k562_feature_offt = np.array(k562_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    if out_dim == 2:
        k562_labels = to_categorical(k562_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    k562_labels = np.array(k562_labels)
    print("[INFO] Encoded dataset K562 features with size of", k562_features.shape)
    print("[INFO] Encoded dataset K562 feature ont with size of", k562_feature_ont.shape)
    print("[INFO] Encoded dataset K562 feature offt with size of", k562_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset K562 is {0}, the active+inactive is {1}.".format(len(k562_labels[k562_labels>0]), len(k562_labels)))
    print("[INFO] Encoded dataset K562 on_epigenetic_code with size of", on_epigenetic_code.shape)
    print("[INFO] Encoded dataset K562 off_epigenetic_code with size of", off_epigenetic_code.shape)
    return k562_features, k562_feature_ont, k562_feature_offt, k562_labels, on_epigenetic_code, off_epigenetic_code

def load_HEK293t_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset HEK293t =====")
    hek293t_features = []
    hek293t_feature_ont = []
    hek293t_feature_offt = []
    hek293t_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/hek293t.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        hek293t_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        hek293t_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        hek293t_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        hek293t_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
    hek293t_features = np.array(hek293t_features)
    hek293t_feature_ont = np.array(hek293t_feature_ont)
    hek293t_feature_offt = np.array(hek293t_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    if out_dim == 2:
        hek293t_labels = to_categorical(hek293t_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    hek293t_labels = np.array(hek293t_labels)
    print("[INFO] Encoded dataset hek293t features with size of", hek293t_features.shape)
    print("[INFO] Encoded dataset hek293t feature ont with size of", hek293t_feature_ont.shape)
    print("[INFO] Encoded dataset hek293t feature offt with size of", hek293t_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset hek293t is {0}, the active+inactive is {1}.".format(len(hek293t_labels[hek293t_labels>0]), len(hek293t_labels)))
    print("[INFO] Encoded dataset HEK293t on_epigenetic_code with size of", on_epigenetic_code.shape)
    print("[INFO] Encoded dataset HEK293t off_epigenetic_code with size of", off_epigenetic_code.shape)
    return hek293t_features, hek293t_feature_ont, hek293t_feature_offt, hek293t_labels, on_epigenetic_code, off_epigenetic_code


if __name__=="__main__":
    features, f2, f3, labels, on_epigenetic_code, off_epigenetic_code=load_HEK293t_encoded_by_both_base_and_base_pair()
    print(labels.shape)
    print(max(labels))
    print(min(labels))
    is_reg = False
    for i in labels:
        if 0<i<1:
            is_reg = True
            break
    print(is_reg)


