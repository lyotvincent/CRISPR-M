import time, os, sys, pickle, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score
from dataset_encoder import gRNA_pair_encoder

sys.path.append("../../codes")
from encoding import encode_by_r_crispr_method
import data_preprocessing_utils
from metrics_utils import compute_auroc_and_auprc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def load_ten_grna_fold_circle(ten_grna_fold_CIRLCE):
    print("[INFO] ===== Start Encoding 10-grna-fold dataset CIRCLE =====")
    SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
    circle_dataset = pd.read_csv(r"../../../datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv")
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
            pair_codes = encode_by_r_crispr_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
            ten_grna_fold_CIRLCE[i]["features"].append(pair_codes)
            ten_grna_fold_CIRLCE[i]["labels"].append(label)
        ten_grna_fold_CIRLCE[i]["features"] = np.array(ten_grna_fold_CIRLCE[i]["features"])
        ten_grna_fold_CIRLCE[i]["features"] = ten_grna_fold_CIRLCE[i]["features"].reshape(ten_grna_fold_CIRLCE[i]["features"].shape[0], 1, 24, 7)
        ten_grna_fold_CIRLCE[i]["labels"] = np.array(ten_grna_fold_CIRLCE[i]["labels"])
        print("[INFO] %s-th-grna-fold set, features shape=%s"%(i, str(ten_grna_fold_CIRLCE[i]["features"].shape)))
        print("[INFO] %s-th-grna-fold set, labels shape=%s, and positive samples number = %s"%(i, str(ten_grna_fold_CIRLCE[i]["labels"].shape), len(ten_grna_fold_CIRLCE[i]["labels"][ten_grna_fold_CIRLCE[i]["labels"]>0])))
    return ten_grna_fold_CIRLCE

def get_ten_grna_fold_train_val(sgRNA_type_index, ten_grna_fold_CIRLCE):
    SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
    train_features = list()
    train_labels = list()
    validation_features = list()
    validation_labels = list()
    for i in range(10):
        if i != sgRNA_type_index:
            print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[sgRNA_type_index]))
            train_features.extend(ten_grna_fold_CIRLCE[i]["features"])
            train_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
        else:
            print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[sgRNA_type_index]))
            validation_features.extend(ten_grna_fold_CIRLCE[i]["features"])
            validation_labels.extend(ten_grna_fold_CIRLCE[i]["labels"])
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    validation_features = np.array(validation_features)
    validation_labels = np.array(validation_labels)
    return train_features, train_labels, validation_features, validation_labels

def ConvBn(inputs,filters, kernel_size, strides=1, padding='same', groups=1):
    x = inputs
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,groups=groups, use_bias=False)(x)
    x = BatchNormalization()(x)
    return x

def RepVGGBlock(inputs, filters, kernel_size, strides=1, padding='same', groups=1, deploy=False):
    x = inputs
    in_channels = inputs.shape[-1]
    rbr_dense = ConvBn(inputs,filters, kernel_size, strides=1, padding='same', groups=1)
    rbr_1x1 = ConvBn(inputs,filters, kernel_size=(1,1), strides=1, padding='same', groups=1)

    if in_channels == filters and strides == 1 :
        rbr_identity = BatchNormalization()(x)
        id_out = rbr_identity
    else:
        id_out = 0

    if deploy:
        rbr_reparam = Conv2D(filters, kernel_size, strides,padding,groups, use_bias=True)(x)
        return tf.nn.relu(rbr_reparam)

    x = tf.nn.relu(rbr_dense + rbr_1x1 + id_out)
    return x

def R_CRISPR_model():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    inputs1=Conv2D(15, (1,1), strides=1, padding='same')(inputs)
    repvgg1 = RepVGGBlock(inputs1, filters=15, kernel_size=(1, 3))

    mixed = Reshape((24, 15))(repvgg1)

    blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(24, 15), name="LSTM_out"))(mixed)
    blstm_out = Flatten()(blstm_out)
    x = Dense(80, activation='relu')(blstm_out)
    x = Dense(20, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)

    prediction = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs, prediction)
    print(model.summary())
    # 输出各层模型参数情况
    return model

def CRISPR_Net_training(X_train, y_train, X_val, y_val):
    # X_train, y_train = load_traininig_data()
    model = R_CRISPR_model()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])
    model.fit(X_train, y_train, batch_size=10000, epochs=100, shuffle=True, validation_data=(X_val, y_val))
    test_loss, test_acc, auroc, auprc = model.evaluate(xtest, ytest)
    accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=1, test_features=xtest, test_labels=ytest)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point
    # SAVE Model
    # model_jason = model.to_json()
    # model_path = "./saved_models"
    # if os.path.isdir(model_path):
    #     pass
    # else:
    #     os.mkdir(model_path)
    # with open(model_path + "/CRISPR_Net_structure_0.json", "w") as jason_file:
    #     jason_file.write(model_jason)
    # model.save_weights(model_path + "/CRISPR_Net_weights_0.h5")
    # print("Saved model to disk!")

# def compute_auroc_and_auprc(model, validation_features, validation_labels):
#     y_pred = model.predict(validation_features).ravel()
#     pred_labels = list()
#     for i in y_pred:
#         if i >= 0.5:
#             pred_labels.append(1.0)
#         else:
#             pred_labels.append(0.0)
#     pred_labels = np.array(pred_labels)
#     pred_score = y_pred
#     validation_labels = validation_labels

#     accuracy = accuracy_score(validation_labels, pred_labels)
#     precision = precision_score(validation_labels, pred_labels)
#     recall = recall_score(validation_labels, pred_labels)
#     f1 = f1_score(validation_labels, pred_labels)

#     auroc = roc_auc_score(validation_labels, pred_score)
#     fpr, tpr, thresholds = roc_curve(validation_labels, pred_score)
#     auroc_by_auc = auc(fpr, tpr)

#     auprc = average_precision_score(validation_labels, pred_score)
#     precision_point, recall_point, thresholds = precision_recall_curve(validation_labels, pred_score)
#     precision_point[(recall_point==0)] = 1.0
#     auprc_by_auc = auc(recall_point, precision_point)

#     print("accuracy=%s, precision=%s, recall=%s, f1=%s, auroc=%s, auprc=%s, auroc_by_auc=%s, auprc_by_auc=%s"%(accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc))
#     return accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc

if __name__ == "__main__":
    time1 = time.time()
    # ten_grna_fold_CIRLCE = dict()
    # for i in range(10):
    #     ten_grna_fold_CIRLCE[i] = {"features":list(), "labels":list()}
    # ten_grna_fold_CIRLCE = load_ten_grna_fold_circle(ten_grna_fold_CIRLCE)
    # test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    # for i in range(10):
    #     xtrain, ytrain, xtest, ytest = get_ten_grna_fold_train_val(sgRNA_type_index=i, ten_grna_fold_CIRLCE=ten_grna_fold_CIRLCE)
    #     print("xtrain = %s"%str(xtrain.shape))
    #     print("ytrain = %s"%str(ytrain.shape))
    #     print("xtest = %s"%str(xtest.shape))
    #     print("ytest = %s"%str(ytest.shape))
    #     print('Training!!')
    #     result = CRISPR_Net_training(xtrain, ytrain, xtest, ytest)
    #     test_loss_sum += result[0]
    #     test_acc_sum += result[1]
    #     auroc_sum += result[2]
    #     auprc_sum += result[3]
    #     accuracy_sum += result[4]
    #     precision_sum += result[5]
    #     recall_sum += result[6]
    #     f1_sum += result[7]
    #     fbeta_sum += result[8]
    #     auroc_skl_sum += result[9]
    #     auprc_skl_sum += result[10]
    #     auroc_by_auc_sum += result[11]
    #     auprc_by_auc_sum += result[12]
    #     spearman_corr_by_pred_score_sum += result[13]
    #     spearman_corr_by_pred_labels_sum += result[14]
    #     fpr_list.append(result[15])
    #     tpr_list.append(result[16])
    #     precision_point_list.append(result[17])
    #     recall_point_list.append(result[18])
    # with open("fpr.csv", "wb") as f:
    #     pickle.dump(fpr_list, f)
    # with open("tpr.csv", "wb") as f:
    #     pickle.dump(tpr_list, f)
    # with open("precision_point.csv", "wb") as f:
    #     pickle.dump(precision_point_list, f)
    # with open("recall_point.csv", "wb") as f:
    #     pickle.dump(recall_point_list, f)
    # print("=====")
    # print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum/10, test_acc_sum/10, auroc_sum/10, auprc_sum/10, accuracy_sum/10, precision_sum/10, recall_sum/10, f1_sum/10, fbeta_sum/10, auroc_skl_sum/10, auprc_skl_sum/10, auroc_by_auc_sum/10, auprc_by_auc_sum/10, spearman_corr_by_pred_score_sum/10, spearman_corr_by_pred_labels_sum/10))
    

    SEED = int(sys.argv[1])
    print(f"SEED={SEED}")
    random.seed(SEED)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(SEED)  # numpy的随机性
    tf.random.set_seed(SEED)  # tensorflow的随机性

    encoding_method = encode_by_r_crispr_method
    xtrain, ytrain = data_preprocessing_utils.load_CIRCLE_dataset(encoding_method=encoding_method)
    xtrain = xtrain.reshape(xtrain.shape[0], 1, 24, 7)
    xtest, ytest = data_preprocessing_utils.load_I_2_dataset(encoding_method=encoding_method)
    xtest = xtest.reshape(xtest.shape[0], 1, 24, 7)
    print("xtrain = %s"%str(xtrain.shape))
    print("ytrain = %s"%str(ytrain.shape))
    print("xtest = %s"%str(xtest.shape))
    print("ytest = %s"%str(ytest.shape))
    print('Training!!')
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    result = CRISPR_Net_training(xtrain, ytrain, xtest, ytest)
    test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12], result[13], result[14]
    # print("i=%s, test_loss=%s, test_acc=%s, auroc=%s, auprc=%s, accuracy=%s, precision=%s, recall=%s, f1=%s, fbeta=%s, auroc_skl=%s, auprc_skl=%s, auroc_by_auc=%s, auprc_by_auc=%s, spearman_corr_by_pred_score=%s, spearman_corr_by_pred_labels=%s" % (i, test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels))
    test_loss_sum += result[0]
    test_acc_sum += result[1]
    auroc_sum += result[2]
    auprc_sum += result[3]
    accuracy_sum += result[4]
    precision_sum += result[5]
    recall_sum += result[6]
    f1_sum += result[7]
    fbeta_sum += result[8]
    auroc_skl_sum += result[9]
    auprc_skl_sum += result[10]
    auroc_by_auc_sum += result[11]
    auprc_by_auc_sum += result[12]
    spearman_corr_by_pred_score_sum += result[13]
    spearman_corr_by_pred_labels_sum += result[14]
    print("=====")
    print("average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (auroc_by_auc_sum, auprc_by_auc_sum))
    f = open("r_crispr_result.txt", "a")
    f.write("seed=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s\n" % (SEED, auroc_by_auc_sum, auprc_by_auc_sum))
    f.close()
    
    
    print(time.time()-time1)

# seed=0, average_auroc_by_auc=0.97221102191385, average_auprc_by_auc=0.33117429718016783
# seed=10, average_auroc_by_auc=0.980920175984066, average_auprc_by_auc=0.21455258346940903
# seed=20, average_auroc_by_auc=0.9657757278512084, average_auprc_by_auc=0.09025204757664579
# seed=30, average_auroc_by_auc=0.9665844877807025, average_auprc_by_auc=0.11888460283356583
# seed=40, average_auroc_by_auc=0.9639805407629405, average_auprc_by_auc=0.21874036659210283
# seed=50, average_auroc_by_auc=0.9711412781754512, average_auprc_by_auc=0.4047313131710089
# seed=60, average_auroc_by_auc=0.9725495247401617, average_auprc_by_auc=0.07244575089186128
# seed=70, average_auroc_by_auc=0.9766286708153523, average_auprc_by_auc=0.1393304993646361
# seed=80, average_auroc_by_auc=0.9736166969791895, average_auprc_by_auc=0.08701047244856952
# seed=90, average_auroc_by_auc=0.9735052809246176, average_auprc_by_auc=0.1654464893447289
# seed=100, average_auroc_by_auc=0.976509820789871, average_auprc_by_auc=0.1645871244993349
# seed=110, average_auroc_by_auc=0.9799958388464721, average_auprc_by_auc=0.19117429083229545
# seed=120, average_auroc_by_auc=0.9550059144485536, average_auprc_by_auc=0.21847257732623573
# seed=130, average_auroc_by_auc=0.9762614607051519, average_auprc_by_auc=0.4026079222925245
# seed=140, average_auroc_by_auc=0.9706791563611881, average_auprc_by_auc=0.11986307012944003
# seed=150, average_auroc_by_auc=0.9709609926922663, average_auprc_by_auc=0.20945326246708595
# seed=160, average_auroc_by_auc=0.974513121659973, average_auprc_by_auc=0.3397323339236034
# seed=170, average_auroc_by_auc=0.9671346016279928, average_auprc_by_auc=0.2931391317178899
# seed=180, average_auroc_by_auc=0.9726319062291064, average_auprc_by_auc=0.13884357578672726
# seed=190, average_auroc_by_auc=0.9725935675112094, average_auprc_by_auc=0.2772898430955618

