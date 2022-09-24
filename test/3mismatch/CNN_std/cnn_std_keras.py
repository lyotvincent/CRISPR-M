# -*- coding: utf-8 -*-
# @Time     :7/17/18 4:01 PM
# @Auther   :Jason Lin
# @File     :cnn_for_cd33$.py
# @Software :PyCharm

import pickle
import numpy as np
# import matplotlib.pyplot as plt
from scipy import interp
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Model
from keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam
import time, sys, tensorflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from tensorflow.keras.metrics import AUC

sys.path.append("../../../codes")
from load_4_cnn_std import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III
from metrics_utils import compute_auroc_and_auprc

# np.random.seed(5)
# from tensorflow import set_random_seed
# set_random_seed(12)


def cnn_model(X_train, y_train, X_test, y_test):

    # X_train, y_train = load_data()

    inputs = Input(shape=(1, 23, 4), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs)

    conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.15)(x)

    prediction = Dense(2, activation="softmax", name='main_output')(x)

    model = Model(inputs, prediction)

    adam_opt = Adam(lr = 0.0001)

    model.compile(loss='binary_crossentropy', optimizer = adam_opt, metrics=['acc', AUC(num_thresholds=4000, curve="ROC", name="auroc", label_weights=[0, 1]), AUC(num_thresholds=4000, curve="PR", name="auprc", label_weights=[0, 1])])
    print(model.summary())
    model.fit(X_train, y_train, batch_size=100, epochs=200, shuffle=True)
    test_loss, test_acc, auroc, auprc = model.evaluate(X_test, y_test)
    accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=2, test_features=X_test, test_labels=y_test)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point


if __name__ == "__main__":
    time1 = time.time()
    six_dataset_fold = dict()
    six_dataset_fold["pkd"] = {"features": list(), "labels": list()}
    six_dataset_fold["pdh"] = {"features": list(), "labels": list()}
    six_dataset_fold["site"] = {"features": list(), "labels": list()}
    six_dataset_fold["guide_i"] = {"features": list(), "labels": list()}
    six_dataset_fold["guide_ii"] = {"features": list(), "labels": list()}
    six_dataset_fold["guide_iii"] = {"features": list(), "labels": list()}
    classification_data_abbr = ["site", "pkd",  "guide_ii", "guide_iii"]
    classification_data_method = [load_SITE, load_PKD, load_GUIDE_II, load_GUIDE_III]
    regression_data_abbr = ["pdh", "guide_i"]
    regression_data_method = [load_PDH, load_GUIDE_I]
    for i in range(4):
        result = classification_data_method[i](out_dim=2)
        six_dataset_fold[classification_data_abbr[i]]["features"] = result[0]
        six_dataset_fold[classification_data_abbr[i]]["labels"] = result[1]
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    for i in range(2):
        xtrain = list()
        ytrain = list()
        xtest = list()
        ytest = list()
        for j in range(4):
            if bool(j) == bool(i):
                xtest.extend(six_dataset_fold[classification_data_abbr[j]]["features"])
                ytest.extend(six_dataset_fold[classification_data_abbr[j]]["labels"])
            else:
                xtrain.extend(six_dataset_fold[classification_data_abbr[j]]["features"])
                ytrain.extend(six_dataset_fold[classification_data_abbr[j]]["labels"])
        xtrain = np.array(xtrain, dtype=np.float32)
        ytrain = np.array(ytrain, dtype=np.float32)
        print("[INFO] Encoded dataset xtrain with size of", xtrain.shape)
        print("[INFO] Encoded dataset ytrain with size of", ytrain.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytrain is {0}, the active+inactive is {1}.".format(len(ytrain[ytrain[:, 1]>0]), len(ytrain)))
        xtest = np.array(xtest, dtype=np.float32)
        ytest = np.array(ytest, dtype=np.float32)
        print("[INFO] Encoded dataset xtest with size of", xtest.shape)
        print("[INFO] Encoded dataset ytest with size of", ytest.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytest is {0}, the active+inactive is {1}.".format(len(ytest[ytest[:, 1]>0]), len(ytest)))
        
        print('Training!!')
        result = cnn_model(xtrain, ytrain, xtest, ytest)
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
        fpr_list.append(result[15])
        tpr_list.append(result[16])
        precision_point_list.append(result[17])
        recall_point_list.append(result[18])
    with open("fpr.csv", "wb") as f:
        pickle.dump(fpr_list, f)
    with open("tpr.csv", "wb") as f:
        pickle.dump(tpr_list, f)
    with open("precision_point.csv", "wb") as f:
        pickle.dump(precision_point_list, f)
    with open("recall_point.csv", "wb") as f:
        pickle.dump(recall_point_list, f)
    print('End of the training!!')
    print("=====")
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum/2, test_acc_sum/2, auroc_sum/2, auprc_sum/2, accuracy_sum/2, precision_sum/2, recall_sum/2, f1_sum/2, fbeta_sum/2, auroc_skl_sum/2, auprc_skl_sum/2, auroc_by_auc_sum/2, auprc_by_auc_sum/2, spearman_corr_by_pred_score_sum/2, spearman_corr_by_pred_labels_sum/2))

    print(time.time()-time1)
