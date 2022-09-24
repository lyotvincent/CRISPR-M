import os, time, sys, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention, Dense, Conv2D, Conv1D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.metrics import AUC, Precision, Recall

sys.path.append("../../../codes")
from data_preprocessing_utils import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III
from metrics_utils import compute_auroc_and_auprc
from encoding import encode_by_crispr_ip_method

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def crispr_ip(xtrain, ytrain, xtest, ytest):
    print("xtrain = %s"%str(xtrain.shape))
    print("ytrain = %s"%str(ytrain.shape))
    print("xtest = %s"%str(xtest.shape))
    print("ytest = %s"%str(ytest.shape))

    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    inputs = Input(shape=(24, 7,))
    conv_1_output_reshape = Reshape(tuple([1, 24, 7]))(inputs)
    conv_1_output = Conv2D(60, (1,7), padding='valid', data_format='channels_first', kernel_initializer=initializer)(conv_1_output_reshape)
    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
    attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    flatten_output = Flatten()(concat_output)
    linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
    linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
    linear_2_output_dropout = Dropout(0.9)(linear_2_output)
    outputs = Dense(num_classes, activation='sigmoid', kernel_initializer=initializer)(linear_2_output_dropout)
        
    model = Model(inputs, outputs)
    model.summary()
    model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc", label_weights=[0, 1]), AUC(num_thresholds=4000, curve="PR", name="auprc", label_weights=[0, 1])])
    
    epochs = 500
    batch_size = 4000
    eary_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.0001,
        patience=10, verbose=1, mode='auto')
    callbacks = [eary_stopping]
    history_model = model.fit(
        xtrain, ytrain,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(xtest, ytest),
        callbacks=callbacks
    )
    model.save('tcrispr_model.h5')
    model = load_model('tcrispr_model.h5')
    print("shape = %s, %s"%(str(xtest.shape), str(ytest.shape)))
    test_loss, test_acc, auroc, auprc = model.evaluate(xtest, ytest)
    accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=num_classes, test_features=xtest, test_labels=ytest)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point


if __name__ == "__main__":
    time1 = time.time()
    num_classes = 2
    # batch_size = 512
    retrain=True
    encoder_shape=(24,7)
    seg_len, coding_dim = encoder_shape


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
        result = classification_data_method[i](encoding_method=encode_by_crispr_ip_method, out_dim=num_classes)
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
        print("[INFO] The labels number of active off-target sites in dataset ytrain is {0}, the active+inactive is {1}.".format(len(ytrain[ytrain[:, 1]>0]), len(ytrain)))
        xtest = np.array(xtest, dtype=np.float32)
        ytest = np.array(ytest, dtype=np.float32)
        print("[INFO] Encoded dataset xtest with size of", xtest.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytest is {0}, the active+inactive is {1}.".format(len(ytest[ytest[:, 1]>0]), len(ytest)))
        
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
            print("remove tcrispr_model.h5")
        print('Training!!')
        result = crispr_ip(xtrain, ytrain, xtest, ytest)
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
            print("remove tcrispr_model.h5")
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
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum/2, test_acc_sum/2, auroc_sum/2, auprc_sum/2, accuracy_sum/2, precision_sum/2, recall_sum/2, f1_sum/2, fbeta_sum/2, auroc_skl_sum/2, auprc_skl_sum/2, auroc_by_auc_sum/2, auprc_by_auc_sum/2, spearman_corr_by_pred_score_sum/2, spearman_corr_by_pred_labels_sum/2))
    print(time.time()-time1)
