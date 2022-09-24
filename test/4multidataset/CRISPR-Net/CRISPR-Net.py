import time, os, sys, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Attention, Dense, Conv2D, Conv1D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score
# import Encoder_sgRNA_off
# from test_model import *
sys.path.append("../../../codes")
from encoding import encode_by_crispr_net_method
from data_preprocessing_utils import load_CIRCLE_dataset, load_I_2_dataset, load_PKD, load_SITE, load_GUIDE_II, load_GUIDE_III
from metrics_utils import compute_auroc_and_auprc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name, trainable=trainable)(x)

    # x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x

def CRISPR_Net_model():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    branch_0 = conv2d_bn(inputs, 10, (1, 1))
    branch_1 = conv2d_bn(inputs, 10, (1, 2))
    branch_2 = conv2d_bn(inputs, 10, (1, 3))
    branch_3 = conv2d_bn(inputs, 10, (1, 5))
    branches = [inputs, branch_0, branch_1, branch_2, branch_3]
    # branches = [branch_0, branch_1, branch_2, branch_3]
    mixed = layers.Concatenate(axis=-1)(branches)
    mixed = Reshape((24, 47))(mixed)
    blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(24, 47), name="LSTM_out"))(mixed)
    # inputs_rs = Reshape((24, 7))(inputs)
    # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
    blstm_out = Flatten()(blstm_out)
    x = Dense(80, activation='relu')(blstm_out)
    x = Dense(20, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.35)(x)
    prediction = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs, prediction)
    print(model.summary())
    return model

def CRISPR_Net_training(X_train, y_train, X_val, y_val):
    # X_train, y_train = load_traininig_data()
    model = CRISPR_Net_model()
    adam_opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])
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

if __name__ == "__main__":
    time1 = time.time()
    train_features_1, train_labels_1 = load_CIRCLE_dataset(encoding_method=encode_by_crispr_net_method)
    train_features_2, train_labels_2 = load_SITE(encoding_method=encode_by_crispr_net_method)
    val_features_1, val_labels_1 = load_I_2_dataset(encoding_method=encode_by_crispr_net_method)
    val_features_2, val_labels_2 = load_PKD(encoding_method=encode_by_crispr_net_method)
    val_features_3, val_labels_3 = load_GUIDE_II(encoding_method=encode_by_crispr_net_method)
    val_features_4, val_labels_4 = load_GUIDE_III(encoding_method=encode_by_crispr_net_method)
    xtrain, ytrain = np.concatenate([train_features_1, train_features_2]), np.concatenate([train_labels_1, train_labels_2])
    xtest, ytest = np.concatenate([val_features_1, val_features_2, val_features_3, val_features_4]), np.concatenate([val_labels_1, val_labels_2, val_labels_3, val_labels_4])
    xtrain = xtrain.reshape(xtrain.shape[0], 1, 24, 7)
    xtest = xtest.reshape(xtest.shape[0], 1, 24, 7)

    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    print("xtrain = %s"%str(xtrain.shape))
    print("ytrain = %s"%str(ytrain.shape))
    print("xtest = %s"%str(xtest.shape))
    print("ytest = %s"%str(ytest.shape))
    print('Training!!')
    if os.path.exists("tcrispr_model.h5"):
        os.remove("tcrispr_model.h5")
    result = CRISPR_Net_training(xtrain, ytrain, xtest, ytest)
    if os.path.exists("tcrispr_model.h5"):
        os.remove("tcrispr_model.h5")
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
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum))
    print(time.time()-time1)
