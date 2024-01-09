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

sys.path.append("../../codes")
from encoding import encode_by_crispr_net_method
from data_preprocessing_utils import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III
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
    inputs = Input(shape=(24, 7), name='main_input')
    inputsr = Reshape(tuple([1, 24, 7]))(inputs)
    branch_0 = conv2d_bn(inputsr, 10, (1, 1))
    branch_1 = conv2d_bn(inputsr, 10, (1, 2))
    branch_2 = conv2d_bn(inputsr, 10, (1, 3))
    branch_3 = conv2d_bn(inputsr, 10, (1, 5))
    branches = [inputsr, branch_0, branch_1, branch_2, branch_3]
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

    SEED = int(sys.argv[1])
    print(f"SEED={SEED}")
    random.seed(SEED)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(SEED)  # numpy的随机性
    tf.random.set_seed(SEED)  # tensorflow的随机性

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
        result = classification_data_method[i](encoding_method=encode_by_crispr_net_method)
        six_dataset_fold[classification_data_abbr[i]]["features"] = result[0]
        six_dataset_fold[classification_data_abbr[i]]["labels"] = result[1]
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
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
        print("[INFO] The labels number of active off-target sites in dataset ytrain is {0}, the active+inactive is {1}.".format(len(ytrain[ytrain>0]), len(ytrain)))
        xtest = np.array(xtest, dtype=np.float32)
        ytest = np.array(ytest, dtype=np.float32)
        print("[INFO] Encoded dataset xtest with size of", xtest.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytest is {0}, the active+inactive is {1}.".format(len(ytest[ytest>0]), len(ytest)))
        
        print('Training!!')
        result = CRISPR_Net_training(xtrain, ytrain, xtest, ytest)
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
    print('End of the training!!')
    print("=====")
    print("average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f = open("crispr-net_mismatch_test_result.txt", "a")
    f.write("seed=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s\n" % (SEED, auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f.close()

    print(time.time()-time1)

# seed=0, average_auroc_by_auc=0.8460876035597928, average_auprc_by_auc=0.42640740092218754
# seed=10, average_auroc_by_auc=0.8286509752100124, average_auprc_by_auc=0.3190569467826963
# seed=20, average_auroc_by_auc=0.8419365274407733, average_auprc_by_auc=0.3880444203473553
# seed=30, average_auroc_by_auc=0.8512711098292617, average_auprc_by_auc=0.4311005117880356
# seed=40, average_auroc_by_auc=0.8318251421181737, average_auprc_by_auc=0.3790875605479453
# seed=50, average_auroc_by_auc=0.8321937661464689, average_auprc_by_auc=0.3728412340785865
# seed=60, average_auroc_by_auc=0.855460191349436, average_auprc_by_auc=0.3345446692562608
# seed=70, average_auroc_by_auc=0.857978271908665, average_auprc_by_auc=0.4332806062365051
# seed=80, average_auroc_by_auc=0.8465863236897857, average_auprc_by_auc=0.3226598445346026
# seed=90, average_auroc_by_auc=0.8564264628443381, average_auprc_by_auc=0.36481551294018366
# seed=100, average_auroc_by_auc=0.8598879510032731, average_auprc_by_auc=0.38725695900586593
# seed=110, average_auroc_by_auc=0.8509337481412804, average_auprc_by_auc=0.2959588971748146
# seed=120, average_auroc_by_auc=0.8400945648103755, average_auprc_by_auc=0.41535143500516863
# seed=130, average_auroc_by_auc=0.8413731872731927, average_auprc_by_auc=0.4275492133224448
# seed=140, average_auroc_by_auc=0.8362754615323978, average_auprc_by_auc=0.34794067405892853
# seed=150, average_auroc_by_auc=0.8331023322443156, average_auprc_by_auc=0.3788237694732964
# seed=160, average_auroc_by_auc=0.8184445956839823, average_auprc_by_auc=0.3982675031554134
# seed=170, average_auroc_by_auc=0.8494623849235483, average_auprc_by_auc=0.2634639111861432
# seed=180, average_auroc_by_auc=0.852349587003909, average_auprc_by_auc=0.3635638949936475
# seed=190, average_auroc_by_auc=0.8133318396544851, average_auprc_by_auc=0.3268301205708606
