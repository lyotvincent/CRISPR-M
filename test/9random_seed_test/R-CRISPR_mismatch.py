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
from encoding import encode_by_r_crispr_method
from data_preprocessing_utils import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III
from metrics_utils import compute_auroc_and_auprc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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
    inputs = Input(shape=(24, 7), name='main_input')
    inputsr = Reshape(tuple([1, 24, 7]))(inputs)
    inputs1=Conv2D(15, (1,1), strides=1, padding='same')(inputsr)
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

if __name__ == "__main__":
    time1 = time.time()

    SEED = int(sys.argv[1])
    print(f"SEED={SEED}")
    random.seed(SEED)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(SEED)  # numpy的随机性
    tf.random.set_seed(SEED)  # tensorflow的随机性

    # classification 0; regression 1;
    cla_or_reg = 0

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
        result = classification_data_method[i](encoding_method=encode_by_r_crispr_method)
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
    print("=====")
    print("average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f = open("r-crispr_mismatch_test_result.txt", "a")
    f.write("seed=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s\n" % (SEED, auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f.close()

    print(time.time()-time1)

# seed=0, average_auroc_by_auc=0.8354788365637615, average_auprc_by_auc=0.31549263036793995
# seed=10, average_auroc_by_auc=0.839608444097159, average_auprc_by_auc=0.3740745878464649
# seed=20, average_auroc_by_auc=0.8522021023588174, average_auprc_by_auc=0.38355483598491064
# seed=30, average_auroc_by_auc=0.8679391352428414, average_auprc_by_auc=0.3852220325160804
# seed=40, average_auroc_by_auc=0.8159270019579261, average_auprc_by_auc=0.4026896528072899
# seed=50, average_auroc_by_auc=0.8703957061285875, average_auprc_by_auc=0.38920067492374516
# seed=60, average_auroc_by_auc=0.8529087516442029, average_auprc_by_auc=0.37565033992352803
# seed=70, average_auroc_by_auc=0.8567327395031317, average_auprc_by_auc=0.40218171431890226
# seed=80, average_auroc_by_auc=0.844617644711082, average_auprc_by_auc=0.37475786726742183
# seed=90, average_auroc_by_auc=0.882421775595485, average_auprc_by_auc=0.4112804934635258
# seed=100, average_auroc_by_auc=0.8615876117715624, average_auprc_by_auc=0.36461005809520913
# seed=110, average_auroc_by_auc=0.8658652945644804, average_auprc_by_auc=0.3473156201231854
# seed=120, average_auroc_by_auc=0.8461211114608564, average_auprc_by_auc=0.3779097514358848
# seed=130, average_auroc_by_auc=0.8565012678650249, average_auprc_by_auc=0.40329839392803185
# seed=140, average_auroc_by_auc=0.8598800711155403, average_auprc_by_auc=0.4241532502428708
# seed=150, average_auroc_by_auc=0.8342946667263729, average_auprc_by_auc=0.3603368687289499
# seed=160, average_auroc_by_auc=0.8550285074265085, average_auprc_by_auc=0.38725636841872296
# seed=170, average_auroc_by_auc=0.8296485805440056, average_auprc_by_auc=0.39219650172134357
# seed=180, average_auroc_by_auc=0.8504572024880881, average_auprc_by_auc=0.38783464661794737
# seed=190, average_auroc_by_auc=0.849683578921324, average_auprc_by_auc=0.41230092633868776

