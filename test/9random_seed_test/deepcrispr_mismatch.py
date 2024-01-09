import time, os, sys, pickle, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score

sys.path.append("../../codes")
from encoding import encode_by_crispr_net_method
from load_deepcrispr import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III
from metrics_utils import compute_auroc_and_auprc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def Net_model():
    input_1 = Input(shape=(23, 4), name='input_1')
    branch_1 = Reshape(tuple([1, 23, 4]))(input_1)
    branch_1 = Conv2D(32, kernel_size=(1, 3), padding='same')(branch_1)
    branch_1 = BatchNormalization(momentum=0, center=False)(branch_1)
    branch_1 = Conv2D(64, kernel_size=(1, 3), padding='same', strides=2)(branch_1)
    branch_1 = BatchNormalization(momentum=0, center=False)(branch_1)
    branch_1 = Conv2D(64, kernel_size=(1, 3), padding='same')(branch_1)
    branch_1 = BatchNormalization(momentum=0, center=False)(branch_1)
    branch_1 = Conv2D(256, kernel_size=(1, 3), padding='same', strides=2)(branch_1)
    branch_1 = BatchNormalization(momentum=0, center=False)(branch_1)
    branch_1 = Conv2D(256, kernel_size=(1, 3), padding='same')(branch_1)
    branch_1 = BatchNormalization(momentum=0, center=False)(branch_1)


    input_2 = Input(shape=(23, 4), name='input_2')
    branch_2 = Reshape(tuple([1, 23, 4]))(input_2)
    branch_2 = Conv2D(32, kernel_size=(1, 3), padding='same')(branch_2)
    branch_2 = BatchNormalization(momentum=0, center=False)(branch_2)
    branch_2 = Conv2D(64, kernel_size=(1, 3), padding='same', strides=2)(branch_2)
    branch_2 = BatchNormalization(momentum=0, center=False)(branch_2)
    branch_2 = Conv2D(64, kernel_size=(1, 3), padding='same')(branch_2)
    branch_2 = BatchNormalization(momentum=0, center=False)(branch_2)
    branch_2 = Conv2D(256, kernel_size=(1, 3), padding='same', strides=2)(branch_2)
    branch_2 = BatchNormalization(momentum=0, center=False)(branch_2)
    branch_2 = Conv2D(256, kernel_size=(1, 3), padding='same')(branch_2)
    branch_2 = BatchNormalization(momentum=0, center=False)(branch_2)

    mixed = layers.Concatenate(axis=-1)([branch_1, branch_2])
    mixed = Conv2D(512, kernel_size=(1, 3), padding='same', strides=2)(mixed)
    mixed = BatchNormalization()(mixed)
    mixed = Conv2D(512, kernel_size=(1, 3), padding='same')(mixed)
    mixed = BatchNormalization()(mixed)
    mixed = Conv2D(1024, kernel_size=(1, 3), padding='valid')(mixed)
    mixed = BatchNormalization()(mixed)
    mixed = Flatten()(mixed)
    outputs = Dense(1, activation='sigmoid', name='output')(mixed)

    model = Model(inputs=[input_1, input_2], outputs=outputs)
    print(model.summary())
    return model

def Net_training(xtrain_on, xtrain_off, y_train, X_val_on, X_val_off, y_val):
    # X_train, y_train = load_traininig_data()
    model = Net_model()
    adam_opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])
    
    eary_stopping = EarlyStopping(
                    monitor='loss', min_delta=0.0001,
                    patience=10, verbose=1, mode='auto')
    model_checkpoint = ModelCheckpoint( # 在每轮过后保存当前权重
                    filepath='tcrispr_model.h5', # 目标模型文件的保存路径
                    # 这两个参数的含义是，如果 val_loss 没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
                    monitor='val_auprc',
                    verbose=1,
                    save_best_only=True,
                    mode='max')
    reduce_lr_on_plateau = ReduceLROnPlateau( # 如果验证损失在 4 轮内都没有改善，那么就触发这个回调函数
                    monitor='val_loss', # 监控模型的验证损失
                    factor=0.8, # 触发时将学习率除以 2
                    patience=4,
                    verbose=1,
                    mode='min',
                    min_lr=1e-7)
    callbacks = [eary_stopping, model_checkpoint, reduce_lr_on_plateau]
    
    model.fit(  x={"input_1": xtrain_on, "input_2": xtrain_off},
                y={"output": y_train},
                batch_size=10000, epochs=100, shuffle=True,
                validation_data=({"input_1": X_val_on, "input_2": X_val_off}, {"output": y_val}),
                callbacks=callbacks
                )
    test_loss, test_acc, auroc, auprc = model.evaluate(x={"input_1": X_val_on, "input_2": X_val_off}, y=y_val)
    accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=1, test_features={"input_1": X_val_on, "input_2": X_val_off}, test_labels=y_val)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point


if __name__ == "__main__":
    time1 = time.time()
    SEED = int(sys.argv[1])
    print(f"SEED={SEED}")
    random.seed(SEED)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(SEED)  # numpy的随机性
    tf.random.set_seed(SEED)  # tensorflow的随机性

    six_dataset_fold = dict()
    six_dataset_fold["pkd"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["pdh"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["site"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_i"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_ii"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_iii"] = {"on_features": list(), "off_features": list(), "labels": list()}
    classification_data_abbr = ["site", "pkd",  "guide_ii", "guide_iii"]
    classification_data_method = [load_SITE, load_PKD, load_GUIDE_II, load_GUIDE_III]
    regression_data_abbr = ["pdh", "guide_i"]
    regression_data_method = [load_PDH, load_GUIDE_I]
    for i in range(4):
        result = classification_data_method[i]()
        six_dataset_fold[classification_data_abbr[i]]["on_features"] = result[0]
        six_dataset_fold[classification_data_abbr[i]]["off_features"] = result[1]
        six_dataset_fold[classification_data_abbr[i]]["labels"] = result[2]
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    for i in range(2):
        xtrain_on = list()
        xtrain_off = list()
        ytrain = list()
        xtest_on = list()
        xtest_off = list()
        ytest = list()
        for j in range(4):
            if bool(j) == bool(i):
                xtest_on.extend(six_dataset_fold[classification_data_abbr[j]]["on_features"])
                xtest_off.extend(six_dataset_fold[classification_data_abbr[j]]["off_features"])
                ytest.extend(six_dataset_fold[classification_data_abbr[j]]["labels"])
            else:
                xtrain_on.extend(six_dataset_fold[classification_data_abbr[j]]["on_features"])
                xtrain_off.extend(six_dataset_fold[classification_data_abbr[j]]["off_features"])
                ytrain.extend(six_dataset_fold[classification_data_abbr[j]]["labels"])
        xtrain_on = np.array(xtrain_on, dtype=np.float32)
        xtrain_off = np.array(xtrain_off, dtype=np.float32)
        ytrain = np.array(ytrain, dtype=np.float32)
        print("[INFO] Encoded dataset xtrain_on with size of", xtrain_on.shape)
        print("[INFO] Encoded dataset xtrain_off with size of", xtrain_off.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytrain is {0}, the active+inactive is {1}.".format(len(ytrain[ytrain>0]), len(ytrain)))
        xtest_on = np.array(xtest_on, dtype=np.float32)
        xtest_off = np.array(xtest_off, dtype=np.float32)
        ytest = np.array(ytest, dtype=np.float32)
        print("[INFO] Encoded dataset xtest_on with size of", xtest_on.shape)
        print("[INFO] Encoded dataset xtest_off with size of", xtest_off.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytest is {0}, the active+inactive is {1}.".format(len(ytest[ytest>0]), len(ytest)))
        
        print('Training!!')
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
        result = Net_training(xtrain_on, xtrain_off, ytrain, xtest_on, xtest_off, ytest)
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
    print('End of the training!!')
    print("=====")
    print("average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f = open("deepcrispr_mismatch_test_result.txt", "a")
    f.write("seed=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s\n" % (SEED, auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f.close()

    print(time.time()-time1)

# seed=0, average_auroc_by_auc=0.6460036578779829, average_auprc_by_auc=0.02278464463952097
# seed=10, average_auroc_by_auc=0.623318526633815, average_auprc_by_auc=0.02053606755389617
# seed=20, average_auroc_by_auc=0.6364126592591597, average_auprc_by_auc=0.02223590339204374
# seed=30, average_auroc_by_auc=0.6316444626331628, average_auprc_by_auc=0.022239506628774323
# seed=40, average_auroc_by_auc=0.6562756730077213, average_auprc_by_auc=0.0218338628236853
# seed=50, average_auroc_by_auc=0.6384638175905474, average_auprc_by_auc=0.022260434004486147
# seed=60, average_auroc_by_auc=0.6325915337495516, average_auprc_by_auc=0.022030600915125875
# seed=70, average_auroc_by_auc=0.633388978615643, average_auprc_by_auc=0.02109445019262771
# seed=80, average_auroc_by_auc=0.6335238565192266, average_auprc_by_auc=0.02070031563100381
# seed=90, average_auroc_by_auc=0.6318350735659828, average_auprc_by_auc=0.020982500606694574
# seed=100, average_auroc_by_auc=0.6353880788454376, average_auprc_by_auc=0.021286124195004977
# seed=110, average_auroc_by_auc=0.6240014905833672, average_auprc_by_auc=0.021002352966780057
# seed=120, average_auroc_by_auc=0.6323348978205705, average_auprc_by_auc=0.02119425094959664
# seed=130, average_auroc_by_auc=0.6532443116637939, average_auprc_by_auc=0.02445011821401616
# seed=140, average_auroc_by_auc=0.6362727868071887, average_auprc_by_auc=0.022924215957386456
# seed=150, average_auroc_by_auc=0.6292552615234447, average_auprc_by_auc=0.021480611415026097
# seed=160, average_auroc_by_auc=0.6276518901315911, average_auprc_by_auc=0.021107803597619483
# seed=170, average_auroc_by_auc=0.6377910093134467, average_auprc_by_auc=0.021808792856296673
# seed=180, average_auroc_by_auc=0.6226250135620489, average_auprc_by_auc=0.020845959074781168
# seed=190, average_auroc_by_auc=0.6407366141118693, average_auprc_by_auc=0.02291705545450847
