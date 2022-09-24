import time, os
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
import Encoder_sgRNA_off
from test_model import *

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
            en = Encoder_sgRNA_off.Encoder(on_seq=gRNA_seq, off_seq=target_seq)
            ten_grna_fold_CIRLCE[i]["features"].append(en.on_off_code)
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
    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    inputs = Input(shape=(1, 24, 7,))
    conv_1_output = Conv2D(60, (1,7), padding='valid', data_format='channels_first', kernel_initializer=initializer)(inputs)
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
    outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(linear_2_output_dropout)

    #inputs = Input(shape=(24, 7,))
    #main = Flatten()(inputs)
    #main = Dense(100, activation='relu')(main)
    #main = Dense(100, activation='relu')(main)
    #main = Dense(100, activation='relu')(main)
    #outputs = Dense(1, activation='sigmoid')(main)

    #inputs = Input(shape=(24, 7,))
    #main = Conv1D(10, 3)(inputs)
    #main = Conv1D(10, 3)(main)
    #main = Conv1D(10, 3)(main)
    #main = Flatten()(main)
    #main = Dense(100, activation='relu')(main)
    #outputs = Dense(1, activation='sigmoid')(main)

    #inputs = Input(shape=(24, 7,))
    #main = GRU(30, return_sequences=True)(inputs)
    #main = Flatten()(main)
    #main = Dense(100)(main)
    #outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs, outputs)
    print(model.summary())
    return model

def CRISPR_Net_training(X_train, y_train, X_val, y_val):
    # X_train, y_train = load_traininig_data()
    model = CRISPR_Net_model()
    adam_opt = keras.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])
    eary_stopping = tf.keras.callbacks.EarlyStopping(
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
    model.fit(X_train, y_train, batch_size=1024, epochs=500, validation_data=(X_val, y_val), callbacks=callbacks)
    model = load_model('tcrispr_model.h5')
    test_loss, test_acc, auroc, auprc = model.evaluate(xtest, ytest)
    accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc = compute_auroc_and_auprc(model, xtest, ytest)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc
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

def compute_auroc_and_auprc(model, validation_features, validation_labels):
    y_pred = model.predict(validation_features).ravel()
    pred_labels = list()
    for i in y_pred:
        if i >= 0.5:
            pred_labels.append(1.0)
        else:
            pred_labels.append(0.0)
    pred_labels = np.array(pred_labels)
    pred_score = y_pred
    validation_labels = validation_labels

    accuracy = accuracy_score(validation_labels, pred_labels)
    precision = precision_score(validation_labels, pred_labels)
    recall = recall_score(validation_labels, pred_labels)
    f1 = f1_score(validation_labels, pred_labels)

    auroc = roc_auc_score(validation_labels, pred_score)
    fpr, tpr, thresholds = roc_curve(validation_labels, pred_score)
    auroc_by_auc = auc(fpr, tpr)

    auprc = average_precision_score(validation_labels, pred_score)
    precision_point, recall_point, thresholds = precision_recall_curve(validation_labels, pred_score)
    precision_point[(recall_point==0)] = 1.0
    auprc_by_auc = auc(recall_point, precision_point)

    print("accuracy=%s, precision=%s, recall=%s, f1=%s, auroc=%s, auprc=%s, auroc_by_auc=%s, auprc_by_auc=%s"%(accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc))
    return accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc

if __name__ == "__main__":
    time1 = time.time()
    ten_grna_fold_CIRLCE = dict()
    for i in range(10):
        ten_grna_fold_CIRLCE[i] = {"features":list(), "labels":list()}
    ten_grna_fold_CIRLCE = load_ten_grna_fold_circle(ten_grna_fold_CIRLCE)
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(10):
        xtrain, ytrain, xtest, ytest = get_ten_grna_fold_train_val(sgRNA_type_index=i, ten_grna_fold_CIRLCE=ten_grna_fold_CIRLCE)
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
        auroc_skl_sum += result[8]
        auprc_skl_sum += result[9]
        auroc_by_auc_sum += result[10]
        auprc_by_auc_sum += result[11]
        print('End of the training!!')
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (test_loss_sum/10, test_acc_sum/10, auroc_sum/10, auprc_sum/10, accuracy_sum/10, precision_sum/10, recall_sum/10, f1_sum/10, auroc_skl_sum/10, auprc_skl_sum/10, auroc_by_auc_sum/10, auprc_by_auc_sum/10))
    print(time.time()-time1)
