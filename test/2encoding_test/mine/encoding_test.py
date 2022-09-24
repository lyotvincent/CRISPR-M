import os, time, sys, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAvgPool1D, Conv1D, BatchNormalization, Activation, Dropout, LayerNormalization, Flatten, Conv2D, Reshape, Bidirectional, LSTM, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy

sys.path.append("../../../codes")
from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary, encode_by_one_hot, encode_by_crispr_net_method, encode_by_crispr_net_method_with_isPAM, encode_by_crispr_ip_method, encode_by_crispr_ip_method_without_minus, encode_by_crispr_ip_method_without_isPAM
from metrics_utils import compute_auroc_and_auprc
from data_preprocessing_utils import load_CIRCLE_dataset, load_I_2_dataset, load_CIRCLE_dataset_encoded_by_base_vocabulary, load_CIRCLE_dataset_encoded_by_both_base_and_base_pair, load_I_2_dataset_encoded_by_both_base_and_base_pair
from test_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Trainer:
    def __init__(self) -> None:
        self.VOCABULARY_SIZE = 50 # 词典的长度，就是句子里的词有几种。在这里是base pair有几种。口算应该是(4+8+12)*2=48种。再加两种__
        self.MAX_STEPS = 24 # 句子长度。21bpGRNA+3bpPAM = 24。
        self.EMBED_SIZE = 6 # 句子里每个词的向量长度。base pair的编码后的向量的长度，编码前是我设计的长度6的向量，编码后是embed_size。

        self.BATCH_SIZE = 1024
        self.N_EPOCHS = 500

        self.circle_feature = None
        self.circle_labels = None

        self.train_features = None
        self.train_feature_ont =  None
        self.train_feature_offt =  None
        self.validation_features = None
        self.validation_feature_ont =  None
        self.validation_feature_offt =  None
        self.train_labels = None
        self.validation_labels = None

        self.ten_grna_fold_CIRLCE = dict()
        for i in range(10):
            # self.ten_grna_fold_CIRLCE[i] = {"features":list(), "labels":list()}
            # self.ten_grna_fold_CIRLCE[i] = {"feature_ont":list(), "feature_offt":list(), "labels":list()}
            self.ten_grna_fold_CIRLCE[i] = {"features":list(), "feature_ont":list(), "feature_offt":list(), "labels":list()}

    def train_model(self):
        print("[INFO] ===== Start train =====")

        # initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        # inputs = Input(shape=(24, 6,))
        # conv_1_output_reshape = Reshape(tuple([1, 24, 6]))(inputs)
        # conv_1_output = Conv2D(60, (1,6), padding='valid', data_format='channels_first', kernel_initializer=initializer)(conv_1_output_reshape)
        # conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
        # conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
        # conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        # conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        # bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        # attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # flatten_output = Flatten()(concat_output)
        # linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        # linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        # outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(linear_2_output_dropout)

        # inputs = Input(shape=(24, 6,))
        # main = Flatten()(inputs)
        # main = Dense(100, activation='relu')(main)
        # main = Dense(100, activation='relu')(main)
        # main = Dense(100, activation='relu')(main)
        # outputs = Dense(1, activation='sigmoid')(main)

        # inputs = Input(shape=(24, 6,))
        # main = Conv1D(10, 3)(inputs)
        # main = Conv1D(10, 3)(main)
        # main = Conv1D(10, 3)(main)
        # main = Flatten()(main)
        # main = Dense(100, activation='relu')(main)
        # outputs = Dense(1, activation='sigmoid')(main)

        # inputs = Input(shape=(24, 6,))
        # main = LSTM(30, return_sequences=True)(inputs)
        # main = Flatten()(main)
        # main = Dense(100)(main)
        # outputs = Dense(1, activation='sigmoid')(main)

        # model = Model(inputs=inputs, outputs=outputs)
        # model.summary()

        # model = model_4(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7)
        model = model_4(VOCABULARY_SIZE=25, MAX_STEPS=24, EMBED_SIZE=7)
        # model = m81212_n13()
        # model = model_for_transformer_using_seperate_onofftarget()

        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['acc', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])

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

        history = model.fit(
            x=self.train_features, y=self.train_labels,
            # x={"input_1": self.train_feature_ont, "input_2": self.train_feature_offt}, y={"output": self.train_labels},
            # x={"input_1": self.train_features, "input_2": self.train_feature_ont, "input_3": self.train_feature_offt}, y={"output": self.train_labels},
            batch_size=self.BATCH_SIZE,
            epochs=self.N_EPOCHS,
            validation_data=(self.validation_features, self.validation_labels),
            # validation_data=({"input_1": self.validation_feature_ont, "input_2": self.validation_feature_offt}, {"output": self.validation_labels}),
            # validation_data=({"input_1": self.validation_features, "input_2": self.validation_feature_ont, "input_3": self.validation_feature_offt}, {"output": self.validation_labels}),
            callbacks=callbacks
        )
        print("[INFO] ===== End train =====")

        model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
        test_loss, test_acc, auroc, auprc = model.evaluate(self.validation_features, self.validation_labels)
        # test_loss, test_acc, auroc, auprc = model.evaluate(x={"input_1": self.validation_features, "input_2": self.validation_feature_ont, "input_3": self.validation_feature_offt}, y=self.validation_labels)
        # accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels = compute_auroc_and_auprc(model=model, out_dim=1, test_features=self.validation_features, test_labels=self.validation_labels)
        accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=1, test_features=self.validation_features, test_labels=self.validation_labels)
        # accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=1, test_features=[self.validation_features, self.validation_feature_ont, self.validation_feature_offt], test_labels=self.validation_labels)
        return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point

if __name__ == "__main__":
    time1 = time.time()

    # CIRCLE InDel leave-one-gRNA-out cross-validation (LOGOCV)
    trainer = Trainer()
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    for i in range(10):
        ## load_CIRCLE_dataset
        trainer.ten_grna_fold_CIRLCE, trainer.train_features, trainer.train_labels, trainer.validation_features, trainer.validation_labels = load_CIRCLE_dataset(encoding_method=encode_by_base_pair_vocabulary, sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE)
        ## load_CIRCLE_dataset_encoded_by_base_vocabulary
        # trainer.ten_grna_fold_CIRLCE, trainer.train_feature_ont, trainer.train_feature_offt, trainer.train_labels, trainer.validation_feature_ont, trainer.validation_feature_offt, trainer.validation_labels = load_CIRCLE_dataset_encoded_by_base_vocabulary(sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE)
        ## load_CIRCLE_dataset_encoded_by_both_base_and_base_pair
        # trainer.ten_grna_fold_CIRLCE, trainer.train_features, trainer.train_feature_ont, trainer.train_feature_offt, trainer.train_labels, trainer.validation_features, trainer.validation_feature_ont, trainer.validation_feature_offt, trainer.validation_labels = load_CIRCLE_dataset_encoded_by_both_base_and_base_pair(sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE)
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
        result = trainer.train_model()
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
    print("=====")
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum/10, test_acc_sum/10, auroc_sum/10, auprc_sum/10, accuracy_sum/10, precision_sum/10, recall_sum/10, f1_sum/10, fbeta_sum/10, auroc_skl_sum/10, auprc_skl_sum/10, auroc_by_auc_sum/10, auprc_by_auc_sum/10, spearman_corr_by_pred_score_sum/10, spearman_corr_by_pred_labels_sum/10))

    # ## CIRCLE-train I-2-validation
    # trainer = Trainer()
    # # encoding_method = encode_by_base_pair_vocabulary
    # # trainer.train_features, trainer.train_labels = load_CIRCLE_dataset(encoding_method=encoding_method)
    # trainer.train_features, trainer.train_feature_ont, trainer.train_feature_offt, trainer.train_labels = load_CIRCLE_dataset_encoded_by_both_base_and_base_pair()
    # # trainer.validation_features, trainer.validation_labels = load_I_2_dataset(encoding_method=encoding_method)
    # trainer.validation_features, trainer.validation_feature_ont, trainer.validation_feature_offt, trainer.validation_labels = load_I_2_dataset_encoded_by_both_base_and_base_pair()
    # test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    # for i in range(10):
    #     if os.path.exists("tcrispr_model.h5"):
    #         os.remove("tcrispr_model.h5")
    #     result = trainer.train_model()
    #     if os.path.exists("tcrispr_model.h5"):
    #         os.remove("tcrispr_model.h5")
    #     # test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12], result[13], result[14]
    #     # print("i=%s test_loss=%s, test_acc=%s, auroc=%s, auprc=%s, accuracy=%s, precision=%s, recall=%s, f1=%s, fbeta=%s, auroc_skl=%s, auprc_skl=%s, auroc_by_auc=%s, auprc_by_auc=%s, spearman_corr_by_pred_score=%s, spearman_corr_by_pred_labels=%s" % (i, test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels))
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

    print(time.time()-time1)
