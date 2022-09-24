import os, time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Conv2D, BatchNormalization, Dropout, Reshape, Bidirectional, LSTM, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, Flatten, concatenate
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary
from metrics_utils import compute_auroc_and_auprc
from data_preprocessing_utils import load_CIRCLE_dataset, load_I_2_dataset
from transformer_utils import add_encoder_layer, add_decoder_layer
from positional_encoding import PositionalEncoding

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

        # 24 * 6 encoding
        self.train_features_1 = None
        self.validation_features_1 = None
        self.train_labels_1 = None
        self.validation_labels_1 = None

        self.ten_grna_fold_CIRLCE_1 = dict()
        for i in range(10):
            self.ten_grna_fold_CIRLCE_1[i] = {"features":list(), "labels":list()}

        # positional encoding
        self.train_features_2 = None
        self.validation_features_2 = None
        self.train_labels_2 = None
        self.validation_labels_2 = None

        self.ten_grna_fold_CIRLCE_2 = dict()
        for i in range(10):
            self.ten_grna_fold_CIRLCE_2[i] = {"features":list(), "labels":list()}


    def train_model(self):
        print("[INFO] ===== Start train =====")

        ###
        #
        #  cnn branch
        #
        ###

        inputs_1 = Input(shape=(1, self.MAX_STEPS, self.EMBED_SIZE,), name="input_1")
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        conv_1_output = Conv2D(60, (1, self.EMBED_SIZE), padding='valid', data_format='channels_first', kernel_initializer=initializer)(inputs_1)
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
        # linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        # linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        # outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(linear_2_output_dropout)

        ###
        #
        #  transformer branch
        #
        ###

        inputs_2 = Input(shape=(self.MAX_STEPS,), name="input_2")

        ### Encoder
        # input_dim支持的输入范围是[0, self.VOCABULARY_SIZE)
        # embedding & encoding
        encoder_embeddings = Embedding(input_dim=self.VOCABULARY_SIZE, output_dim=self.EMBED_SIZE)(inputs_2)
        encoder_positional_encoding = PositionalEncoding(max_steps=self.MAX_STEPS, max_dims=self.EMBED_SIZE)(encoder_embeddings)
        # 1 * Encoder Layer
        encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)

        ### Decoder
        decoder_embeddings = Embedding(input_dim=self.VOCABULARY_SIZE, output_dim=self.EMBED_SIZE)(inputs_2)
        decoder_positional_encoding = PositionalEncoding(max_steps=self.MAX_STEPS, max_dims=self.EMBED_SIZE)(decoder_embeddings)
        # 1 * Decoder Layer
        decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)

        transformer_branch = Flatten()(decoder_output)
        # transformer_branch = Dense(32, activation='relu')(transformer_branch)
        # outputs = Dense(1, activation='sigmoid')(transformer_branch)

        ###
        #
        #  merge branch
        #
        ###
        
        ensemble = concatenate([flatten_output, transformer_branch], axis=-1)
        ensemble = Dense(124, activation='relu')(ensemble)
        ensemble = BatchNormalization()(ensemble)
        ensemble = Dropout(rate=0.2)(ensemble)
        ensemble = Dense(32, activation='relu')(ensemble)
        ensemble = BatchNormalization()(ensemble)
        ensemble = Dropout(rate=0.2)(ensemble)
        output_tensor = Dense(1, activation='sigmoid', name="output")(ensemble)

        model = Model(inputs=[inputs_1, inputs_2], outputs=output_tensor)
        model.summary()

        model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['acc', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])

        eary_stopping = EarlyStopping(
                        monitor='loss', min_delta=0.0001,
                        patience=8, verbose=1, mode='auto')
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
            x={"input_1": self.train_features_1, "input_2": self.train_features_2},
            y={"output": self.train_labels_1},
            batch_size=self.BATCH_SIZE,
            epochs=self.N_EPOCHS,
            validation_data=({"input_1": self.validation_features_1, "input_2": self.validation_features_2}, {"output": self.validation_labels_1}),
            callbacks=callbacks
        )
        print("[INFO] ===== End train =====")

        model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
        test_loss, test_acc, auroc, auprc = model.evaluate({"input_1": self.validation_features_1, "input_2": self.validation_features_2}, {"output": self.validation_labels_1})
        accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc = compute_auroc_and_auprc(model=model, out_dim=1, test_features={"input_1": self.validation_features_1, "input_2": self.validation_features_2}, test_labels=self.validation_labels_1)
        return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc

if __name__ == "__main__":
    time1 = time.time()

    ## CIRCLE InDel leave-one-gRNA-out cross-validation (LOGOCV)
    # trainer = Trainer()
    # test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # for i in range(10):
    #     trainer.ten_grna_fold_CIRLCE_1, trainer.train_features_1, trainer.train_labels_1, trainer.validation_features_1, trainer.validation_labels_1 = load_CIRCLE_dataset(encoding_method=encode_in_6_dimensions, sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE_1)
    #     trainer.ten_grna_fold_CIRLCE_2, trainer.train_features_2, trainer.train_labels_2, trainer.validation_features_2, trainer.validation_labels_2 = load_CIRCLE_dataset(encoding_method=encode_by_base_pair_vocabulary, sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE_2)
    #     if os.path.exists("tcrispr_model.h5"):
    #         os.remove("tcrispr_model.h5")
    #     result = trainer.train_model()
    #     test_loss_sum += result[0]
    #     test_acc_sum += result[1]
    #     auroc_sum += result[2]
    #     auprc_sum += result[3]
    #     accuracy_sum += result[4]
    #     precision_sum += result[5]
    #     recall_sum += result[6]
    #     f1_sum += result[7]
    #     auroc_skl_sum += result[8]
    #     auprc_skl_sum += result[9]
    #     auroc_by_auc_sum += result[10]
    #     auprc_by_auc_sum += result[11]
    # print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (test_loss_sum/10, test_acc_sum/10, auroc_sum/10, auprc_sum/10, accuracy_sum/10, precision_sum/10, recall_sum/10, f1_sum/10, auroc_skl_sum/10, auprc_skl_sum/10, auroc_by_auc_sum/10, auprc_by_auc_sum/10))

    print(time.time()-time1)
