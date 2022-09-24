import os, time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAvgPool1D, Conv1D, BatchNormalization, Activation, Dropout, LayerNormalization, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall

from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary
from metrics_utils import compute_auroc_and_auprc
from data_preprocessing_utils import load_CIRCLE_dataset
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

        self.train_features = None
        self.validation_features = None
        self.train_labels = None
        self.validation_labels = None

        self.ten_grna_fold_CIRLCE = dict()
        for i in range(10):
            self.ten_grna_fold_CIRLCE[i] = {"features":list(), "labels":list()}

    def train_model(self):
        print("[INFO] ===== Start train =====")
        inputs = Input(shape=(self.MAX_STEPS,))

        ### Encoder
        # input_dim支持的输入范围是[0, self.VOCABULARY_SIZE)
        # embedding & encoding
        encoder_embeddings = Embedding(input_dim=self.VOCABULARY_SIZE, output_dim=self.EMBED_SIZE)(inputs)
        encoder_positional_encoding = PositionalEncoding(max_steps=self.MAX_STEPS, max_dims=self.EMBED_SIZE)(encoder_embeddings)
        # 2 * Encoder Layer
        encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)

        ### Decoder
        decoder_embeddings = Embedding(input_dim=self.VOCABULARY_SIZE, output_dim=self.EMBED_SIZE)(inputs)
        decoder_positional_encoding = PositionalEncoding(max_steps=self.MAX_STEPS, max_dims=self.EMBED_SIZE)(decoder_embeddings)
        # 2 * Decoder Layer
        decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)

        main_branch = Flatten()(decoder_output)
        main_branch = Dense(32, activation='relu')(main_branch)
        outputs = Dense(1, activation='sigmoid')(main_branch)

        model = Model(inputs=inputs, outputs=outputs)
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
            x=self.train_features, y=self.train_labels,
            batch_size=self.BATCH_SIZE,
            epochs=self.N_EPOCHS,
            validation_data=(self.validation_features, self.validation_labels),
            callbacks=callbacks
        )
        print("[INFO] ===== End train =====")

        model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
        test_loss, test_acc, auroc, auprc = model.evaluate(self.validation_features, self.validation_labels)
        accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc = compute_auroc_and_auprc(model=model, out_dim=1, test_features=self.validation_features, test_labels=self.validation_labels)
        return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc

if __name__ == "__main__":
    time1 = time.time()
    trainer = Trainer()
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(10):
        trainer.ten_grna_fold_CIRLCE, trainer.train_features, trainer.train_labels, trainer.validation_features, trainer.validation_labels = load_CIRCLE_dataset(encoding_method=encode_by_base_pair_vocabulary, sgRNA_type_index=i, out_dim=1, ten_grna_fold_CIRLCE=trainer.ten_grna_fold_CIRLCE)
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
        result = trainer.train_model()
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
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (test_loss_sum/10, test_acc_sum/10, auroc_sum/10, auprc_sum/10, accuracy_sum/10, precision_sum/10, recall_sum/10, f1_sum/10, auroc_skl_sum/10, auprc_skl_sum/10, auroc_by_auc_sum/10, auprc_by_auc_sum/10))
    print(time.time()-time1)
