import os, time
import pandas as pd
import numpy as np
import tensorflow as tf
from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score
from positional_encoding import PositionalEncoding
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAvgPool1D, Conv1D, BatchNormalization, Activation, Dropout, LayerNormalization, Flatten, Conv2D, Reshape, Bidirectional, LSTM, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy

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

    def load_CIRCLE_dataset(self, encoding_method=encode_by_base_pair_vocabulary, sgRNA_type_index=None, out_dim=2):
        print("[INFO] ===== Start Loading dataset CIRCLE =====")
        circle_feature = []
        circle_labels = []
        ## load
        circle_dataset = pd.read_csv(r"../datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv")
        # 判断是否按照sgRNA分组
        if sgRNA_type_index == None: # 不按照sgRNA分组，随机分
            ## shuffle
            circle_dataset = shuffle(circle_dataset, random_state=42)
            ## encode
            for i, row in circle_dataset.iterrows():
                gRNA_seq = row['sgRNA_seq']
                target_seq = row['off_seq']
                label = row['label']
                pair_code = encoding_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
                circle_feature.append(pair_code)
                circle_labels.append(label)
            self.circle_feature = np.array(circle_feature)
            print(self.circle_feature[-3:])
            self.circle_labels = np.array(circle_labels)
            # self.circle_labels = to_categorical(self.circle_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
            print(self.circle_labels[-3:])
            print("[INFO] Encoded dataset CIRCLE feature with size of", self.circle_feature.shape)
            print("[INFO] The labels number of active off-target sites in dataset CIRCLE is {0}, the active+inactive is {1}.".format(len(self.circle_labels[self.circle_labels>0]), len(self.circle_labels)))
            ## split
            self.train_features, self.validation_features, self.train_labels, self.validation_labels = train_test_split(self.circle_feature, self.circle_labels, test_size=0.2, random_state=42, stratify=self.circle_labels)
        else: # 按照sgRNA分组
            SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
            self.train_features = list()
            self.train_labels = list()
            self.validation_features = list()
            self.validation_labels = list()
            if len(self.ten_grna_fold_CIRLCE[0]["features"]) != 0: # 如果已经分好了就直接从里面取
                for i in range(10):
                    if i != sgRNA_type_index:
                        print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[sgRNA_type_index]))
                        self.train_features.extend(self.ten_grna_fold_CIRLCE[i]["features"])
                        self.train_labels.extend(self.ten_grna_fold_CIRLCE[i]["labels"])
                    else:
                        print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[sgRNA_type_index]))
                        self.validation_features.extend(self.ten_grna_fold_CIRLCE[i]["features"])
                        self.validation_labels.extend(self.ten_grna_fold_CIRLCE[i]["labels"])
            else: # 如果是none就新分配十份
                print("[INFO] ===== Start Encoding 10-grna-fold dataset CIRCLE =====")
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
                        pair_code = encoding_method(on_target_seq=gRNA_seq, off_target_seq=target_seq)
                        self.ten_grna_fold_CIRLCE[i]["features"].append(pair_code)
                        self.ten_grna_fold_CIRLCE[i]["labels"].append(label)          
                    self.ten_grna_fold_CIRLCE[i]["features"] = np.array(self.ten_grna_fold_CIRLCE[i]["features"])
                    if encoding_method == encode_in_6_dimensions:
                        self.ten_grna_fold_CIRLCE[i]["features"] = self.ten_grna_fold_CIRLCE[i]["features"].reshape(self.ten_grna_fold_CIRLCE[i]["features"].shape[0], 1, self.MAX_STEPS, self.EMBED_SIZE)
                    if out_dim == 2:
                        self.ten_grna_fold_CIRLCE[i]["labels"] = to_categorical(self.ten_grna_fold_CIRLCE[i]["labels"])
                    self.ten_grna_fold_CIRLCE[i]["labels"] = np.array(self.ten_grna_fold_CIRLCE[i]["labels"])
                    print("[INFO] %s-th-grna-fold set, features shape=%s"%(i, str(self.ten_grna_fold_CIRLCE[i]["features"].shape)))
                    print("[INFO] %s-th-grna-fold set, labels shape=%s, and positive samples number = %s"%(i, str(self.ten_grna_fold_CIRLCE[i]["labels"].shape), len(self.ten_grna_fold_CIRLCE[i]["labels"][self.ten_grna_fold_CIRLCE[i]["labels"][:, 1]>0])))
                    if i != sgRNA_type_index:
                        print("[INFO] use %s-th-grna-fold grna (%s) for train"%(i, SGRNA_TYPE[i]))
                        self.train_features.extend(self.ten_grna_fold_CIRLCE[i]["features"])
                        self.train_labels.extend(self.ten_grna_fold_CIRLCE[i]["labels"])
                    else:
                        print("[INFO] use %s-th-grna-fold grna (%s) for validation"%(i, SGRNA_TYPE[i]))
                        self.validation_features.extend(self.ten_grna_fold_CIRLCE[i]["features"])
                        self.validation_labels.extend(self.ten_grna_fold_CIRLCE[i]["labels"])
                print("[INFO] ===== End Encoding 10-grna-fold dataset CIRCLE =====")
            self.train_features = np.array(self.train_features)
            self.train_labels = np.array(self.train_labels)
            self.validation_features = np.array(self.validation_features)
            self.validation_labels = np.array(self.validation_labels)
        print("[INFO] train_feature.shape = %s"%str(self.train_features.shape))
        print("[INFO] train_labels.shape = %s, and positive samples number = %s"%(self.train_labels.shape, len(self.train_labels[self.train_labels[:, 1]>0])))
        print("[INFO] validation_features.shape = %s"%str(self.validation_features.shape))
        print("[INFO] validation_labels.shape = %s, and positive samples number = %s"%(self.validation_labels.shape, len(self.validation_labels[self.validation_labels[:, 1]>0])))
        print("[INFO] ===== End Loading dataset CIRCLE =====")

    def FeedForwardNetwork(self, units_dim, model_dim):
        return Sequential([Dense(units_dim, activation='relu'),Dense(model_dim)])

    def add_encoder_layer(self, input_tensor):
        attention_output = MultiHeadAttention(num_heads=8, key_dim=8)(input_tensor, input_tensor)
        attention_output = Dropout(rate=0.2)(attention_output, training=True)
        out1 = LayerNormalization()(input_tensor + attention_output)

        ffn = self.FeedForwardNetwork(units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)
        ffn_output = ffn(out1)
        ffn_output = Dropout(rate=0.2)(ffn_output, training=True)
        out2 = LayerNormalization()(out1 + ffn_output)
        out3 = Dropout(rate=0.2)(out2)
        return out3

    def add_decoder_layer(self, input_tensor, encoder_output):
        attention_output_1 = MultiHeadAttention(num_heads=8, key_dim=8)(input_tensor, input_tensor)
        attention_output_1 = Dropout(rate=0.2)(attention_output_1, training=True)
        out1 = LayerNormalization()(input_tensor + attention_output_1)

        attention_output_2 = MultiHeadAttention(num_heads=8, key_dim=8)(encoder_output, encoder_output, out1)
        attention_output_2 = Dropout(rate=0.2)(attention_output_2, training=True)
        out2 = LayerNormalization()(out1 + attention_output_2)

        ffn = self.FeedForwardNetwork(units_dim=self.EMBED_SIZE, model_dim=self.EMBED_SIZE)
        ffn_output = ffn(out2)
        ffn_output = Dropout(rate=0.2)(ffn_output, training=True)
        out3 = LayerNormalization()(out2 + ffn_output)
        out4 = Dropout(rate=0.2)(out3)
        return out4

    def train_model(self):
        print("[INFO] ===== Start train =====")
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')

        inputs = Input(shape=(1, self.MAX_STEPS, self.EMBED_SIZE,))
        conv_1_output = Conv2D(60, (1, self.EMBED_SIZE), padding='valid', data_format='channels_first', kernel_initializer=initializer)(inputs)
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
        outputs = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(optimizer=Adam(learning_rate=1e-3), loss=CategoricalCrossentropy(), metrics=['acc', AUC(num_thresholds=4000, curve="ROC", name="auroc", label_weights=[0, 1]), AUC(num_thresholds=4000, curve="PR", name="auprc", label_weights=[0, 1])])

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
            batch_size=self.BATCH_SIZE,
            epochs=self.N_EPOCHS,
            validation_data=(self.validation_features, self.validation_labels),
            callbacks=callbacks
        )
        print("[INFO] ===== End train =====")

        model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
        test_loss, test_acc, auroc, auprc = model.evaluate(self.validation_features, self.validation_labels)
        accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc = self.compute_auroc_and_auprc(model)
        return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc

    def compute_auroc_and_auprc(self, model):
        y_pred = model.predict(self.validation_features)
        pred_labels = np.argmax(y_pred, axis=1)
        pred_score = y_pred[:, 1]
        validation_labels = self.validation_labels[:, 1]

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
    trainer = Trainer()
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(10):
        trainer.load_CIRCLE_dataset(encoding_method=encode_in_6_dimensions, sgRNA_type_index=i, out_dim=2)
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