import os, time, sys, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import to_categorical
from crispr_ip_encoding import my_encode_on_off_dim
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.metrics import AUC, Precision, Recall

sys.path.append("../../../codes")
from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary, encode_by_one_hot, encode_by_crispr_net_method, encode_by_crispr_net_method_with_isPAM, encode_by_crispr_ip_method, encode_by_crispr_ip_method_without_minus, encode_by_crispr_ip_method_without_isPAM
import data_preprocessing_utils
from metrics_utils import compute_auroc_and_auprc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def transformIO(xtrain, xtest, ytrain, ytest, seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1, seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0], 1, seq_len, coding_dim)
    input_shape = (1, seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape


def load_CIRCLE_dataset(sgRNA_type_index=None):
    print("[INFO] ===== Start Encoding dataset CIRCLE =====")
    ## load
    circle_dataset = pd.read_csv(r"../datasets/CIRCLE(mismatch&insertion&deletion)/CIRCLE_seq_data.csv")
    # CIRCLE dataset里一共十种sgRNA，可以用来作10-fold交叉验证
    SGRNA_TYPE = ['GAACACAAAGCATAGACTGCNGG', 'GGGAAAGACCCAGCATCCGTNGG', 'GGCACTGCGGCTGGAGGTGGNGG', 'GGAATCCCTTCTGCAGCACCNGG', 'GAGTCCGAGCAGAAGAAGAANGG', 'GTTGCCCCACAGGGCAGTAANGG', 'GACCCCCTCCACCCCGCCTCNGG', 'GGCCCAGACTGAGCACGTGANGG', 'GGGTGGGGGGAGTTTGCTCCNGG', 'GGTGAGTGAGTGTGTGCGTGNGG']
    print("[INFO] 10-fold, use %s-th grna (%s) for validation"%(sgRNA_type_index, SGRNA_TYPE[sgRNA_type_index]))
    train_dataset = circle_dataset[circle_dataset['sgRNA_type']!=SGRNA_TYPE[sgRNA_type_index]]
    print("[INFO] 9/10 grna set for train, shape=%s"%str(train_dataset.shape))
    validation_dataset = circle_dataset[circle_dataset['sgRNA_type']==SGRNA_TYPE[sgRNA_type_index]]
    print("[INFO] 1/10 grna set for validation, shape=%s"%str(validation_dataset.shape))
    ## encode train_dataset
    train_features = list()
    train_labels = list()
    for _, row in train_dataset.iterrows():
        gRNA_seq = row['sgRNA_seq']
        target_seq = row['off_seq']
        label = row['label']
        pair_code = my_encode_on_off_dim(target_seq=gRNA_seq, off_target_seq=target_seq)
        train_features.append(pair_code)
        train_labels.append(label)
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    ## encode validation_dataset
    validation_features = list()
    validation_labels = list()
    for _, row in validation_dataset.iterrows():
        gRNA_seq = row['sgRNA_seq']
        target_seq = row['off_seq']
        label = row['label']
        pair_code = my_encode_on_off_dim(target_seq=gRNA_seq, off_target_seq=target_seq)
        validation_features.append(pair_code)
        validation_labels.append(label)
    validation_features = np.array(validation_features)
    validation_labels = np.array(validation_labels)
    print("[INFO] train_feature.shape = %s"%str(train_features.shape))
    print("[INFO] train_labels.shape = %s, and positive samples number = %s"%(train_labels.shape, len(train_labels[train_labels>0])))
    print("[INFO] validation_features.shape = %s"%str(validation_features.shape))
    print("[INFO] validation_labels.shape = %s, and positive samples number = %s"%(validation_labels.shape, len(validation_labels[validation_labels>0])))
    print("[INFO] ===== End Encoding dataset CIRCLE =====")
    return train_features, train_labels, validation_features, validation_labels

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
            pair_code = my_encode_on_off_dim(target_seq=gRNA_seq, off_target_seq=target_seq)
            ten_grna_fold_CIRLCE[i]["features"].append(pair_code)
            ten_grna_fold_CIRLCE[i]["labels"].append(label)          
        ten_grna_fold_CIRLCE[i]["features"] = np.array(ten_grna_fold_CIRLCE[i]["features"])
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

# def compute_auroc_and_auprc(model, validation_features, validation_labels):
#     y_pred = model.predict(validation_features)
#     pred_labels = np.argmax(y_pred, axis=1)
#     pred_score = y_pred[:, 1]
#     validation_labels = validation_labels[:, 1]

#     accuracy = accuracy_score(validation_labels, pred_labels)
#     precision = precision_score(validation_labels, pred_labels)
#     recall = recall_score(validation_labels, pred_labels)
#     f1 = f1_score(validation_labels, pred_labels)

#     auroc = roc_auc_score(validation_labels, pred_score)
#     fpr, tpr, thresholds = roc_curve(validation_labels, pred_score)
#     auroc_by_auc = auc(fpr, tpr)

#     auprc = average_precision_score(validation_labels, pred_score)
#     precision_point, recall_point, thresholds = precision_recall_curve(validation_labels, pred_score)
#     precision_point[(recall_point==0)] = 1.0
#     auprc_by_auc = auc(recall_point, precision_point)

#     # Spearman's rank correlation coefficient
#     df = pd.DataFrame({"y_pred": pred_score, "y_label": validation_labels})
#     spearman_corr_by_pred_score = df.corr("spearman")["y_pred"]["y_label"]
#     df = pd.DataFrame({"y_pred": pred_labels, "y_label": validation_labels})
#     spearman_corr_by_pred_labels = df.corr("spearman")["y_pred"]["y_label"]

#     print("accuracy=%s, precision=%s, recall=%s, f1=%s, auroc=%s, auprc=%s, auroc_by_auc=%s, auprc_by_auc=%s, spearman_corr_by_pred_score=%s, spearman_corr_by_pred_labels=%s"%(accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels))
#     return accuracy, precision, recall, f1, auroc, auprc, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels

def crispr_ip(xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+crispr_ip.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        # (batch_size,  channels,   height, width)
        # (None,        1,          24,     7)
        # input_shape = (1, 24, 7)
        conv_1_output = Conv2D(60, (1,input_shape[-1]), padding='valid', data_format='channels_first', kernel_initializer=initializer)(input_value)
        # (batch_size,  channels,   height, width)
        # [None,        60,         24,     1]
        print("conv_1_output=%s"%str(conv_1_output.shape.as_list()))
        print("conv_1_output=%s"%str(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None])))
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
        # (batch_size,  channels,   height)
        # [None,        60,         24]
        print("conv_1_output_reshape=%s"%str(conv_1_output_reshape.shape.as_list()))
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
        # (batch_size,  height, channels)
        # [None,        24,     60]
        print("conv_1_output_reshape2=%s"%str(conv_1_output_reshape2.shape.as_list()))
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        # (batch_size,  height, channels)
        # [None,        24,     30]
        print("conv_1_output_reshape_average=%s"%str(conv_1_output_reshape_average.shape.as_list()))
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        # (batch_size,  height, channels)
        # [None,        24,     30]
        print("conv_1_output_reshape_max=%s"%str(conv_1_output_reshape_max.shape.as_list()))
        bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        # bidirectional_1_output shape (None, 24, 60)
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        # (None, 24, 60)
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        # (None, 60)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        # (None, 60)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # (None, 120)
        flatten_output = Flatten()(concat_output)
        # (None, 120)
        linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        # (None, 200)
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        # (None, 100)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.summary()
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', AUC(num_thresholds=4000, curve="ROC", name="auroc", label_weights=[0, 1]), AUC(num_thresholds=4000, curve="PR", name="auprc", label_weights=[0, 1])])
        history_model = model.fit(
            xtrain, ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(xtest, ytest),
            callbacks=callbacks
        )
        model.save('{}crispr_ip.h5'.format(saved_prefix))
    model = load_model('{}crispr_ip.h5'.format(saved_prefix))
    print("shape = %s, %s"%(str(xtest.shape), str(ytest.shape)))
    test_loss, test_acc, auroc, auprc = model.evaluate(xtest, ytest)
    accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=num_classes, test_features=xtest, test_labels=ytest)
    return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point


if __name__ == "__main__":
    time1 = time.time()
    num_classes = 2
    epochs = 500
    batch_size = 4000
    # batch_size = 512
    retrain=True
    encoder_shape=(24,7)
    seg_len, coding_dim = encoder_shape

    eary_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.0001,
        patience=5, verbose=0, mode='auto')
    callbacks = [eary_stopping]

    # ten_grna_fold_CIRLCE = dict()
    # for i in range(10):
    #     ten_grna_fold_CIRLCE[i] = {"features":list(), "labels":list()}
    # ten_grna_fold_CIRLCE = load_ten_grna_fold_circle(ten_grna_fold_CIRLCE)
    # test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    # for i in range(10):
    #     xtrain, ytrain, xtest, ytest = get_ten_grna_fold_train_val(sgRNA_type_index=i, ten_grna_fold_CIRLCE=ten_grna_fold_CIRLCE)
    #     xtrain, xtest, ytrain, ytest, inputshape = transformIO(xtrain, xtest, ytrain, ytest, seg_len, coding_dim, num_classes)
    #     print("inputshape = %s"%str(inputshape))
    #     print("xtrain = %s"%str(xtrain.shape))
    #     print("ytrain = %s"%str(ytrain.shape))
    #     print("xtest = %s"%str(xtest.shape))
    #     print("ytest = %s"%str(ytest.shape))
    #     print('Training!!')
    #     if os.path.exists("crispr_ip.h5"):
    #         os.remove("crispr_ip.h5")
    #     result = crispr_ip(xtrain, ytrain, xtest, ytest, inputshape, num_classes, batch_size, epochs, callbacks, 
    #                                 '', retrain)
    #     if os.path.exists("crispr_ip.h5"):
    #         os.remove("crispr_ip.h5")
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
    
    encoding_method = encode_by_crispr_ip_method
    xtrain, ytrain = data_preprocessing_utils.load_CIRCLE_dataset(encoding_method=encoding_method)
    xtest, ytest = data_preprocessing_utils.load_I_2_dataset(encoding_method=encoding_method)
    xtrain, xtest, ytrain, ytest, inputshape = transformIO(xtrain, xtest, ytrain, ytest, seg_len, coding_dim, num_classes)
    print("inputshape = %s"%str(inputshape))
    print("xtrain = %s"%str(xtrain.shape))
    print("ytrain = %s"%str(ytrain.shape))
    print("xtest = %s"%str(xtest.shape))
    print("ytest = %s"%str(ytest.shape))
    print('Training!!')
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    for i in range(10):
        if os.path.exists("crispr_ip.h5"):
            os.remove("crispr_ip.h5")
        result = crispr_ip(xtrain, ytrain, xtest, ytest, inputshape, num_classes, batch_size, epochs, callbacks, 
                                        '', retrain)
        if os.path.exists("crispr_ip.h5"):
            os.remove("crispr_ip.h5")
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
    
    
    print(time.time()-time1)
