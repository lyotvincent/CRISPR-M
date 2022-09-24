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

def model_1():

    inputs = Input(shape=(24, 7,))
    main = Conv1D(10, 3)(inputs)
    main = Conv1D(10, 3)(main)
    main = Conv1D(10, 3)(main)
    main = Bidirectional(LSTM(30, return_sequences=True))(main)
    main = Attention()([main, main])
    main = Flatten()(main)
    main = Dense(100, activation='relu')(main)
    main = Dense(100, activation='relu')(main)
    main = Dense(100, activation='relu')(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs, outputs)
    print(model.summary())
    return model
# average_test_loss=0.04120755698531866, average_test_acc=0.9881451070308686, average_auroc=0.9647179424762726, average_auprc=0.483604896068573, average_accuracy=0.9881451191387557, average_precision=0.7278143288586923, average_recall=0.24215536966203888, average_f1=0.314858178238003, average_auroc_skl=0.9670462568219926, average_auprc_skl=0.48498075394081913, average_auroc_by_auc=0.9670462568219926, average_auprc_by_auc=0.48377013766081384
###############################################################################################

def model_2():

    inputs = Input(shape=(24, 7,))
    main = Conv1D(10, 3)(inputs)
    main = Conv1D(10, 3)(main)
    main = Bidirectional(LSTM(30, return_sequences=True))(main)
    main = Flatten()(main)
    main = Dense(100, activation='relu')(main)
    main = Dense(100, activation='relu')(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs, outputs)
    print(model.summary())
    return model
# average_test_loss=0.05074595138430595, average_test_acc=0.9860087752342224, average_auroc=0.9587154030799866, average_auprc=0.4667608305811882, average_accuracy=0.9860087706059089, average_precision=0.6601385147976292, average_recall=0.2882084328282516, average_f1=0.2764510093012992, average_auroc_skl=0.9645676543563528, average_auprc_skl=0.4700300910805565, average_auroc_by_auc=0.9645676543563528, average_auprc_by_auc=0.46876454658528555
###############################################################################################

def model_3():

    inputs = Input(shape=(24, 7,))
    main = Conv1D(10, 3)(inputs)
    main = Bidirectional(LSTM(30, return_sequences=True))(main)
    main = Flatten()(main)
    main = Dense(100, activation='relu')(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs, outputs)
    print(model.summary())
    return model
# average_test_loss=0.042009782418608664, average_test_acc=0.987685602903366, average_auroc=0.9636790692806244, average_auprc=0.494310662150383, average_accuracy=0.9876856058562943, average_precision=0.6345087267135364, average_recall=0.32672256773156116, average_f1=0.3399826233828887, average_auroc_skl=0.9686960458646625, average_auprc_skl=0.4962469231874608, average_auroc_by_auc=0.9686960458646625, average_auprc_by_auc=0.4950886889996404
###############################################################################################



