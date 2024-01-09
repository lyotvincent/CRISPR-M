import time, os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Attention, Dense, Conv2D, Conv1D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPooling1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU, concatenate, AveragePooling2D, MaxPooling2D, SeparableConv2D, MultiHeadAttention
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model

sys.path.append("../../codes")
from positional_encoding import PositionalEncoding
from transformer_utils import add_encoder_layer, add_decoder_layer


def m81212_n13(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model


def m81212_n13_without_LSTM(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    # bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    # bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(conv_2)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    # bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(conv_3)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    # bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(conv_4)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model
