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

def model_4(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=6):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)

    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    conv_1_output_reshape = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    conv_1_output = Conv2D(60, (1,EMBED_SIZE), padding='valid', data_format='channels_first', kernel_initializer=initializer)(conv_1_output_reshape)
    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1_output_reshape2)
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

    # main = Flatten()(encoder_positional_encoding)
    # main = Dense(100, activation='relu')(main)
    # main = Dense(100, activation='relu')(main)
    # main = Dense(100, activation='relu')(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    # main = Conv1D(10, 3)(encoder_positional_encoding)
    # main = Conv1D(10, 3)(main)
    # main = Conv1D(10, 3)(main)
    # main = Flatten()(main)
    # main = Dense(100, activation='relu')(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    # main = LSTM(30, return_sequences=True)(encoder_positional_encoding)
    # main = Flatten()(main)
    # main = Dense(100)(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def model_for_transformer(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    # 1 * Encoder Layer
    encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    # encoder_output = add_encoder_layer(encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    
    ### Decoder
    decoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    decoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(decoder_embeddings)
    # 2 * Decoder Layer
    decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    decoder_output = add_decoder_layer(decoder_output, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

    # main = LSTM(30, return_sequences=True)(decoder_output)
    main = Flatten()(decoder_output)
    main = Dense(32, activation='relu')(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_t1(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    print("[INFO] ===== Start train =====")

    ###
    #
    #  cnn branch
    #
    ###

    inputs_1 = Input(shape=(MAX_STEPS,))
    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    conv_1_output_reshape = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding_1)
    conv_1_output = Conv2D(60, (1, EMBED_SIZE), padding='valid', data_format='channels_first', kernel_initializer=initializer)(conv_1_output_reshape)
    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1_output_reshape2)
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

    ### Encoder
    # input_dim支持的输入范围是[0, VOCABULARY_SIZE)
    # embedding & encoding
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    # 1 * Encoder Layer
    encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

    ### Decoder
    decoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    decoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(decoder_embeddings)
    # 1 * Decoder Layer
    decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

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

    model = Model(inputs=[inputs_1], outputs=output_tensor)
    model.summary()
    return model

def m81212_i1(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)

    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    conv_1_output_reshape = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    conv_1_output = Conv2D(60, (1,EMBED_SIZE), padding='valid', data_format='channels_first', kernel_initializer=initializer)(conv_1_output_reshape)
    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1_output_reshape2)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(bidirectional_1_output)
    # attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    flatten_output = Flatten()(concat_output)
    linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
    linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
    linear_2_output_dropout = Dropout(0.9)(linear_2_output)
    outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(linear_2_output_dropout)

    # main = Flatten()(encoder_positional_encoding)
    # main = Dense(100, activation='relu')(main)
    # main = Dense(100, activation='relu')(main)
    # main = Dense(100, activation='relu')(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    # main = Conv1D(10, 3)(encoder_positional_encoding)
    # main = Conv1D(10, 3)(main)
    # main = Conv1D(10, 3)(main)
    # main = Flatten()(main)
    # main = Dense(100, activation='relu')(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    # main = LSTM(30, return_sequences=True)(encoder_positional_encoding)
    # main = Flatten()(main)
    # main = Dense(100)(main)
    # outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i2(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Conv1D(60, 3)(encoder_positional_encoding)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(main)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    main = Flatten()(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i3(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Conv1D(60, 3)(encoder_positional_encoding)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(main)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    main = Flatten()(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i4(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (1, EMBED_SIZE), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(main)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    main = Flatten()(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i5(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(main)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    main = Flatten()(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i6(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i7(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (3,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(60, (3,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i8(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (1,2), padding='valid', data_format='channels_first')(main)
    avgpool_1 = AveragePooling2D(pool_size=(1, 2), data_format='channels_first')(main)
    maxpool_1 = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(main)
    con = Concatenate(axis=1)([avgpool_1, maxpool_1])
    main = Conv2D(120, (1,3), padding='valid', data_format='channels_first')(con)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_t2(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    print("[INFO] ===== Start train =====")

    ###
    #
    #  cnn branch
    #
    ###

    inputs_1 = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    flatten_output = Flatten()(concat_output)

    ###
    #
    #  transformer branch
    #
    ###

    ### Encoder
    # input_dim支持的输入范围是[0, VOCABULARY_SIZE)
    # embedding & encoding
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    # 1 * Encoder Layer
    encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

    ### Decoder
    decoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    decoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(decoder_embeddings)
    # 1 * Decoder Layer
    decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

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

    model = Model(inputs=[inputs_1], outputs=output_tensor)
    model.summary()
    return model

def m81212_i9(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_output)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i10(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = SeparableConv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = SeparableConv2D(60, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_i11(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)
    main = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(main)
    main = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(main)
    main = Reshape(tuple([x for x in main.shape.as_list() if x != 1 and x is not None]))(main)
    main = tf.transpose(main, perm=[0,2,1])
    main = Conv1D(64, 3)(main)
    main = Conv1D(64, 3)(main)
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(main)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(main)
    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(con)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dense(100)(main)
    main = Dropout(0.9)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n1(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)

    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(main)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1)

    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(main)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_2)

    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average, conv_2_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(con)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n2(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    main = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding)

    conv_1 = add_encoder_layer(main, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1)

    conv_2 = add_encoder_layer(main, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_2)

    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average, conv_2_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(con)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n3(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding_1)
    branch_1 = add_encoder_layer(branch_1, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding_2)
    branch_2 = add_encoder_layer(branch_2, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_2)

    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average, conv_2_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(con)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n4(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding_1)
    branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(branch_1, branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(encoder_positional_encoding_2)
    branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(branch_2, branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_2)

    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average, conv_2_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(con)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n5(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_1, encoder_positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_2, encoder_positional_encoding_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_output_reshape_max = MaxPooling1D(data_format='channels_first')(conv_2)

    con = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average, conv_2_output_reshape_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(con)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(bidirectional_1_output)
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(bidirectional_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(bidirectional_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    main = Flatten()(concat_output)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(100)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n6(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_1, encoder_positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    bidirectional_1_output = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_2, encoder_positional_encoding_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    bidirectional_2_output = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(conv_2)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n7(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=10):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = MultiHeadAttention(num_heads=8, key_dim=EMBED_SIZE)(encoder_positional_encoding_1, encoder_positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    bidirectional_1_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = MultiHeadAttention(num_heads=8, key_dim=EMBED_SIZE)(encoder_positional_encoding_2, encoder_positional_encoding_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_2)
    conv_2 = Conv2D(64, (1,EMBED_SIZE), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    bidirectional_2_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_2)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, encoder_positional_encoding_1, encoder_positional_encoding_2])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def model_for_transformer_using_seperate_onofftarget(VOCABULARY_SIZE=5, MAX_STEPS=24, EMBED_SIZE=7):
    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    encoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_1)
    encoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings)
    # 1 * Encoder Layer
    encoder_output = add_encoder_layer(encoder_positional_encoding, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    # encoder_output = add_encoder_layer(encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    ### Decoder
    decoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs_2)
    decoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(decoder_embeddings)
    # 2 * Decoder Layer
    decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    # decoder_output = add_decoder_layer(decoder_output, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

    # main = LSTM(30, return_sequences=True)(decoder_output)
    main = Flatten()(decoder_output)
    main = Dense(32, activation='relu')(main)
    outputs = Dense(1, activation='sigmoid', name="output")(main)

    model = Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model.summary()
    return model

def m81212_n8(VOCABULARY_SIZE=50, MAX_STEPS=24, EMBED_SIZE=7):
    inputs = Input(shape=(MAX_STEPS,))

    encoder_embeddings_1 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_1 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_1)
    branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_1, encoder_positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(branch_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    bidirectional_1_output = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(conv_1)

    encoder_embeddings_2 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_2 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_2)
    branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(encoder_positional_encoding_2, encoder_positional_encoding_2)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(branch_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    bidirectional_2_output = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(conv_2)

    encoder_embeddings_3 = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    encoder_positional_encoding_3 = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(encoder_embeddings_3)
    # 1 * Encoder Layer
    encoder_output = add_encoder_layer(encoder_positional_encoding_3, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    ### Decoder
    decoder_embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)(inputs)
    decoder_positional_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)(decoder_embeddings)
    # 2 * Decoder Layer
    decoder_output = add_decoder_layer(decoder_positional_encoding, encoder_output, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    
    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, decoder_output])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid')(main)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def m81212_n9(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
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

    # branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_3, positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    # conv_1 = Conv1D(128, 5)(conv_1)
    # conv_1 = MaxPooling1D()(conv_1)
    # conv_1 = Conv1D(128, 5)(conv_1)
    # conv_1 = MaxPooling1D()(conv_1)
    bidirectional_1_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_1)

    # branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_3, positional_encoding_1)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    # conv_2 = Conv1D(128, 5)(conv_2)
    # conv_2 = MaxPooling1D()(conv_2)
    # conv_2 = Conv1D(256, 5)(conv_2)
    # conv_2 = MaxPooling1D()(conv_2)
    bidirectional_2_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_3 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    conv_3 = tf.transpose(conv_3, perm=[0,2,1])
    # conv_3 = Conv1D(128, 5)(conv_3)
    # conv_3 = MaxPooling1D()(conv_3)
    # conv_3 = Conv1D(256, 5)(conv_3)
    # conv_3 = MaxPooling1D()(conv_3)
    bidirectional_3_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_3)

    branch_4 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    conv_4 = tf.transpose(conv_4, perm=[0,2,1])
    # conv_4 = Conv1D(128, 5)(conv_4)
    # conv_4 = MaxPooling1D()(conv_4)
    # conv_4 = Conv1D(256, 5)(conv_4)
    # conv_4 = MaxPooling1D()(conv_4)
    bidirectional_4_output = Bidirectional(GRU(32, return_sequences=True, dropout=0.2))(conv_4)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n9(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
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

    # branch_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_3, positional_encoding_1)
    branch_1 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_1)
    # branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid', data_format='channels_first')(conv_1)
    # conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid', data_format='channels_first')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1 = tf.transpose(conv_1, perm=[0,2,1])
    # conv_1 = Conv1D(128, 5)(conv_1)
    # conv_1 = MaxPooling1D()(conv_1)
    # conv_1 = Conv1D(128, 5)(conv_1)
    # conv_1 = MaxPooling1D()(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    # branch_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_3, positional_encoding_1)
    branch_2 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_1)
    # branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2 = tf.transpose(conv_2, perm=[0,2,1])
    # conv_2 = Conv1D(128, 5)(conv_2)
    # conv_2 = MaxPooling1D()(conv_2)
    # conv_2 = Conv1D(256, 5)(conv_2)
    # conv_2 = MaxPooling1D()(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_3 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_2)
    # branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    conv_3 = tf.transpose(conv_3, perm=[0,2,1])
    # conv_3 = Conv1D(128, 5)(conv_3)
    # conv_3 = MaxPooling1D()(conv_3)
    # conv_3 = Conv1D(256, 5)(conv_3)
    # conv_3 = MaxPooling1D()(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)

    branch_4 = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_3)
    # branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid', data_format='channels_first')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    conv_4 = tf.transpose(conv_4, perm=[0,2,1])
    # conv_4 = Conv1D(128, 5)(conv_4)
    # conv_4 = MaxPooling1D()(conv_4)
    # conv_4 = Conv1D(256, 5)(conv_4)
    # conv_4 = MaxPooling1D()(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n10(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
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
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n11(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
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

    attention_1 = add_encoder_layer(positional_encoding_1, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    attention_2 = add_encoder_layer(positional_encoding_2, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)
    attention_3 = add_encoder_layer(positional_encoding_3, num_heads=8, key_dim=6, units_dim=EMBED_SIZE, model_dim=EMBED_SIZE)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n12(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n13(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n14(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)

    con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model

def m81212_n15(VOCABULARY_SIZE=55, MAX_STEPS=24, EMBED_SIZE=7):
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
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)

    con = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    model.summary()
    return model