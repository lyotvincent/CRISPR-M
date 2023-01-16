
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims

        if max_dims % 2 == 1: max_dims += 1 # max_dims must be even

        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_steps': self.max_steps,
            'max_dims': self.max_dims
        })
        return config

def __test1():
    print(tf.constant([[1,2],[3,4]])+tf.constant([5,6]))
    max_steps = 24
    max_dims = 6
    pos_emb = PositionalEncoding(max_steps, max_dims)
    test_data = np.zeros((2, max_steps, max_dims), np.float32)
    # print(test_data)
    print(test_data.shape)
    PE = pos_emb(test_data).numpy()
    # print(PE)
    print(PE.shape)

def __test2():
    vocab_size = 48 # 词典的长度，就是句子里的词有几种。在这里是base pair有几种。口算应该是(4+8+12)*2=48种。
    max_steps = 24 # 句子长度。21bpGRNA+3bpPAM = 24。
    embed_size = 6 # 句子里每个词的向量长度。base pair的编码后的向量的长度，编码前是我设计的长度6的向量，编码后是embed_size。
    embeddings = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
    test_data1 = [[
        [1,0,0,0,0,0],[0,0,0,1,0,0],[0,1,0,0,0,0],[0,1,0,1,0,0],
        [1,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],
        [0,1,0,0,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0],[0,1,0,1,0,0],
        [0,0,0,0,0,1],[0,0,0,1,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],
        [0,0,0,0,0,1],[0,0,0,1,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0],
        [0,0,0,1,0,0],[0,1,0,1,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0]
    ]]
    test_data2 = [[
        1,1,44,22,
        23,1,21,16,
        37,44,17,29,
        1,1,44,22,
        2,3,4,5,
        6,7,0,47
    ]] # 这个嵌入是给每个数字都分配一个output_dim长度的向量，不是给向量分配，所以test_data1变成1*24*6*6，而test_data2就是1*24*6
    test_data = np.array(test_data2, dtype=np.int32)
    encoder_embeddings = embeddings(test_data)
    print(encoder_embeddings) 
    # [ 0.019984    0.04981184 -0.01986524  0.00656086  0.01347306   -0.04285581] 
    # [ 0.019984    0.04981184 -0.01986524  0.00656086  0.01347306   -0.04285581]
    # 上面两行是test_data2里前两个1，后续的也是对应的都是
    positional_encoding = PositionalEncoding(max_steps=max_steps, max_dims=embed_size)
    print(positional_encoding.positional_embedding)
    # [ 0.          1.          0.          1.          0.    1.        ]
    # [ 0.84147096  0.5403023   0.04639922  0.998923    0.00215443    0.9999977 ]
    encoder_in = positional_encoding(encoder_embeddings)
    print(encoder_in)
    # [ 0.019984    1.0498118  -0.01986524  1.0065608   0.01347306    0.9571442 ]
    # [ 0.86145496  0.5901141   0.02653399  1.0054839   0.01562749    0.9571419 ]

def __test3():
    # 这是位置编码在机器学习实战第二版的例子，但是我需要的是只有一个输入的，
    # 而不是有encoder_inputs和decoder_inputs两个，看一下Transformer4keras的，他只有一个输入
    vocab_size = 10000 # 词典的长度，就是句子里的词有几种。在这里是base pair有几种。口算应该是(4+8+12)*2=48种。
    max_steps = 500 # 句子长度。21bpGRNA+3bpPAM = 24。
    embed_size = 512 # 句子里每个词的向量长度。base pair的编码后的向量的长度，编码前是我设计的长度6的向量，编码后是embed_size。
    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    embeddings = keras.layers.Embedding(vocab_size, embed_size)
    encoder_embeddings = embeddings(encoder_inputs)
    decoder_embeddings = embeddings(decoder_inputs)
    positional_encoding = PositionalEncoding(max_steps=max_steps, max_dims=embed_size)
    encoder_in = positional_encoding(encoder_embeddings)
    decoder_in = positional_encoding(decoder_embeddings)

    Z = encoder_in
    for N in range(6):
        Z = keras.layers.Attention(use_scale=True)([Z, Z])

    encoder_outputs = Z
    Z = decoder_in
    for N in range(6):
        Z = keras.layers.Attention(use_scale=True, causal=True)([Z, Z])
        Z = keras.layers.Attention(use_scale=True)([Z, encoder_outputs])

    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(vocab_size, activation="softmax"))(Z)

if __name__ == "__main__":
    # test1()
    __test2()
