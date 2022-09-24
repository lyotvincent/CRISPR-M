from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Sequential

def FeedForwardNetwork(units_dim, model_dim):
    return Sequential([Dense(units_dim, activation='relu'),Dense(model_dim)])

def add_encoder_layer(input_tensor, num_heads, key_dim, units_dim, model_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_tensor, input_tensor)
    attention_output = Dropout(rate=0.2)(attention_output, training=True)
    out1 = LayerNormalization()(input_tensor + attention_output)

    ffn = FeedForwardNetwork(units_dim=units_dim, model_dim=model_dim)
    ffn_output = ffn(out1)
    ffn_output = Dropout(rate=0.2)(ffn_output, training=True)
    out2 = LayerNormalization()(out1 + ffn_output)
    out3 = Dropout(rate=0.2)(out2)
    return out3

def add_decoder_layer(input_tensor, encoder_output, num_heads, key_dim, units_dim, model_dim):
    attention_output_1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_tensor, input_tensor)
    attention_output_1 = Dropout(rate=0.2)(attention_output_1, training=True)
    out1 = LayerNormalization()(input_tensor + attention_output_1)

    attention_output_2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(encoder_output, encoder_output, out1)
    attention_output_2 = Dropout(rate=0.2)(attention_output_2, training=True)
    out2 = LayerNormalization()(out1 + attention_output_2)

    ffn = FeedForwardNetwork(units_dim=units_dim, model_dim=model_dim)
    ffn_output = ffn(out2)
    ffn_output = Dropout(rate=0.2)(ffn_output, training=True)
    out3 = LayerNormalization()(out2 + ffn_output)
    out4 = Dropout(rate=0.2)(out3)
    return out4