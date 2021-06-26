import tensorflow as tf
from encoder import Encoder
from decoder import Decoder


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class TransformerModel:
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def transformer(self,vocab_size,num_layers,units,d_model,num_heads,dropout,name="transformer"):
        inputs = tf.keras.Input(shape=(None,),name="inputs")
        dec_input = tf.keras.Input(shape=(None,),name="dec_input")

        enc_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_input)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_output = self.encoder.encoder(vocab_size,num_layers,units,d_model,num_heads,dropout)(inputs=[inputs,enc_padding_mask])
        dec_output = self.decoder.decoder(vocab_size,num_layers,units,d_model,num_heads,dropout)(inputs=[dec_input,enc_output,look_ahead_mask,dec_padding_mask])

        outputs = tf.keras.layers.Dense(vocab_size,name="outputs")(dec_output)

        return tf.keras.Model(inputs=[inputs,dec_input],outputs=outputs,name=name)

if __name__ == '__main__':
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1
    VOCAB_SIZE = 8192
    transformer = TransformerModel()

    model = transformer.transformer(vocab_size=VOCAB_SIZE,num_layers=NUM_LAYERS,units=UNITS,d_model=D_MODEL,num_heads=NUM_HEADS,dropout=DROPOUT)
    tf.keras.utils.plot_model(
        model, to_file='model-graphs/Transformer.png', show_shapes=True)