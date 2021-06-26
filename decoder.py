import tensorflow as tf
from postional import PositionalEncoding
from attentionlayer import MultiHeadAttention
import graphviz
import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


class Decoder:
    def decoder(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_output")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
        for i in range(num_layers):
            outputs = self._decoderlayer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name='decoder_layer_{}'.format(i),
            )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def _decoderlayer(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")

        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(
            d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
        attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)


if __name__ == '__main__':
    sample_decoder_layer = Decoder()._decoderlayer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_decoder_layer"
    )
    tf.keras.utils.plot_model(
        sample_decoder_layer, to_file='model-graphs/decoder_layer.png', show_shapes=True)

    sample_decoder = Decoder().decoder(vocab_size=8192,
                                       num_layers=2,
                                       units=512,
                                       d_model=128,
                                       num_heads=4,
                                       dropout=0.3,
                                       name="sample_decoder")
    tf.keras.utils.plot_model(
        sample_decoder, to_file='model-graphs/decoder.png', show_shapes=True)
