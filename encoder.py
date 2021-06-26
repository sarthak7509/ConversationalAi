import tensorflow as tf
from postional import PositionalEncoding
from attentionlayer import MultiHeadAttention
import graphviz
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


class Encoder:
    def encoder(self,vocab_size,num_layers,units,d_model,num_heads,dropout,name='encoder'):
        inputs = tf.keras.Input(shape=(None,),name="inputs")
        padding_mask = tf.keras.Input(shape=(1,1,None),name="padding_mask")

        embedding = tf.keras.layers.Embedding(vocab_size,d_model)(inputs)
        embedding *= tf.math.sqrt(tf.cast(d_model,tf.float32))
        embedding = PositionalEncoding(vocab_size,d_model)(embedding)
        
        outputs = tf.keras.layers.Dropout(rate=dropout)(embedding)
        for i in range(num_layers):
            outputs = self._encoderlayer(units,d_model,num_heads,dropout,name=f"encoder_layer_{i}")([outputs,padding_mask])
        return tf.keras.Model(inputs=[inputs,padding_mask],outputs=outputs,name=name)
        
    def _encoderlayer(self,units, d_model, num_heads, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None,d_model),name="inputs")
        padding_mask = tf.keras.Input(shape=(1,1,None),name="padding_mask")

        attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads ,name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
        attention = tf.keras.layers.Dropout(rate=0.3)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs+attention)

        outputs = tf.keras.layers.Dense(units,activation='relu')(attention)
        outputs = tf.keras.layers.Dense(d_model)(outputs)
        outputs = tf.keras.layers.Dropout(dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        return tf.keras.Model(inputs=[inputs,padding_mask],outputs=outputs,name=name)
if __name__ == '__main__':
    sample_encoder_layer = Encoder()._encoderlayer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder_layer"
    )
    tf.keras.utils.plot_model(
        sample_encoder_layer, to_file='model-graphs/encoder_LAYER.png', show_shapes=True)
    sample_encoder = Encoder().encoder(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder"
    )
    tf.keras.utils.plot_model(
        sample_encoder, to_file='model-graphs/encoder.png', show_shapes=True)
