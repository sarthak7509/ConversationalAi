import tensorflow as tf
import matplotlib.pyplot as plt


# creating postional encoding for the layer to get the word postion
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, postion, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.postional_encoding(postion, d_model)

    def get_angles(self, postion, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, dtype=tf.float32))
        return postion * angles

    def postional_encoding(self, postion, d_model):
        angle_rads = self.get_angles(
            postion=tf.range(postion, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


if __name__ == '__main__':
    sample_pos_encoding = PositionalEncoding(50,512)
    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()