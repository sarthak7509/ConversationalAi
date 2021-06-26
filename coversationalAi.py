import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from Transformer import TransformerModel
import pickle
import re


class conversationalApi:
    def __init__(self, tokenizer, model_weight_path):
        super(conversationalApi, self).__init__()
        self.tokenizer = tokenizer
        self.model_weight_path = model_weight_path
        self.transformer = TransformerModel()
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2
        self.MAXLENGTH = 40
        self.NUM_LAYERS = 2
        self.D_MODEL = 256
        self.NUM_HEADS = 8
        self.UNITS = 512
        self.DROPOUT = 0.1
        print("[+]model loading")
        self.model = self.transformer.transformer(self.VOCAB_SIZE, self.NUM_LAYERS, self.UNITS, self.D_MODEL, self.NUM_HEADS,
                                     self.DROPOUT)
        self.model.load_weights(self.model_weight_path)
        print("[+]model loaded")

    def predict(self, sentence):
        predictions = self._evaluate(sentence)
        predicted_sentence = self.tokenizer.decode(
            [i for i in predictions if i < self.tokenizer.vocab_size]
        )
        print(f'Input: {sentence}')
        print(f"Outputs: {predicted_sentence}")
        return predicted_sentence

    def _evaluate(self, sentence):
        sentence = self._preprocess_sentence(sentence)
        sentence = tf.expand_dims(self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)
        output = tf.expand_dims(self.START_TOKEN, axis=0)
        for i in range(self.MAXLENGTH):
            prediction = self.model(inputs=[sentence, output], training=False)

            # selecting the last word from the seq_length
            prediction = prediction[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)

    # helper method to preprocess data removing unwanted or unkown symbol inshot applying regex opertation
    def _preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        # sentence = '<start> ' + sentence + ' <end>'
        return sentence


if __name__ == '__main__':
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    model_path = 'model_weight/model3'
    chatbot = conversationalApi(tokenizer=tokenizer, model_weight_path=model_path)

    # test base
    sentence1 = chatbot.predict("I am not crazy, my mother had me tested")

