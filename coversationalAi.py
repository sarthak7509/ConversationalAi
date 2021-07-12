import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from Transformer import TransformerModel
import pickle
import re
from modelHelpers import transformer
from modelHelpers import CustomSchedule
from modelHelpers import loss_function,accuracy
from modelHelpers import predict


class conversationalApi:
    def __init__(self, tokenizer, model_weight_path,MAXLENGTH,NUM_LAYERS,D_MODEL,NUM_HEADS,UNITS,DROPOUT):
        super(conversationalApi, self).__init__()
        self.tokenizer = tokenizer
        self.model_weight_path = model_weight_path
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2
        self.MAXLENGTH = MAXLENGTH
        self.NUM_LAYERS = NUM_LAYERS
        self.D_MODEL = D_MODEL
        self.NUM_HEADS = NUM_HEADS
        self.UNITS = UNITS
        self.DROPOUT = DROPOUT
        self.learning_rate  = CustomSchedule(self.D_MODEL)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        print("[+]model loading")
        self.model = transformer(self.VOCAB_SIZE, self.NUM_LAYERS, self.UNITS, self.D_MODEL, self.NUM_HEADS,
                                     self.DROPOUT)
        self.model.load_weights(self.model_weight_path)
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=[accuracy])
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
    #To use WkiiQNa change path to wikiiQna tokenizer pickle file
    with open('Wikiqatokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    #choosing hyperparameter
    NUM_LAYERS=2
    D_MODEL=256
    NUM_HEADS=8
    UNITS = 512
    DROPOUT=0.1

    #TO use wkiiQna replace it with wiikiQna weight path the weights are available on drive
    model_path = 'model_weight_WkiiQNa\model3.h5'
    chatbot = conversationalApi(tokenizer=tokenizer, model_weight_path=model_path,MAXLENGTH=100,NUM_LAYERS=NUM_LAYERS,D_MODEL=D_MODEL,NUM_HEADS=NUM_HEADS,UNITS=UNITS,DROPOUT=DROPOUT)

    # test base
    sentence1 = chatbot.predict("What did movie theaters do for sound before synchronized sound was introduced into film")

