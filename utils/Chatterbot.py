from pathlib import Path
from keras.models import Model
from utils import Encoder, Decoder
from keras.layers import Input, Embedding
from tensorflow.keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Chatterbot:
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 batch_size: int = 64, epochs: int = 100):

        self.epochs = epochs
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        """

        :param encoder_input_data:
        :param decoder_input_data:
        :param decoder_target_data:
        :return:
        """
        enc_inputs = Input(shape=(None, self.input_vocab_size), name="Encoder_Inputs")
        dec_inputs = Input(shape=(None, self.num_decoder_tokens), name="Decoder_Inputs")

        enc_output, state_h, state_c = self.encoder(enc_inputs)
        dec_outputs = self.decoder([enc_output, state_h, state_c])

        model = Model(inputs=[enc_inputs, dec_inputs], outputs=dec_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_cross_entropy', metrics=['loss', 'accuracy'])

        plot_model(model, to_file=Path('./model/model.png'))

        model.fit([encoder_input_data, decoder_input_data],
                  decoder_target_data, batch_size=self.batch_size, epochs=self.epochs)

        model.save(Path('./model/chatBot.hdf5'))

    @staticmethod
    def preprocessing():
        """

        :return:
        """
        q_a = {}
        texts = []

        for i in ['preguntas', 'respuestas']:
            with open(f'./corpus/{i}.txt', 'r') as file:
                for line in file:
                    texts.append(line)

                q_a[str(i)] = texts
                texts = []
        file.close()

        tokenizer = Tokenizer(num_words=vocab_size)
        encoder_sequences = tokenizer.texts_to_sequences(q_a['preguntas'])
        decoder_sequences = tokenizer.texts_to_sequences(q_a['respuestas'])

        encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_len, dtype='int32', padding='post', truncating='post')
        decoder_input_data = pad_sequences(decoder_sequences, maxlen=max_len, dtype='int32', padding='post', truncating='post')
