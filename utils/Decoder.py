from keras.models import Model
from keras.layers import LSTM, Dense


class Decoder(Model):
    def __init__(self, num_decoder_tokens: int, latent_dim: int = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.num_decoder_tokens = num_decoder_tokens

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        enc_outputs, state_h, state_c = inputs
        enc_states = [state_h, state_c]

        lstm_0 = LSTM(self.latent_dim, return_sequences=True, name="Decoder_LSTM_0")(enc_outputs)
        lstm_1 = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="Decoder_LSTM_1")

        outputs, _, _ = lstm_1(lstm_0, initial_state=enc_states)
        outputs = Dense(self.num_decoder_tokens, activation='softmax', name="Decoder_Dense")(outputs)

        return outputs

    def get_config(self):
        pass
