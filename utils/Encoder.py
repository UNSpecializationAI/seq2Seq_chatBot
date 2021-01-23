from keras.models import Model
from keras.layers import LSTM


class Encoder(Model):
    def __init__(self, num_encoder_tokens: int, latent_dim: int = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        encoder = LSTM(self.latent_dim, return_sequences=True, name="Encoder_LSTM_0")(inputs)
        encoder = LSTM(self.latent_dim, return_state=True, name="Encoder_LSTM_1")(encoder)

        return encoder

    def get_config(self):
        pass
