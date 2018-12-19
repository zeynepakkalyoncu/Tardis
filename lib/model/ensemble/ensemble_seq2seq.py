import numpy as np
from keras.initializers import RandomUniform
from keras.layers import Input, LSTM, GRU, Embedding, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects

from lib.model.ensemble.util import EncoderSlice, DecoderSlice
from lib.model.metrics import bleu_score

class EnsembleSeq2Seq:
    def __init__(self, config):
        self.config = config
        recurrent_unit = self.config.recurrent_unit.lower()
        get_custom_objects().update({'EncoderSlice': EncoderSlice, 'DecoderSlice': DecoderSlice})

        initial_weights = RandomUniform(minval=-0.08, maxval=0.08, seed=config.seed)
        stacked_input = Input(shape=(None,))

        # encoder_input = Lambda(lambda x: x[:, config.input_split_index:])(stacked_input)
        encoder_input = EncoderSlice(config.input_split_index)(stacked_input)
        encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim,
                                      weights=[config.source_embedding_map],
                                      trainable=False)
        encoder_embedded = encoder_embedding(encoder_input)

        if recurrent_unit == 'lstm':
            encoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True,
                           recurrent_initializer=initial_weights)(encoder_embedded)
            for i in range(1, self.config.num_encoder_layers):
                encoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(encoder)
            _, state_h, state_c = encoder
            encoder_states = [state_h, state_c]
        else:
            encoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True,
                          recurrent_initializer=initial_weights)(encoder_embedded)
            for i in range(1, self.config.num_encoder_layers):
                encoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(encoder)
            _, state_h = encoder
            encoder_states = [state_h]

        # decoder_input = Lambda(lambda x: x[:, config.input_split_index:])(stacked_input)
        decoder_input = DecoderSlice(config.input_split_index)(stacked_input)
        decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim,
                                      weights=[config.target_embedding_map],
                                      trainable=False)
        decoder_embedded = decoder_embedding(decoder_input)

        if recurrent_unit.lower() == 'lstm':
            decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
            for i in range(1, self.config.num_decoder_layers):
                decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
            decoder_output, decoder_state = decoder[0], decoder[1:]
        else:
            decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
            for i in range(1, self.config.num_decoder_layers):
                decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
            decoder_output, decoder_state = decoder[0], decoder[1]

        decoder_dense = Dense(config.target_vocab_size, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        self.output = decoder_output

        self.model = Model(stacked_input, decoder_output)
        optimizer = Adam(lr=config.lr, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())
