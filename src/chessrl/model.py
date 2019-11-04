"""
This module contains the neural network model used by the agent to make
decisions.
"""

from tensorflow.keras.layers import (Dense, Conv2D, BatchNormalization,
                                     Activation, Flatten, Input, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE, categorical_crossentropy
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard


class ChessModel(object):

    def __init__(self, compile_model=True, weights=None):
        """
        Creates the model. This code builds a ResNet that will act as both
        the policy and value network (see AlphaZero paper for more info).
        """
        inp = Input((8, 8, 127))

        x = Conv2D(filters=5, kernel_size=5, padding='same',
                   kernel_regularizer='l2')(inp)

        for i in range(4):
            x = self.__res_block(x)

        # Policy Head
        pol_head = Conv2D(filters=2, kernel_size=1, padding='valid',
                          kernel_regularizer='l2')(x)
        pol_head = BatchNormalization(axis=-1)(pol_head)
        pol_head = Activation("relu")(pol_head)
        pol_head = Flatten()(pol_head)
        pol_head = Dense(1968, kernel_regularizer='l2', activation='softmax',
                         name='policy_out')(pol_head)

        # Value Head
        val_head = Conv2D(filters=4, kernel_size=1, padding='valid',
                          kernel_regularizer='l2')(x)
        val_head = BatchNormalization(axis=-1)(val_head)
        val_head = Activation("relu")(val_head)
        val_head = Flatten()(val_head)
        val_head = Dense(1, kernel_regularizer='l2', activation='tanh',
                         name='value_out')(val_head)

        self.model = Model(inp, [pol_head, val_head])

        if weights:
            self._load_weights(weights)

        if compile_model:
            self.model.compile(Adam(lr=0.002),
                               loss=['categorical_crossentropy',
                                     'mean_squared_error'])

    def predict(self, inp):
        return self.model.predict(inp)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)

    def train(self, game_state, game_outcome, next_action):
        # TODO
        pass

    def train_generator(self, generator, epochs=5, logdir=None):
        callbacks = []
        if logdir is not None:
            tensorboard_callback = TensorBoard(log_dir=logdir,
                                               histogram_freq=0,
                                               batch_size=1,
                                               write_graph=True,
                                               update_freq='epoch')
            callbacks.append(tensorboard_callback)

        self.model.fit_generator(generator, epochs=epochs,
                                 callbacks=callbacks)

    def __loss(self, y_true, y_pred):
        policy_pred, val_pred = y_pred[0], y_pred[1]
        policy_true, val_true = y_true[0], y_true[1]

        return MSE(val_true, val_pred) - \
            categorical_crossentropy(policy_true, policy_pred)

    def __res_block(self, block_input):
        """ Builds a residual block """
        x = Conv2D(filters=5, kernel_size=3, padding="same",
                   kernel_regularizer='l2')(block_input)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=5, kernel_size=3, padding="same",
                   kernel_regularizer='l2')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([block_input, x])
        x = Activation("relu")(x)
        return x
