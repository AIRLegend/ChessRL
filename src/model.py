"""
This module contains the neural network model used by the agent to make
decisions.
"""

import tensorflow as tf

from tensorflow import keras as k


class Model(object):

    def __init__(self, compile_model=True):
        """
        Creates the model. This code builds a ResNet that will act as both
        the policy and value network (see AlphaZero paper for more info).
        """
        inp = k.layers.Input((8, 8, 126))

        x = k.layers.Conv2D(filters=5, kernel_size=5, padding='same',
                            kernel_regularizer='l2')(inp)

        for i in range(3):
            x = self.__res_block(x)

        # Policy Head
        pol_head = k.layers.Conv2D(filters=2, kernel_size=1, padding='valid',
                                   kernel_regularizer='l2')(x)
        pol_head = k.layers.BatchNormalization(axis=-1)(pol_head)
        pol_head = k.layers.Activation("relu")(pol_head)
        pol_head = k.layers.Flatten()(pol_head)
        pol_head = k.layers.Dense(1698, kernel_regularizer='l2',
                                  activation='softmax',
                                  name='policy_out')(pol_head)

        # Value Head
        val_head = k.layers.Conv2D(filters=4, kernel_size=1, padding='valid',
                                   kernel_regularizer='l2')(x)
        val_head = k.layers.BatchNormalization(axis=-1)(val_head)
        val_head = k.layers.Activation("relu")(val_head)
        val_head = k.layers.Flatten()(val_head)
        val_head = k.layers.Dense(1, kernel_regularizer='l2',
                                  activation='tanh',
                                  name='value_out')(val_head)

        self.model = k.Model(inp, [pol_head, val_head], name='chessnet')

        if compile_model:
            self.model.compile(k.optimizers.Adam(lr=0.002), loss=self.__loss)

    def predict(self, inp):
        return self.model.predict(inp)

    def load_weights(self, path):
        pass

    def save_weights(self, path):
        pass

    def train(self, game_state, game_outcome, next_action):
        pass

    def __loss(self, y_true, y_pred):
        policy_pred, val_pred = y_pred
        policy_true, val_true = y_true

        return k.losses.MSE(val_true, val_pred) - \
            k.losses.categorical_crossentropy(policy_true, policy_pred)

    def __res_block(self, block_input):
        """ Builds a residual block """
        x = k.layers.Conv2D(filters=5, kernel_size=3, padding="same",
                            kernel_regularizer='l2')(block_input)
        x = k.layers.BatchNormalization(axis=-1)(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Conv2D(filters=5, kernel_size=3, padding="same",
                            kernel_regularizer='l2')(x)
        x = k.layers.BatchNormalization(axis=-1)(x)
        x = k.layers.Add()([block_input, x])
        x = k.layers.Activation("relu")(x)
        return x
