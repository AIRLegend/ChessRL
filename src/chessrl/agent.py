import numpy as np

import mctree
import netencoder

from player import Player
from model import ChessModel
from dataset import DatasetGame


class Agent(Player):
    """ AI agent which will play thess.

    Parameters:
        model: Model. Model encapsulating the neural network operations.
        move_encodings: list. List of all possible uci movements.
        uci_dict: dict. Dictionary with mappings 'uci'-> int. It's used
        to predict the policy only over the legal movements.
    """
    def __init__(self, color, weights=None):
        super().__init__(color)

        self.model = ChessModel(compile_model=True, weights=weights)
        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

    def best_move(self, game:'Game', real_game=False, max_iters=900) -> str:  # noqa: E0602, F821
        """ Finds and returns the best possible move (UCI encoded)

        Parameters:
            game: Game. Current game before the move of this agent is made.
            real_game: Whether to use MCTS or only the neural network (self
            play vs playing in a real environment).
            max_iters: if not playing a real game, the max number of iterations
            of the MCTS algorithm.

        Returns:
            str. UCI encoded movement.
        """
        best_move = '00000'  # Null move
        if real_game:
            policy = self.predict_policy(game)
            best_move = game.get_legal_moves()[np.argmax(policy)]
        else:
            if game.get_result() is None:
                current_tree = mctree.Tree(game)
                best_move = current_tree.search_move(self, max_iters=max_iters)
        return best_move

    def predict_outcome(self, game:'Game') -> float:  # noqa: E0602, F821
        """ Predicts the outcome of a game from the current position """
        game_matr = netencoder.get_game_state(game)
        return self.model.predict(np.expand_dims(game_matr, axis=0))[1][0][0]

    def predict_policy(self, game:'Game', mask_legal_moves=True) -> float:  # noqa: E0602, F821
        """ Predict the policy distribution over all possible moves. """
        game_matr = netencoder.get_game_state(game)
        policy = self.model.predict(np.expand_dims(game_matr, axis=0))[0][0]
        if mask_legal_moves:
            legal_moves = game.get_legal_moves()
            policy = [policy[self.uci_dict[x]] for x in legal_moves]
        return policy

    def train(self, dataset: DatasetGame,
              epochs=1, logdir=None, batch_size=1,
              validation_split=0):
        """ Trains the model using previous recorded games """
        if len(dataset) <= 0:
            return

        if validation_split > 0:
            split_point = len(dataset) - int(validation_split * len(dataset))

            games_train = DatasetGame(dataset[:split_point])
            games_val = DatasetGame(dataset[split_point:])
            val_gen = netencoder.DataGameSequence(games_val,
                                                  batch_size=batch_size)
        else:
            games_train = dataset
            val_gen = None

        train_gen = netencoder.DataGameSequence(games_train,
                                                batch_size=batch_size,
                                                random_flips=.1)

        self.model.train_generator(train_gen,
                                   epochs=epochs,
                                   logdir=logdir,
                                   val_gen=val_gen)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
