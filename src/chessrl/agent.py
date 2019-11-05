import mctree
import numpy as np
import netencoder

from player import Player
from model import ChessModel


class Agent(Player):
    """ AI agent which will play thess.

    Parameters:
        model: Model. Model encapsulating the neural network operations.
        move_encodings: list. List of all possible uci movements.
        uci_dict: dict. Dictionary with mappings 'uci'-> int. It's used
        to predict the policy only over the legal movements.
    """
    def __init__(self, color):
        super().__init__(color)

        self.model = ChessModel(compile_model=True)
        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

        # TODO: DEBUG purposes
        self.current_tree = None

    def best_move(self, game:'Game') -> str:  # noqa: E0602, F821
        """ Finds and returns the best possible move (UCI encoded)

        Parameters:
            game: Game. Current game before the move of this agent is made.
        """
        best_move = None
        if game.get_result() is None:
            self.current_tree = mctree.Tree(game)
            best_move = self.current_tree.select(self, 800, verbose=False)
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

    def train(self, dataset: netencoder.DatasetGame, epochs=2, logdir=None):
        """ Trains the model using previous recorded games """
        if len(dataset) > 0:
            datagen = netencoder.DataGameSequence(dataset, batch_size=1)
            self.model.train_generator(datagen, epochs=epochs, logdir=logdir)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)