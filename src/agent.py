import netencoder
import mctree
import numpy as np

from player import Player
from model import Model


class Agent(Player):

    def __init__(self, color):
        super().__init__(color)

        self.model = Model(compile_model=False)
        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

        # DEBUG purposes
        self.current_tree = None

    def best_move(self, game:'Game') -> str:  # noqa: E0602, F821
        best_move = None
        if game.get_result is None:
            self.current_tree = mctree.Tree(game)
            best_move = self.current_tree.select(self, 10)
        return best_move

    def predict_outcome(self, game:'Game') -> float:  # noqa: E0602, F821
        game_matr = netencoder.get_game_state(game)
        return self.model.predict(np.expand_dims(game_matr, axis=0))[1][0][0]

    def predict_policy(self, game:'Game', mask_legal_moves=True) -> float:  # noqa: E0602, F821
        game_matr = netencoder.get_game_state(game)
        policy = self.model.predict(np.expand_dims(game_matr, axis=0))[0][0]
        if mask_legal_moves:
            legal_moves = game.get_legal_moves()
            policy = [policy[self.uci_dict[x]] for x in legal_moves]
        return policy

    def train(self, dataset: DatasetGame):
        """ Trains the model using previous recorded games """
        pass


