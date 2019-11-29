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
    def __init__(self, color, worker=None):
        super().__init__(color)

        self.worker = worker
        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

        #self.pool_pipes = [worker.get_pipe() for _ in range(4)]

    def best_move(self, game:'Game', real_game=False, max_iters=900, verbose=False) -> str:  # noqa: E0602, F821
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
                best_move = current_tree.search_move(self, max_iters=max_iters, verbose=verbose)
        return best_move

    def predict_outcome(self, game:'Game') -> float:  # noqa: E0602, F821
        """ Predicts the outcome of a game from the current position """
        # game_matr = netencoder.get_game_state(game)
        #self.model.predict(np.expand_dims(game_matr, axis=0))[1][0][0]
        response = self.__send_game(game)
        return response[1]

    def predict_policy(self, game:'Game', mask_legal_moves=True) -> float:  # noqa: E0602, F821
        """ Predict the policy distribution over all possible moves. """
        #game_matr = netencoder.get_game_state(game)
        #policy = self.model.predict(np.expand_dims(game_matr, axis=0))[0][0]
        response = self.__send_game(game)
        policy = response[0]

        if mask_legal_moves:
            legal_moves = game.get_legal_moves()
            policy = [policy[self.uci_dict[x]] for x in legal_moves]
        return policy

    def __send_game(self, game:'Game'):  # noqa: E0602, F821
        #pipe = self.pool_pipes.pop()
        pipe = self.pipe
        pipe.send(game)
        response = self.pipe.recv()
        #self.pool_pipes.append(pipe)
        return response

    def get_copy(self):
        return Agent(self.color)
