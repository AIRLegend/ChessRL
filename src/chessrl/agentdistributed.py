import numpy as np

import mctree
import netencoder

from player import Player

from multiprocessing.connection import Client


class AgentDistributed(Player):
    """ AI agent which will play chess. This is a parallel friendly version
    of the Agent. The only difference is that this agent sends the data to
    a worker process which is responsible of making the predictions. Based on
    that, it builds a pool of connections to the worker and uses a parallelized
    version of the Monte Carlo Tree Search.

    Attributes:
        move_encodings: list. List of all possible uci movements.
        uci_dict: dict. Dictionary with mappings 'uci'-> int. It's used
        to predict the policy only over the legal movements.
        endpoint: (str, int). Tuple with the address, port of the PredictWorker
        num_threads: int. Number of threads to use during MTCS
        conn: Connection. Connection to the prediction worker that is in use
        pool_conns: list[Connection]. Pool of connections that will be used
        during MCTS.
    """
    def __init__(self, color, endpoint=None, num_threads=6):
        super().__init__(color)

        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

        self.conn = None
        self.pool_conns = None
        self.address = endpoint
        self.num_threads = num_threads

    def best_move(self, game:'Game', real_game=False, max_iters=900,  # noqa: E0602, F821
                  ai_move=True, verbose=False) -> str:
        """ Finds and returns the best possible move (UCI encoded)

        Parameters:
            game: Game. Current game before the move of this agent is made.
            real_game: Whether to use MCTS or only the neural network (self
            play vs playing in a real environment).
            max_iters: if not playing a real game, the max number of iterations
            of the MCTS algorithm.
            verbose: Whether to print debug info
            ai_move: bool. Whether to return the next move of the AI

        Returns:
            str. UCI encoded movement.
        """
        best_move = '00000'  # Null move
        if real_game:
            policy = self.predict_policy(game)
            best_move = game.get_legal_moves()[np.argmax(policy)]
        else:
            if game.get_result() is None:
                current_tree = mctree.SelfPlayTree(
                    game,
                    threads=self.num_threads)
                best_move = current_tree.search_move(self, max_iters=max_iters,
                                                     verbose=verbose,
                                                     ai_move=ai_move)

        return best_move

    def predict_outcome(self, game:'Game') -> float:  # noqa: E0602, F821
        """ Predicts the outcome of a game from the current position """
        response = self.__send_game(game)
        return response[1]

    def predict_policy(self, game:'Game', mask_legal_moves=True) -> float:  # noqa: E0602, F821
        """ Predict the policy distribution over all possible moves. """
        response = self.__send_game(game)
        policy = response[0]

        if mask_legal_moves:
            legal_moves = game.get_legal_moves()
            policy = [policy[self.uci_dict[x]] for x in legal_moves]
        return policy

    def predict(self, game:'Game'):  # noqa: E0602, F821
        """ Predicts from a game board and returns policy / value"""
        response = self.__send_game(game)
        return response

    def __send_game(self, game:'Game'):  # noqa: E0602, F821
        """ Sends a game to the neural net. and blocks the caller thread until
        the prediction is done.

        Parameters:
            game: Game. Game to send. to worker.
        """
        self.conn.send(game)
        response = self.conn.recv()
        return response

    def get_copy(self):
        """ Returns an empty agent with the color of this one """
        copy = AgentDistributed(self.color, endpoint=self.address)
        return copy

    def connect(self):
        self.conn = Client(self.address)

    def disconnect(self):
        self.conn.close()
        self.conn = None
