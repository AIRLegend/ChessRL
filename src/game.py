import chess
import numpy as np


class Game(object):

    def __init__(self, board=None):
        if self.board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def switch_turn(self):
        """ Next player turn."""
        self.board.push(chess.Move.null())

    def get_legal_moves(self):
        """ Gets a list of legal moves in the current turn """
        return list(self.board.legal_moves)

    def get_copy(self):
        return Game(board=self.board.copy())

    def get_pieces_one_hot(self, color=False):
        """ Returns a 3D-matrix representation of the pieces for one color.
        The matrix ins constructed as follows:
            8x8 (Chess board) x 7 possible pieces (including empty). = 448.

        Parameters:
            color: Boolean, True for white, False for black
        Returns:
            mask: numpy array, 3D matrix with the pieces of the player.
        """
        mask = np.zeros((7, 8, 8))
        for i in list(chess.PIECE_TYPES):
            mask[i, :, :] = np.array(self.board.pieces(i, color).mirror()
                                     .tolist()).reshape(8, 8)
        # Encode blank positions
        mask[0, :, :] = (~np.array(mask.sum(axis=0), dtype=bool)).astype(int)
        return mask

    def get_game_history(self):
        # TODO: Extract from the board queue the last T=8 moves
        raise NotImplementedError()

    def get_game_state(self):
        """ This method returns the matrix representation of a game with its
        history of moves.

        Returns:
            current: numpy array. 3D Matrix with dimensions (14T)x8x8. Where T
            corresponds to the number of backward turns in time. (And the 14 is
            the current representation of the two players) """

        current = np.concatenate((self.get_pieces_one_hot(color=False),
                                  self.get_pieces_one_hot(color=True)), axis=0)

        # TODO: Concat the history to current
        return current
