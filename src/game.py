import chess
import numpy as np


class Game(object):

    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def switch_turn(self):
        """ Next player turn."""
        # self.board.push(chess.Move.null())
        # self.board.turn = not self.board.turn
        self.board.turn = True if self.board.turn else False

    def get_legal_moves(self):
        """ Gets a list of legal moves in the current turn """
        return list(self.board.legal_moves)

    def get_copy(self):
        return Game(board=self.board.copy())

    @staticmethod
    def get_pieces_one_hot(board, color=False):
        """ Returns a 3D-matrix representation of the pieces for one color.
        The matrix ins constructed as follows:
            8x8 (Chess board) x 7 possible pieces (including empty). = 448.

        Parameters:
            board: Python Chess board. Current board
            color: Boolean, True for white, False for black
        Returns:
            mask: numpy array, 3D matrix with the pieces of the player.
        """
        mask = np.zeros((7, 8, 8))
        for i in list(chess.PIECE_TYPES):
            mask[i, :, :] = np.array(board.pieces(i, color).mirror()
                                     .tolist()).reshape(8, 8)
        # Encode blank positions
        mask[0, :, :] = (~np.array(mask.sum(axis=0), dtype=bool)).astype(int)
        return mask

    @staticmethod
    def get_current_game_state(board):
        """ This method returns the matrix representation of a game turn
        (positions of the pieces of the two colors)

        Params:
            board: Python Chess board. Current board
        Returns:
            current: numpy array. 3D Matrix with dimensions 14x8x8.
        """

        return np.concatenate((Game.get_pieces_one_hot(board, color=False),
                               Game.get_pieces_one_hot(board, color=True)),
                              axis=0)

    @staticmethod
    def get_game_history(board, T=8):
        """ Returns matrix representation of the last T states

        Params:
            board: Python Chess board. Current board with moves in the stack of
            moves

        Returns:
            history: numpy array. 3D Matrix with dimensions (14*T)x8x8.
        """
        board_copy = board.copy()
        history = np.zeros((14 * T, 8, 8))

        for i in range(T):
            try:
                board_copy.pop()
            except IndexError:
                break
            history[i * 14:(i + 1) * 14, :, :] = Game.get_current_game_state(
                board_copy)

        return history

    @staticmethod
    def get_game_state(board):
        """ This method returns the matrix representation of a game with its
        history of moves.

        Parameters:
            board: Python Chess board. Current board with the previous moves in
            the stack of moves.

        Returns:
            current: numpy array. 3D Matrix with dimensions (14T + 1)x8x8.
            Where T corresponds to the number of backward turns in time. (And
            the 14 is the current representation of the two players)
        """

        current = Game.get_current_game_state(board)
        history = Game.get_game_history(board)
        current = np.concatenate((current, history), axis=0)
        return current
