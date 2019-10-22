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
        self.board.turn = True if self.board.turn else False

    def move(self, movement):
        self.board.push(chess.Move.from_uci(movement))
        # self.switch_turn() # TODO: Maybe remove this and call switch turn
        # manually.

    def get_legal_moves(self, final_states=False):
        """ Gets a list of legal moves in the current turn """
        # moves = list(self.board.legal_moves)
        moves = [m.uci() for m in self.board.legal_moves]
        if final_states:
            states = []
            for m in moves:
                gi = self.get_copy()
                gi.move(m)
                states.append(gi)
            moves = (moves, states)
        return moves

    def get_copy(self):
        return Game(board=self.board.copy())

    def get_result(self):
        """ Returns the result of the game for the white pieces. None if the
        game is not over
        """
        result = None
        if self.board.is_game_over():
            r = self.board.result()
            if r == "1-0":
                result = 1  # Whites win
            elif r == "0-1":
                result = -1  # Whites dont win
            else:
                result = 0  # Draw
        return result

    @staticmethod
    def get_pieces_one_hot(board, color=False):
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
            mask[i, :, :] = np.array(board.pieces(i, color).mirror()
                                     .tolist()).reshape(8, 8)
        # Encode blank positions
        mask[0, :, :] = (~np.array(mask.sum(axis=0), dtype=bool)).astype(int)
        return mask

    @staticmethod
    def get_current_game_state(board):
        """ This method returns the matrix representation of a game turn
        (positions of the pieces of the two colors)

        Returns:
            current: numpy array. 3D Matrix with dimensions 14x8x8.
        """

        return np.concatenate((Game.get_pieces_one_hot(board, color=False),
                               Game.get_pieces_one_hot(board, color=True)),
                              axis=0)

    @staticmethod
    def get_game_history(board, T=8):
        board_copy = board.copy()
        history = np.zeros((14 * T, 8, 8))

        for i in range(T):
            try:
                board_copy.pop()
            except IndexError:
                break
            history[i * 14: (i + 1) * 14, :, :] = Game.get_current_game_state(
                board_copy)
        return history

    @staticmethod
    def get_game_state(board):
        """ This method returns the matrix representation of a game with its
        history of moves.

        Returns:
            current: numpy array. 3D Matrix with dimensions (14T)x8x8. Where T
            corresponds to the number of backward turns in time. (And the 14 is
            the current representation of the two players) """

        current = Game.get_current_game_state(board)
        history = Game.get_game_history(board)
        current = np.concatenate((current, history), axis=0)
        return current
