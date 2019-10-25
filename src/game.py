import chess
import chess.svg
import numpy as np
import cairosvg

from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt

from stockfish import Stockfish


class Game(object):

    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def _switch_turn(self):
        """ Used to force the turn change. This is not necessary.
        TODO: Maybe remove it
        """
        self.board.turn = True if self.board.turn else False

    def move(self, movement):
        """ Makes a move.
        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        Returns:
            success: boolean. Whether the move could be executed
        """
        # This is to prevent python-chess to put a illegal move in the
        # move stack before launching the exception
        success = False
        if movement in self.get_legal_moves():
            self.board.push(chess.Move.from_uci(movement))
            success = True
        return success

    def get_legal_moves(self, final_states=False):
        """ Gets a list of legal moves in the current turn.
        Parameters:
            final_states: bool. Whether copies of the board after executing
            each legal movement are returned.
        """
        moves = [m.uci() for m in self.board.legal_moves]
        if final_states:
            states = []
            for m in moves:
                gi = self.get_copy()
                gi.move(m)
                states.append(gi)
            moves = (moves, states)
        return moves

    @property
    def turn(self):
        """ Returns whether is white turn."""
        return self.board.turn

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

    def plot_board(self, save_path=None):
        """ Plots the current state of the board. This is useful for debug/log
        purposes while working outside a notebook

        Parameters:
            save_path: str, where to save the image. None if you want to plot
            on the screen only
        """
        svg = chess.svg.board(self.board)
        out = BytesIO()
        cairosvg.svg2png(svg, write_to=out)
        image = Image.open(out)
        if save_path is None:
            # image.show()
            plt.imshow(image)
            plt.axis('off')
            # plt.show()
            plt.draw()
        else:
            image.save(save_path)

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
            history[i * 14: (i + 1) * 14, :, :] =\
                Game.get_current_game_state(board_copy)

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


class GameStockfish(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self, stockfish,
                 stockfish_color=chess.BLACK, board=None):
        super().__init__(board=board)
        if stockfish is None:
            raise ValueError('A Stockfish object or a path is needed.')

        self.stockfish = stockfish

        if type(stockfish) == str:
            self.stockfish = Stockfish(stockfish_color, stockfish)
        elif type(stockfish) == Stockfish:
            self.stockfish = stockfish

    def move(self, movement):
        """ Makes a move.
        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        """
        if super().move(movement):
            print(f"Turn after player: {self.turn}")

            # stockfish move
            # TODO: Should test if stockfish gave up
            stockfish_best_move = self.stockfish.best_move(self)
            self.board.push(chess.Move.from_uci(stockfish_best_move))
            print(f"Turn after stockfish: {self.turn}")
        else:
            print("Wrong movement. Ignored")

    def get_copy(self):
        return GameStockfish(board=self.board.copy(), stockfish=self.stockfish)

    def tearup(self):
        """ Free resources. This cannot be done in __del__ as the instances
        will be intensivily cloned but maintaining the same stockfish AI
        engine. We don't want it deleted. Should only be called on the end of
        the program.
        """
        self.stockfish.quit()
