import chess
import chess.svg
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
            with plt.style.context("seaborn-dark"):
                fig, ax = plt.subplots(num="game")
                ax.imshow(image)
                ax.axis('off')
                fig.canvas.draw_idle()
        else:
            image.save(save_path)


class GameStockfish(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self, stockfish,
                 stockfish_color=chess.BLACK, board=None):
        super().__init__(board=board)
        if stockfish is None:
            raise ValueError('A Stockfish object or a path is needed.')

        self.stockfish = stockfish

        if type(stockfish) == str:
            self.stockfish = Stockfish(stockfish_color, stockfish,
                                       thinking_time=0.01)
        elif type(stockfish) == Stockfish:
            self.stockfish = stockfish

    def move(self, movement):
        """ Makes a move. If it's not your turn, Stockfish will play and if
    the move is illegal, it will be ignored.
        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        """
        # If stockfish moves first
        if self.stockfish.color and len(self.board.move_stack) == 0:
            stockfish_best_move = self.stockfish.best_move(self)
            self.board.push(chess.Move.from_uci(stockfish_best_move))
        else:
            if super().move(movement):
                # stockfish move
                # TODO: Should test if stockfish gave up
                stockfish_best_move = self.stockfish.best_move(self)
                self.board.push(chess.Move.from_uci(stockfish_best_move))

    def get_copy(self):
        return GameStockfish(board=self.board.copy(), stockfish=self.stockfish)

    def tearup(self):
        """ Free resources. This cannot be done in __del__ as the instances
        will be intensivily cloned but maintaining the same stockfish AI
        engine. We don't want it deleted. Should only be called on the end of
        the program.
        """
        self.stockfish.kill()
