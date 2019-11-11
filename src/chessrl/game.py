import chess
import chess.svg
import cairosvg

from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

from .stockfish import Stockfish
from .agent import Agent


class Game(object):

    NULL_MOVE = chess.Move.from_uci('0000')

    def __init__(self, board=None, player_color=chess.WHITE, date=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.player_color = player_color

        self.date = date
        if self.date is None:
            self.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

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

    def get_history(self):
        moves = [x.uci() for x in self.board.move_stack]
        res = self.get_result()
        return {'moves': moves,
                'result': res,
                'player_color': self.player_color,
                'date': self.date}
        return moves

    def get_fen(self):
        return self.board.board_fen()

    def set_fen(self, fen):
        self.board.set_board_fen(fen)

    @property
    def turn(self):
        """ Returns whether is white turn."""
        return self.board.turn

    def get_copy(self):
        return Game(board=self.board.copy())

    def reset(self):
        self.board.reset()

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

    def __len__(self):
        return len(self.board.move_stack)

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
            plt.ion()
            with plt.style.context("seaborn-dark"):
                fig, ax = plt.subplots(num="game")
                ax.imshow(image)
                ax.axis('off')
                plt.draw()
                # plt.show()
        else:
            image.save(save_path)


class GameStockfish(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self, stockfish,
                 player_color=chess.WHITE,
                 board=None,
                 date=None):
        super().__init__(board=board, player_color=player_color, date=date)
        if stockfish is None:
            raise ValueError('A Stockfish object or a path is needed.')

        self.stockfish = stockfish
        stockfish_color = not self.player_color

        if type(stockfish) == str:
            self.stockfish = Stockfish(stockfish_color, stockfish,
                                       #thinking_time=0.005,
                                       search_depth=5)
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
            made_movement = super().move(movement)
            if made_movement and self.get_result() is None:
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


class GameAgent(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self,
                 agent,
                 player_color=chess.WHITE,
                 board=None,
                 date=None):
        super().__init__(board=board, player_color=player_color, date=date)

        if isinstance(agent, Agent):
            self.agent = Agent
        elif type(agent) == str:
            self.agent = Agent(color=not player_color)
            self.agent.load(agent)  # Load its weights
        else:
            raise ValueError("An agent or path to the agents "
                             "weights (.h5) is needed")

    def move(self, movement):
        """ Makes a move. If it's not your turn, the agent will play and if
        the move is illegal, it will be ignored.

        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        """
        # If agent moves first (whites and first move)
        if self.agent.color and len(self.board.move_stack) == 0:
            agents_best_move = self.agent.best_move(self)
            self.board.push(chess.Move.from_uci(agents_best_move))
        else:
            made_movement = super().move(movement)
            if made_movement and self.get_result() is None:
                agents_best_move = self.agent.best_move(self)
                self.board.push(chess.Move.from_uci(agents_best_move))

    def get_copy(self):
        # TODO
        return GameAgent(board=self.board.copy(), agent=self.agent,
                         player_color=self.player_color)

    def tearup(self):
        # TODO
        """ Free resources.         """
        pass
