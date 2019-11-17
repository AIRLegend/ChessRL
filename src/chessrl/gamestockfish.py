import chess
from game import Game
from stockfish import Stockfish


class GameStockfish(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self, stockfish,
                 player_color=Game.WHITE,
                 board=None,
                 date=None,
                 stockfish_depth=10):
        super().__init__(board=board, player_color=player_color, date=date)
        if stockfish is None:
            raise ValueError('A Stockfish object or a path is needed.')

        self.stockfish = stockfish
        stockfish_color = not self.player_color

        if type(stockfish) == str:
            self.stockfish = Stockfish(stockfish_color, stockfish,
                                       search_depth=stockfish_depth)
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

    def free(self):
        """ Unlinks the game from the stockfish engine. """
        self.stockfish = None

    def __del__(self):
        self.free()
        del(self.board)
