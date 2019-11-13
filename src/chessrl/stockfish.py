import chess
import chess.engine

from player import Player


class Stockfish(Player):
    """ AI using Stockfish to play a game of chess."""

    def __init__(self, color: bool, binary_path: str,
                 thinking_time=0.01,
                 search_depth=5):
        super().__init__(color)
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)

        self.thinking_time = thinking_time
        self.search_depth = search_depth

    def best_move(self, game: 'Game'):  # noqa: E0602, F821
        # Page 77 of
        # http://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf
        # gives some study about the relation of search depth vs ELO.
        result = self.engine.play(game.board,
                                  # chess.engine.Limit(time=self.thinking_time)
                                  chess.engine.Limit(depth=5)
                                  )
        return result.move.uci()

    def kill(self):
        self.engine.quit()
