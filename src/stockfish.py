import chess
import chess.engine

from player import Player


class Stockfish(Player):
    """ AI using Stockfish to play a game of chess."""

    def __init__(self, color: bool, binary_path: str,
                 thinking_time=0.04):
        super().__init__(color)
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        self.thinking_time = thinking_time

    def best_move(self, game: 'Game'):  # noqa: E0602, F821
        result = self.engine.play(game.board,
                                  chess.engine.Limit(time=self.thinking_time))
        return result.move.uci()

    def kill(self):
        self.engine.quit()
