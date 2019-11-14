import chess

from game import Game
from agent import Agent


class GameAgent(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self,
                 agent,
                 player_color=True, #TODO: Game.WHITE
                 board=None,
                 date=None):
        super().__init__(board=board, player_color=player_color, date=date)

        if isinstance(agent, Agent):
            self.agent = agent
        elif type(agent) == str:
            self.agent = Agent(not player_color, weights=agent)
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
            agents_best_move = self.agent.best_move(self, real_game=True)
            self.board.push(chess.Move.from_uci(agents_best_move))
        else:
            made_movement = super().move(movement)
            if made_movement and self.get_result() is None:
                agents_best_move = self.agent.best_move(self, real_game=True)
                self.board.push(chess.Move.from_uci(agents_best_move))

    def get_copy(self):
        return GameAgent(board=self.board.copy(), agent=self.agent,
                         player_color=self.player_color)

    def tearup(self):
        """ Free resources."""
        del(self.agent)
