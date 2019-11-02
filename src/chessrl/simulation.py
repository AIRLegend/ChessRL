from player import Player
from game import Game
from simpolicies import SimulationPolicy


class Simulation(object):
    """ Simulation of an agent agaisnt Other oponent """

    def __init__(self, agent: Player, game: Game,
                 agent_policy: SimulationPolicy):
        """ Parameters:
            agent: Player. Our agent.
            agent_policy: SimulationPolicy. Move selection strategy for the
            agent (for example, random selection.)
            stockfish_bin_path: str, PATH to the stockfish engine.
        """
        self.agent = agent
        self.game = game
        self.agent_policy = agent_policy

    def run(self, max_moves=50):
        """ Runs the game simulation for max_moves """
        n_mov = 0

        # If it's the oponent's turn...
        if self.game.turn is not self.agent.color:
            self.game.move(None)
            n_mov += 1

        # While the game is not finished and the nb of moves is low
        while n_mov < max_moves and self.game.get_result() is None:
            # Our agent turn
            # self.agent.best_move(self.game)
            best_move_policy = self.agent_policy.best_movement(self.agent,
                                                               self.game)
            self.game.move(best_move_policy)  # The oponent also moves here
            n_mov += 2

            # Check if in last turn we won of got to draw
            #  if self.game.get_result() is None:
            #      self.game.move(self.oponent.best_move(self.game))
            #      self.game.switch_turn()
            #      n_mov += 1

        result = self.game.get_result()

        if result is not None:
            result = result if self.agent.color else result * -1
        else:
            result = 0  # Max nb of moves = draw.
        return result

    @property
    def simulation_result(self):
        """ Returns the simulation outcome.

        Return:
            -1 if Blacks win
            1 if Whites win
            0 if Draw
            None if the simulation is not finished.
        """
        return self.game.get_result()
