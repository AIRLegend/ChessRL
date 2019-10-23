from player import Player
from game import Game
from simpolicies import SimulationPolicy


class Simulation(object):
    """ Simulation of an agent agaisnt a oponent (other agent or an algorithm
    like Stockfish)
    """

    def __init__(self, agent: Player, oponent: Player, game: Game,
                 agent_policy: SimulationPolicy):
        """ Parameters:
            agent: Player. Our agent.
            oponent: Player. Other agent which serves as oponent in the game
            environment.
            game: Game. Game environment
            agent_policy: SimulationPolicy. Move selection strategy for the
            agent (for example, random selection.)
        """
        self.agent = agent
        self.oponent = oponent
        self.game = game
        self.agent_policy = agent_policy

    def run(self, max_moves=50):
        """ Runs the game simulation for max_moves """
        n_mov = 0

        # If is the oponents turn, let him move
        if self.game.turn is not self.agent.color:
            n_mov = 1
            self.game.move(self.oponent.best_move(self.game))
            self.game.switch_turn()

        # While the game is not finished and the nb of moves is low
        while n_mov < max_moves and self.game.get_result() is None:
            # Our agent turn
            # self.agent.best_move(self.game)
            best_move_policy = self.agent_policy.best_movement(self.agent,
                                                               self.game)
            self.game.move(best_move_policy)
            self.game.switch_turn()

            # Check if in last turn we won of got to draw
            if self.game.get_result() is None:
                self.game.move(self.oponent.best_move(self.game))
                self.game.switch_turn()
                n_mov += 1

            input()
            self.game.plot_board()

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
